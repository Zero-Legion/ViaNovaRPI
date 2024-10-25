import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
import hailo
from hailo_rpi_common import (
  get_caps_from_pad,
  get_numpy_from_buffer,
  app_callback_class,
)
from detection_pipeline import GStreamerDetectionApp
import math
import json
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
import threading

# -----------------------------------------------------------------------------------------------
# User-defined class to store and track objects with IDs and counts
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
  def __init__(self, connection_string, container_name):
    super().__init__()
    self.tracked_objects = {}
    self.next_id = 1
    self.total_car_count = 0
    self.total_person_count = 0
    self.detected_car_ids = set()
    self.detected_person_ids = set()
    self.tracking_threshold = 50
    self.connection_string = connection_string
    self.container_name = container_name
    self.upload_timestamp = None # Store the last upload timestamp

    # Initialize Azure Blob service client
    self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    self.detections = []
    self.lock = threading.Lock()
    self.start_timer()

  def start_timer(self):
    """Start a timer that triggers every 60 seconds to upload detections."""
    self.timer = threading.Timer(60.0, self.upload_detections)
    self.timer.start()

  def stop_timer(self):
    """Stop the running timer."""
    if self.timer:
      self.timer.cancel()

  def upload_detections(self):
    """Upload accumulated detections to Azure Blob Storage."""
    with self.lock:
      if self.detections:
        # Create a JSON object from the detections
        detection_data = {
          'timestamp': datetime.utcnow().isoformat(),
          'detections': self.detections
        }
        json_data = json.dumps(detection_data)

        # Upload the JSON data to Azure Blob Storage
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=f"detections_{datetime.utcnow().timestamp()}.json")
        blob_client.upload_blob(json_data)

        # Clear the detections list for the next interval
        self.detections = []

        # Update the upload timestamp
        self.upload_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Restart the timer for the next upload
    self.start_timer()

  def track_objects(self, detections):
    """Tracks objects and assigns unique IDs."""
    current_objects = {}
    current_detections = []

    for detection in detections:
      label = detection.get_label()
      bbox = detection.get_bbox()
      confidence = detection.get_confidence()

      bbox_x, bbox_y, bbox_w, bbox_h = self._get_bbox_coordinates(bbox)

      if label == "car":
        matched_id = self._match_existing_object(bbox_x, bbox_y, bbox_w, bbox_h, self.tracking_threshold)

        if matched_id is None:
          object_id = self.next_id
          self.next_id += 1
          if object_id not in self.detected_car_ids:
            self.total_car_count += 1
            self.detected_car_ids.add(object_id)
        else:
          object_id = matched_id

        current_objects[object_id] = (label, (bbox_x, bbox_y, bbox_w, bbox_h), confidence)

      elif label == "person":
        matched_id = self._match_existing_object(bbox_x, bbox_y, bbox_w, bbox_h, self.tracking_threshold)

        if matched_id is None:
          object_id = self.next_id
          self.next_id += 1
          if object_id not in self.detected_person_ids:
            self.total_person_count += 1
            self.detected_person_ids.add(object_id)
        else:
          object_id = matched_id

        current_objects[object_id] = (label, (bbox_x, bbox_y, bbox_w, bbox_h), confidence)

      current_detections.append({
        'object_id': object_id,
        'label': label,
        'bbox': {
          'x': bbox_x,
          'y': bbox_y,
          'w': bbox_w,
          'h': bbox_h
        },
        'confidence': confidence
      })

    with self.lock:
      self.detections.extend(current_detections)

    self.tracked_objects = current_objects
    return current_objects

  def _get_bbox_coordinates(self, bbox):
    """Safely extract bounding box coordinates."""
    try:
      bbox_x = bbox.x
      bbox_y = bbox.y
      bbox_w = bbox.w
      bbox_h = bbox.h
    except AttributeError:
      bbox_x = bbox.get_x() if hasattr(bbox, 'get_x') else 0
      bbox_y = bbox.get_y() if hasattr(bbox, 'get_y') else 0
      bbox_w = bbox.get_w() if hasattr(bbox, 'get_w') else 0
      bbox_h = bbox.get_h() if hasattr(bbox, 'get_h') else 0
     
    return bbox_x, bbox_y, bbox_w, bbox_h

  def _match_existing_object(self, bbox_x, bbox_y, bbox_w, bbox_h, threshold):
    """Check if the detected object matches any previously tracked objects."""
    for object_id, (_, (tracked_x, tracked_y, tracked_w, tracked_h), _) in self.tracked_objects.items():
      if self._is_close(bbox_x, bbox_y, bbox_w, bbox_h, tracked_x, tracked_y, tracked_w, tracked_h, threshold):
        return object_id
    return None

  def _is_close(self, x1, y1, w1, h1, x2, y2, w2, h2, threshold):
    """Check if two bounding boxes are close enough to be considered the same object."""
    center_x1, center_y1 = x1 + w1 / 2, y1 + h1 / 2
    center_x2, center_y2 = x2 + w2 / 2, y2 + h2 / 2
    distance = math.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)
    return distance < threshold

# -----------------------------------------------------------------------------------------------
# User-defined callback function for the pipeline
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
  buffer = info.get_buffer()
  if buffer is None:
    return Gst.PadProbeReturn.OK

  format, width, height = get_caps_from_pad(pad)

  roi = hailo.get_roi_from_buffer(buffer)
  detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

  tracked_objects = user_data.track_objects(detections)

  car_count = 0
  person_count = 0

  for object_id, (label, bbox, confidence) in tracked_objects.items():
    print(f"Object ID {object_id}: {label} with confidence {confidence:.2f}")
    if label == "car":
      car_count += 1
    elif label == "person":
      person_count += 1

  frame = None
  if user_data.use_frame and format and width and height:
    frame = get_numpy_from_buffer(buffer, format, width, height)

    for object_id, (label, (bbox_x, bbox_y, bbox_w, bbox_h), confidence) in tracked_objects.items():
      cv2.rectangle(frame, (int(bbox_x), int(bbox_y)), 
             (int(bbox_x + bbox_w), int(bbox_y + bbox_h)), 
             (0, 255, 0), 2)
      label_text = f"{label} ID: {object_id}"
      cv2.putText(frame, label_text, (int(bbox_x), int(bbox_y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.putText(frame, f"Car Detected: {car_count}", (10, 30),
          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Person Detected: {person_count}", (10, 70),
          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame, f"Total Cars Detected: {user_data.total_car_count}", (10, 110),
          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Persons Detected: {user_data.total_person_count}", (10, 150),
          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display upload notification if available
    if user_data.upload_timestamp:
      cv2.putText(frame, f"File uploaded: {user_data.upload_timestamp}", (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    user_data.set_frame(frame)

  return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Main code to run the app
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
  AZURE_CONNECTION_STRING = ""
  AZURE_CONTAINER_NAME = ""

  user_data = user_app_callback_class(AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME)
  app = GStreamerDetectionApp(app_callback, user_data)
  app.run()
