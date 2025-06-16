# File: modules/video_analysis/face_detection.py

import logging
import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        """
        Initializes the FaceDetector with MediaPipe Face Detection and Face Mesh models.
        """
        logger.info("Initializing FaceDetector.")
        try:
            # MediaPipe Face Detection model
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, # 0 for short-range (faces < 2m), 1 for full-range (faces up to 5m)
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe Face Detection initialized.")

            # MediaPipe Face Mesh model for landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1, # Assuming one face per analysis for simplicity
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Face Mesh initialized.")

            self.mp_drawing = mp.solutions.drawing_utils # Utility for drawing landmarks

        except Exception as e:
            logger.error(f"Error initializing FaceDetector models: {e}", exc_info=True)
            self.face_detection = None
            self.face_mesh = None

    def detect_faces(self, frame: np.ndarray):
        """
        Detects faces in a video frame.

        Args:
            frame: A numpy array representing the video frame (BGR format).

        Returns:
            A list of detected faces. Each face is represented by the detection
            object from MediaPipe, which includes bounding box and key points.
            Returns an empty list if no faces are detected or if the detector
            was not initialized.
        """
        if self.face_detection is None:
            logger.warning("Face Detection model not initialized.")
            return []

        # Convert the BGR frame to RGB as MediaPipe models expect RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find faces
        results = self.face_detection.process(frame_rgb)

        if results.detections:
            logger.debug(f"Detected {len(results.detections)} face(s).")
            return results.detections
        else:
            logger.debug("No faces detected.")
            return []

    def extract_landmarks(self, frame: np.ndarray, detection):
        """
        Extracts facial landmarks for a detected face.

        Args:
            frame: A numpy array representing the video frame (BGR format).
            detection: The MediaPipe face detection object for the face.

        Returns:
            A list of facial landmarks (MediaPipe NormalizedLandmark objects)
            or None if landmarks cannot be extracted.
        """
        if self.face_mesh is None:
            logger.warning("Face Mesh model not initialized.")
            return None

        # MediaPipe Face Mesh works best on the whole image, not just the cropped face.
        # It uses the detection result to refine the landmark detection.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        # The Face Mesh model returns multi_face_landmarks, which is a list of
        # landmark sets, one for each detected face (up to max_num_faces).
        # Since we set max_num_faces=1 in __init__, we expect at most one set.
        if results.multi_face_landmarks:
            # Assuming we only process the first detected face for simplicity
            landmarks = results.multi_face_landmarks[0].landmark
            logger.debug(f"Extracted {len(landmarks)} facial landmarks.")
            return landmarks
        else:
            logger.debug("Could not extract facial landmarks.")
            return None

    def draw_landmarks(self, frame: np.ndarray, landmarks):
        """
        Draws facial landmarks on the frame for visualization.

        Args:
            frame: The video frame (numpy array, BGR format).
            landmarks: A list of facial landmarks (MediaPipe NormalizedLandmark objects).

        Returns:
            The frame with landmarks drawn.
        """
        if landmarks and self.mp_drawing:
            # Create a drawing spec for the landmarks
            drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

            # Create a dummy structure similar to the output of face_mesh.process
            # to use the drawing_utils
            class LandmarkList:
                def __init__(self, landmark):
                    self.landmark = landmark

            dummy_landmark_list = LandmarkList(landmarks)

            # Draw the landmarks
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=dummy_landmark_list,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION, # Or other connections like FACEMESH_CONTOURS
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
        return frame


# Example Usage (for testing)
if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()

    detector = FaceDetector()

    # Create a dummy black image for testing
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "No Face Here", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    print("\nTesting face detection on a dummy frame (no face):")
    detections_no_face = detector.detect_faces(dummy_frame)
    print(f"Detected {len(detections_no_face)} faces.")

    # Note: To properly test face and landmark detection, you would typically
    # use an image file with a face or a webcam feed.
    # Example using a placeholder image path (replace with a real image path if testing)
    # dummy_image_path = "path/to/an/image_with_face.jpg"
    # if os.path.exists(dummy_image_path):
    #     print(f"\nTesting face detection on image: {dummy_image_path}")
    #     image_with_face = cv2.imread(dummy_image_path)
    #     if image_with_face is not None:
    #         detections_with_face = detector.detect_faces(image_with_face)
    #         print(f"Detected {len(detections_with_face)} faces.")
    #         if detections_with_face:
    #             print("Extracting landmarks for the first detected face:")
    #             landmarks = detector.extract_landmarks(image_with_face, detections_with_face[0])
    #             if landmarks:
    #                 print(f"Extracted {len(landmarks)} landmarks.")
    #                 # You could optionally draw and display the image here
    #                 # frame_with_landmarks = detector.draw_landmarks(image_with_face.copy(), landmarks)
    #                 # cv2.imshow("Face and Landmarks", frame_with_landmarks)
    #                 # cv2.waitKey(0)
    #                 # cv2.destroyAllWindows()
    #             else:
    #                 print("Landmark extraction failed.")
    #     else:
    #         print(f"Could not read image file: {dummy_image_path}")
    # else:
    #     print(f"\nSkipping image test: Dummy image not found at {dummy_image_path}")

    # Example of processing a video file (requires a video file)
    # dummy_video_path = "path/to/a/video_with_face.mp4"
    # if os.path.exists(dummy_video_path):
    #      print(f"\nTesting face and landmark detection on video: {dummy_video_path}")
    #      cap = cv2.VideoCapture(dummy_video_path)
    #      if cap.isOpened():
    #          ret, frame = cap.read()
    #          if ret:
    #              print("Processing first frame...")
    #              detections = detector.detect_faces(frame)
    #              if detections:
    #                  print(f"Detected {len(detections)} face(s) in the first frame.")
    #                  landmarks = detector.extract_landmarks(frame, detections[0])
    #                  if landmarks:
    #                      print(f"Extracted {len(landmarks)} landmarks from the first frame.")
    #                      # frame_with_landmarks = detector.draw_landmarks(frame.copy(), landmarks)
    #                      # cv2.imshow("First Frame with Landmarks", frame_with_landmarks)
    #                      # cv2.waitKey(0)
    #                      # cv2.destroyAllWindows()
    #                  else:
    #                      print("Landmark extraction failed for the first frame.")
    #              else:
    #                  print("No faces detected in the first frame.")
    #          else:
    #              print("Could not read the first frame of the video.")
    #          cap.release()
    #      else:
    #          print(f"Error opening video file: {dummy_video_path}")
    # else:
    #      print(f"\nSkipping video test: Dummy video not found at {dummy_video_path}")
