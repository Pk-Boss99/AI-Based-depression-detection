# File: modules/video_analysis/emotion_recognition.py

import logging
import numpy as np
import cv2
# Ensure deepface is installed: pip install deepface
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    logging.error("DeepFace library not found. Please install it (`pip install deepface`). Emotion recognition will be disabled.")
    DEEPFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

class EmotionRecognizer:
    def __init__(self):
        """
        Initializes the EmotionRecognizer.
        Uses DeepFace for emotion analysis. DeepFace handles model loading internally.
        """
        logger.info("Initializing EmotionRecognizer.")
        self.model = None
        if DEEPFACE_AVAILABLE:
            try:
                # DeepFace can analyze emotions directly.
                # We don't explicitly load a model here, DeepFace does it on first use.
                # You can specify models if needed, e.g., models = ["Emotion"]
                logger.info("DeepFace available for emotion recognition.")
                # Optional: Force model download/load on init (can be slow)
                # DeepFace.build_model("Emotion")
                # logger.info("DeepFace Emotion model built.")

            except Exception as e:
                logger.error(f"Error initializing DeepFace: {e}", exc_info=True)
                # DeepFace might fail if dependencies are missing or models can't be downloaded
                logger.warning("DeepFace initialization failed. Emotion recognition will not be available.")
                DEEPFACE_AVAILABLE = False
        else:
             logger.warning("DeepFace not available. Emotion recognition will not be performed.")


    def recognize_emotions(self, frame: np.ndarray, face_location=None) -> dict:
        """
        Recognizes emotions in a detected face within a video frame.

        Args:
            frame: A numpy array representing the video frame (BGR format).
            face_location: Optional. A tuple (x, y, w, h) representing the
                           bounding box of the face. If None, DeepFace will
                           attempt to detect faces in the frame.

        Returns:
            A dictionary of emotion probabilities (e.g., {'sad': 0.1, 'happy': 0.9, ...}),
            or an empty dictionary if no face is detected or analysis fails.
        """
        if not DEEPFACE_AVAILABLE:
            logger.warning("DeepFace not available. Cannot perform emotion recognition.")
            return {}

        if frame is None:
            logger.warning("No frame provided for emotion recognition.")
            return {}

        # DeepFace expects RGB input if not using its internal detector,
        # but its analyze function handles BGR input from OpenCV correctly.
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Not strictly necessary for analyze()

        try:
            # DeepFace.analyze can take a frame (numpy array) directly.
            # It can also take a list of bounding boxes if you want to use
            # your own face detector results.
            # Setting enforce_detection=False allows it to proceed even if
            # its internal detector fails, assuming you provide face_location.
            # If face_location is None, DeepFace will use its own detector.

            analysis_results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'], # Specify the analysis task
                enforce_detection=False if face_location else True, # Enforce detection if no location provided
                detector_backend='opencv', # Use OpenCV detector (or 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe')
                silent=True # Suppress verbose output from DeepFace
            )

            # DeepFace.analyze returns a list of results, one for each detected face.
            # If enforce_detection=False and no face is found, it might return [].
            # If enforce_detection=True and no face is found, it will raise an exception.

            if analysis_results:
                # Assuming we are interested in the first detected face's emotions
                # If you used face_location, DeepFace might still return multiple results
                # if its detector finds more faces. You might need to match results
                # to your provided face_location if analyzing multiple faces.
                # For simplicity, we take the first result's emotion data.
                first_face_result = analysis_results[0]
                emotions = first_face_result.get('emotion', {})
                logger.debug(f"Emotion recognition successful. Top emotion: {first_face_result.get('dominant_emotion')}")
                return emotions # Returns a dictionary like {'sad': 0.1, 'happy': 0.9, ...}
            else:
                logger.debug("No face detected by DeepFace for emotion analysis.")
                return {}

        except Exception as e:
            # DeepFace can raise exceptions for various reasons (no face, model issues)
            logger.error(f"An error occurred during DeepFace emotion analysis: {e}", exc_info=True)
            return {} # Return empty dict on error

# Example Usage (for testing)
if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()

    recognizer = EmotionRecognizer()

    # Create a dummy image with a face for testing
    # You would typically load a real image file here
    dummy_image_path = "dummy_face_image.jpg"
    # To make this example runnable, let's create a simple dummy image
    # Note: DeepFace needs an actual face to analyze. A blank image won't work.
    # You'll need to replace this with a path to a real image containing a face.
    # For demonstration, we'll just create a blank image and note that analysis will fail.
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "Replace with image with face", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.imwrite(dummy_image_path, dummy_frame) # Don't save blank image for DeepFace test

    print("\nTesting emotion recognition on a dummy frame (no face):")
    # This will likely fail or return empty results as there's no face
    emotions_dummy = recognizer.recognize_emotions(dummy_frame)
    print(f"Emotions detected (dummy frame): {emotions_dummy}")

    # --- To properly test, replace with a path to a real image with a face ---
    # real_image_path = "path/to/your/image_with_face.jpg"
    # if os.path.exists(real_image_path):
    #     print(f"\nTesting emotion recognition on image: {real_image_path}")
    #     real_frame = cv2.imread(real_image_path)
    #     if real_frame is not None:
    #         emotions_real = recognizer.recognize_emotions(real_frame)
    #         print(f"Emotions detected (real image): {emotions_real}")
    #     else:
    #         print(f"Could not read image file: {real_image_path}")
    # else:
    #     print(f"\nSkipping real image test: Image not found at {real_image_path}")

    # Example with None input
    print("\nTesting emotion recognition with None input:")
    emotions_none = recognizer.recognize_emotions(None)
    print(f"Emotions detected (None input): {emotions_none}")
