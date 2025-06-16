import os

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
MODELS_DIR = os.path.join(BASE_DIR, 'modules', 'models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Text Analysis Configuration ---
TEXT_MODEL_NAME = "bert-base-uncased" # Example: Use a pre-trained BERT model
TEXT_MAX_LENGTH = 128
TEXT_SENTIMENT_THRESHOLD = 0.5 # Example threshold for sentiment classification
TEXT_EMOTION_THRESHOLD = 0.5   # Example threshold for emotion classification

# --- Audio Analysis Configuration ---
AUDIO_SAMPLE_RATE = 22050
AUDIO_DURATION = 10 # seconds
AUDIO_N_MFCC = 40
AUDIO_MODEL_PATH = os.path.join(MODELS_DIR, 'audio_classifier.pkl') # Example path for a trained model

# --- Video Analysis Configuration ---
VIDEO_CAPTURE_DURATION = 10 # seconds
VIDEO_FACE_MODEL_PATH = os.path.join(MODELS_DIR, 'face_detection_model.pb') # Example path
VIDEO_EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, 'emotion_recognition_model.h5') # Example path

# --- Multimodal Fusion Configuration ---
FUSION_METHOD = "weighted_average" # Options: "majority_voting", "weighted_average", "neural_network"
FUSION_WEIGHTS = {"text": 0.4, "audio": 0.3, "video": 0.3} # Example weights for weighted average

# --- GUI Configuration ---
GUI_TITLE = "Multimodal Depression Detection System"
GUI_DESCRIPTION = "Analyze text, audio, and video for depression indicators."
GUI_STREAMLIT = True # Set to False if using Gradio

# --- Logging Configuration ---
LOG_LEVEL = "INFO" # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_FILE = os.path.join(LOGS_DIR, 'system.log')

# --- Output Configuration ---
RISK_SCORE_RANGE = (0, 100)
CONFIDENCE_INTERVAL_RANGE = (0, 1)

# --- Question Sets ---
# Define paths or variables for question sets
PHQ9_QUESTIONS_PATH = os.path.join(BASE_DIR, 'modules', 'text_analysis', 'question_sets.py')

# --- Error Handling ---
# Define specific error codes or messages if needed
