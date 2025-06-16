# Multimodal Depression Detection System

This project evaluates the effectiveness of deep learning models in detecting depression using multimodal inputs: text, images, audio, and video. Built with TensorFlow, OpenCV, and Streamlit, the system allows users to input data and receive a real-time depression risk analysis.

---

## Project Objective

To explore how deep learning can predict depression severity from diverse input types—emulating human mental health evaluations that consider facial expression, voice tone, written responses, and video behavior.

---

## Modalities Supported

| Modality | Description | Model Used |
|----------|-------------|------------|
| Text     | User answers to psychological questions | BERT + Dense |
| Image    | Facial emotion analysis from a photo     | CNN (FER2013-based) |
| Audio    | Speech emotion recognition via voice tone | MFCC + LSTM |
| Video    | Temporal facial emotion patterns from video | Frame-wise CNN or 3D-CNN |

---

## Tech Stack

- **Frontend:** Streamlit  
- **Backend & ML:** Python, TensorFlow, Keras  
- **Audio Processing:** Librosa  
- **Image/Video Processing:** OpenCV  
- **Text Embeddings:** HuggingFace Transformers (BERT)

---

## How To Run the Project

# Step 1: Clone the Repository
git clone https://github.com/yourusername/depression-detection-tf-opencv.git
cd depression-detection-tf-opencv

# Step 2: Create a Virtual Environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Step 3: Install All Required Python Packages
pip install -r requirements.txt

# Step 4: Run the Streamlit App
streamlit run app.py

# Project Srtructure 

   depression_detection/
│
├── app.py                    # Streamlit frontend
├── requirements.txt          # Python dependencies
├── model/                    # Pretrained model files (.h5)
│   ├── text_model.h5
│   ├── image_model.h5
│   ├── audio_model.h5
│   └── video_model.h5
├── utils/                    # Preprocessing scripts
│   ├── preprocess_text.py
│   ├── preprocess_image.py
│   ├── preprocess_audio.py
│   ├── preprocess_video.py
│   └── helper.py
├── data/                     # Sample inputs (optional)
│   └── sample_inputs/
│       ├── sample.txt
│       ├── sample.jpg
│       ├── sample.wav
│       └── sample.mp4
└── README.md

Future Improvements
Integrate with a clinical dataset for better validation

Add biometric sensor data (e.g., heart rate, EEG)

Use multimodal fusion transformers for higher accuracy

License
This project is licensed under the MIT License.

Author
Pushkar Kumar
AI/ML Research Enthusiast
Email: your-email@example.com

Acknowledgments
FER2013 Dataset

RAVDESS Audio Dataset

HuggingFace Transformers

