# File: modules/audio_analysis/feature_extraction.py

import logging
import numpy as np
import librosa
import soundfile as sf
import io # To handle file-like objects from Streamlit

from config import AUDIO_SAMPLE_RATE, AUDIO_DURATION, AUDIO_N_MFCC

logger = logging.getLogger(__name__)

def extract_audio_features(audio_input, sr: int = AUDIO_SAMPLE_RATE, duration: int = AUDIO_DURATION):
    """
    Extracts acoustic features from audio input.

    Args:
        audio_input: The audio data. Can be a file path (str),
                     a file-like object (e.g., from Streamlit uploader),
                     or raw audio data (numpy array).
        sr: The target sample rate.
        duration: The maximum duration of the audio segment to analyze in seconds.

    Returns:
        A numpy array containing the extracted features, or None if extraction fails.
    """
    y = None # audio time series
    loaded_sr = None # original sampling rate

    logger.info("Starting audio feature extraction.")

    try:
        if isinstance(audio_input, str):
            # Input is a file path
            logger.debug(f"Loading audio from file path: {audio_input}")
            y, loaded_sr = librosa.load(audio_input, sr=None, duration=duration) # Load with original sr first
        elif isinstance(audio_input, io.BytesIO):
             # Input is a file-like object (e.g., from Streamlit uploader)
             logger.debug("Loading audio from file-like object.")
             # soundfile can read directly from file-like objects
             with sf.SoundFile(audio_input) as sound_file:
                 y = sound_file.read(frames=int(duration * sound_file.samplerate), dtype='float32')
                 loaded_sr = sound_file.samplerate
        elif isinstance(audio_input, np.ndarray):
            # Input is raw audio data (numpy array) - assumes it's already at target sr
            logger.debug("Using raw audio data (numpy array).")
            y = audio_input
            loaded_sr = sr # Assume provided data matches target sr
        else:
            logger.error(f"Unsupported audio input type: {type(audio_input)}")
            return None

        if y is None or len(y) == 0:
            logger.warning("Audio data is empty after loading.")
            return None

        # Resample if necessary
        if loaded_sr is not None and loaded_sr != sr:
            logger.debug(f"Resampling audio from {loaded_sr} Hz to {sr} Hz.")
            y = librosa.resample(y=y, orig_sr=loaded_sr, target_sr=sr)

        # Trim audio to the specified duration if it's longer
        if len(y) > sr * duration:
             logger.debug(f"Trimming audio to {duration} seconds.")
             y = y[:sr * duration]

        logger.debug(f"Audio loaded successfully. Shape: {y.shape}, Sample Rate: {sr}")

        # --- Feature Extraction ---
        features = []

        # 1. MFCCs (Mel-Frequency Cepstral Coefficients)
        # MFCCs are commonly used in speech and audio analysis
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=AUDIO_N_MFCC)
        # Take the mean of MFCCs over time to get a single feature vector
        mfccs_mean = np.mean(mfccs.T, axis=0)
        features.extend(mfccs_mean)
        logger.debug(f"Extracted MFCCs (mean). Shape: {mfccs_mean.shape}")

        # 2. Pitch (Fundamental Frequency - F0)
        # Pitch can be indicative of emotional state
        # Using pyin for robust pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(y=y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C5'), sr=sr)
        # Handle NaNs in f0 (unvoiced frames) - replace with 0 or interpolate
        f0[~voiced_flag] = 0 # Set unvoiced frames to 0
        # Calculate mean and standard deviation of voiced pitch
        voiced_f0 = f0[voiced_flag]
        pitch_mean = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0.0
        pitch_std = np.std(voiced_f0) if len(voiced_f0) > 0 else 0.0
        features.extend([pitch_mean, pitch_std])
        logger.debug(f"Extracted Pitch (mean, std). Values: {pitch_mean:.2f}, {pitch_std:.2f}")

        # 3. Speaking Rate (Requires voice activity detection or external tool)
        # This is more complex and often requires a separate VAD step.
        # Placeholder: For simplicity, we'll skip speaking rate for now or
        # assume it's handled externally. If needed, integrate a VAD library.
        # Example: Use a simple energy-based VAD or a pre-trained model.
        # For now, we'll just add placeholders or skip.

        # 4. Jitter and Shimmer (Requires precise pitch/amplitude tracking, often done with tools like Praat)
        # These are measures of vocal perturbation. librosa doesn't directly provide these.
        # You would typically use a dedicated speech analysis library or tool.
        # Placeholder: Add dummy values or skip for now.
        # jitter = 0.0 # Dummy
        # shimmer = 0.0 # Dummy
        # features.extend([jitter, shimmer])
        # logger.debug(f"Added placeholder Jitter and Shimmer.")


        # 5. Other features (Optional but potentially useful)
        # - Chromagram: Represents the intensity of the 12 different pitch classes.
        # - Spectral Centroid: Indicates the "center of mass" of the spectrum.
        # - Spectral Bandwidth: Measures the spread of the spectrum around the spectral centroid.
        # - Spectral Contrast: Measures the difference in energy between peaks and valleys in the spectrum.
        # - Tonnetz: Represents the tonal harmony or consonance.
        # - Zero-Crossing Rate: Rate at which the signal changes sign (useful for distinguishing speech from noise).

        # Example: Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)
        features.append(cent_mean)
        logger.debug(f"Extracted Spectral Centroid (mean). Value: {cent_mean:.4f}")

        # Example: Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        features.append(zcr_mean)
        logger.debug(f"Extracted Zero-Crossing Rate (mean). Value: {zcr_mean:.4f}")


        # Combine all features into a single numpy array
        feature_vector = np.hstack(features)
        logger.info(f"Feature extraction complete. Total features: {feature_vector.shape[0]}")

        return feature_vector

    except Exception as e:
        logger.error(f"An error occurred during audio feature extraction: {e}", exc_info=True)
        return None

# Example Usage (for testing)
if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()

    # Create a dummy audio file for testing
    dummy_audio_path = "dummy_audio_features_test.wav"
    try:
        # Requires soundfile
        sr = AUDIO_SAMPLE_RATE
        duration = 5 # seconds
        frequency = 440 # Hz
        t = np.linspace(0., duration, int(sr * duration))
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        sf.write(dummy_audio_path, data.astype(np.int16), sr)
        print(f"Created dummy audio file: {dummy_audio_path}")

        # Extract features from the dummy audio file
        print(f"\nExtracting features from dummy audio file: {dummy_audio_path}")
        features = extract_audio_features(dummy_audio_path)

        if features is not None:
            print(f"Extracted features with shape: {features.shape}")
            # print(f"Features: {features[:10]}...") # Print first 10 features
        else:
            print("Feature extraction failed.")

    except ImportError:
        print("\n'soundfile' library not found. Cannot create dummy audio for testing.")
        print("Please install it (`pip install soundfile`) to test feature extraction.")
    except Exception as e:
        print(f"\nAn error occurred during dummy audio feature extraction test: {e}")
    finally:
        # Clean up the dummy file
        if os.path.exists(dummy_audio_path):
             # os.remove(dummy_audio_path) # Keep for inspection if needed
             pass # Keep the dummy file for now

    # Example with a dummy numpy array (assuming correct sample rate)
    print("\nExtracting features from dummy numpy array:")
    dummy_audio_data = np.random.randn(AUDIO_SAMPLE_RATE * 5) # 5 seconds of random noise
    features_np = extract_audio_features(dummy_audio_data, sr=AUDIO_SAMPLE_RATE, duration=5)
    if features_np is not None:
         print(f"Extracted features from numpy array with shape: {features_np.shape}")
    else:
         print("Feature extraction from numpy array failed.")

    # Example with None input
    print("\nExtracting features from None input:")
    features_none = extract_audio_features(None)
    if features_none is None:
        print("Feature extraction correctly returned None for None input.")
    else:
        print("Feature extraction from None input returned features unexpectedly.")
