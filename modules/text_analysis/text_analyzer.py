# File: modules/text_analysis/text_analyzer.py

import logging
from transformers import pipeline, AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import TEXT_MODEL_NAME, TEXT_MAX_LENGTH, TEXT_SENTIMENT_THRESHOLD, TEXT_EMOTION_THRESHOLD

logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self):
        """
        Initializes the TextAnalyzer with pre-trained NLP models.
        Uses transformers pipelines for sentiment and zero-shot classification (for emotion).
        Uses a transformer model for generating embeddings for semantic similarity.
        """
        logger.info(f"Initializing TextAnalyzer with model: {TEXT_MODEL_NAME}")
        try:
            # Sentiment Analysis Pipeline
            # Using a general sentiment model. Could be fine-tuned later.
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english", # Example sentiment model
                tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("Sentiment analysis pipeline loaded.")

            # Emotion Detection using Zero-Shot Classification
            # This allows classifying text into arbitrary categories (emotions)
            self.emotion_analyzer = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli" # Example zero-shot model
            )
            self.emotion_labels = ["sadness", "joy", "anger", "fear", "disgust", "surprise", "neutral", "anxiety", "hopelessness", "apathy"] # Example emotion labels
            logger.info(f"Emotion analysis pipeline loaded with labels: {self.emotion_labels}")

            # Model for Semantic Similarity (Embeddings)
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
            self.model = AutoModel.from_pretrained(TEXT_MODEL_NAME)
            logger.info(f"Semantic similarity model and tokenizer loaded: {TEXT_MODEL_NAME}")

            # Reference texts representing depression indicators for semantic similarity
            self.depression_reference_texts = [
                "I feel sad and hopeless.",
                "I have lost interest in things I used to enjoy.",
                "I feel tired all the time.",
                "It's hard to concentrate.",
                "I feel worthless.",
                "I have trouble sleeping.",
                "I feel empty inside.",
                "Everything feels overwhelming.",
                "I don't see a future.",
                "I feel numb."
            ]
            self.depression_reference_embeddings = self._get_embeddings(self.depression_reference_texts)
            logger.info(f"Generated embeddings for {len(self.depression_reference_texts)} reference texts.")

        except Exception as e:
            logger.error(f"Error initializing TextAnalyzer models: {e}", exc_info=True)
            raise

    def _get_embeddings(self, texts):
        """Helper function to get embeddings for a list of texts."""
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=TEXT_MAX_LENGTH)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the mean of the last hidden states as the sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embeddings.numpy() # Convert to numpy array

    def analyze(self, text_responses: dict) -> float:
        """
        Analyzes text responses for depression indicators.

        Args:
            text_responses: A dictionary where keys are question identifiers
                            and values are the user's text responses.

        Returns:
            A float score between 0 and 1, representing the likelihood of
            depression based on text analysis. Higher score indicates higher likelihood.
        """
        if not text_responses:
            logger.warning("No text responses provided for analysis.")
            return 0.0

        sentiment_scores = []
        emotion_scores = {label: [] for label in self.emotion_labels}
        semantic_similarity_scores = []

        for q_id, response_text in text_responses.items():
            if not response_text or not response_text.strip():
                logger.debug(f"Skipping empty response for question: {q_id}")
                continue

            logger.debug(f"Analyzing text response for {q_id}: '{response_text[:50]}...'")

            try:
                # 1. Sentiment Analysis
                # The pipeline returns a list of dicts, e.g., [{'label': 'NEGATIVE', 'score': 0.99}]
                sentiment_result = self.sentiment_analyzer(response_text)[0]
                # We are interested in the score for the 'NEGATIVE' label
                if sentiment_result['label'] == 'NEGATIVE':
                    sentiment_scores.append(sentiment_result['score'])
                else:
                    # If sentiment is positive, the negative score is 1 - positive_score
                    sentiment_scores.append(1 - sentiment_result['score'])
                logger.debug(f"Sentiment result: {sentiment_result}")

                # 2. Emotion Detection (using zero-shot classification)
                # The pipeline returns a dict with 'sequence', 'labels', and 'scores'
                emotion_result = self.emotion_analyzer(response_text, self.emotion_labels)
                # Store scores for each emotion label
                for label, score in zip(emotion_result['labels'], emotion_result['scores']):
                     if label in emotion_scores: # Ensure label is one we track
                         emotion_scores[label].append(score)
                logger.debug(f"Emotion result (top 3): {list(zip(emotion_result['labels'], emotion_result['scores']))[:3]}")


                # 3. Semantic Similarity
                response_embedding = self._get_embeddings([response_text])
                # Calculate cosine similarity between the response embedding and reference embeddings
                similarities = cosine_similarity(response_embedding, self.depression_reference_embeddings)
                # Use the maximum similarity to any of the reference texts
                max_similarity = np.max(similarities)
                semantic_similarity_scores.append(max_similarity)
                logger.debug(f"Semantic similarity (max): {max_similarity:.4f}")

            except Exception as e:
                logger.error(f"Error analyzing response for {q_id}: {e}", exc_info=True)
                # Decide how to handle errors - skip this response or return an error score?
                # For now, we'll log and skip this specific response's scores

        # --- Aggregate Scores ---
        # Calculate average scores across all valid responses
        avg_negative_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        # Calculate average score for key depression-related emotions (e.g., sadness, hopelessness, apathy)
        key_emotion_avg_scores = []
        for label in ["sadness", "hopelessness", "apathy", "anxiety"]:
             if emotion_scores.get(label):
                 key_emotion_avg_scores.append(np.mean(emotion_scores[label]))
        avg_key_emotion_score = np.mean(key_emotion_avg_scores) if key_emotion_avg_scores else 0.0

        avg_semantic_similarity = np.mean(semantic_similarity_scores) if semantic_similarity_scores else 0.0

        logger.info(f"Aggregated Scores - Sentiment: {avg_negative_sentiment:.4f}, Key Emotion: {avg_key_emotion_score:.4f}, Semantic Similarity: {avg_semantic_similarity:.4f}")

        # --- Combine Aggregated Scores into a Single Text Score ---
        # This is a simple linear combination. You might train a small model here later.
        # Assign weights based on perceived importance (example weights)
        weight_sentiment = 0.3
        weight_emotion = 0.4
        weight_semantic = 0.3

        # Normalize scores if necessary (e.g., if they aren't naturally between 0 and 1)
        # Sentiment and cosine similarity are typically 0-1. Zero-shot scores are also probabilities (0-1).
        # So, direct combination might be okay, but consider scaling if ranges differ.

        text_depression_score = (
            weight_sentiment * avg_negative_sentiment +
            weight_emotion * avg_key_emotion_score +
            weight_semantic * avg_semantic_similarity
        )

        # Ensure the score is within [0, 1]
        text_depression_score = max(0.0, min(1.0, text_depression_score))

        logger.info(f"Final Text Depression Score: {text_depression_score:.4f}")

        return text_depression_score

# Example Usage (for testing)
if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()

    analyzer = TextAnalyzer()

    sample_responses_positive = {
        "q1": "I feel great today, very happy and energetic.",
        "q2": "I enjoyed spending time with my friends.",
        "q3": "I slept well and feel rested."
    }

    sample_responses_negative = {
        "q1": "I feel really down and tired.",
        "q2": "I don't have energy for anything, nothing feels fun.",
        "q3": "I've been having trouble sleeping and feel restless."
    }

    sample_responses_mixed = {
        "q1": "I had a tough morning but the afternoon was okay.",
        "q2": "Sometimes I feel sad, but other times I'm fine.",
        "q3": "" # Empty response
    }


    print("\nAnalyzing positive responses:")
    score_positive = analyzer.analyze(sample_responses_positive)
    print(f"Text Score (Positive): {score_positive:.4f}")

    print("\nAnalyzing negative responses:")
    score_negative = analyzer.analyze(sample_responses_negative)
    print(f"Text Score (Negative): {score_negative:.4f}")

    print("\nAnalyzing mixed responses:")
    score_mixed = analyzer.analyze(sample_responses_mixed)
    print(f"Text Score (Mixed): {score_mixed:.4f}")
