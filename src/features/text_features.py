# src/features/text_features.py: Extraction of deep learning and NLP text features for deal timelines.

import importlib.util
import json
import logging
import os
import pickle
import re
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.config import AppConfig
from src.ingestion.timeline_builder import DealTimelineModel

logger = logging.getLogger(__name__)

# --- Optional deep learning package imports ---
HAS_TORCH = importlib.util.find_spec("torch") is not None
if not HAS_TORCH:
    logger.warning("PyTorch is not installed. Disabling deep learning models.")

HAS_SBERT = False
if HAS_TORCH:
    try:
        from sentence_transformers import SentenceTransformer

        HAS_SBERT = True
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. Using HashingVectorizer fallback for SBERT."
        )

HAS_BERTOPIC = False
if HAS_TORCH:
    try:
        from bertopic import BERTopic

        HAS_BERTOPIC = True
    except ImportError:
        logger.warning(
            "bertopic not installed. Using KMeans clustering fallback for BERTopic."
        )

if not HAS_BERTOPIC:
    BERTopic = Any

HAS_TRANSFORMERS = False
if HAS_TORCH:
    try:
        from transformers import pipeline

        HAS_TRANSFORMERS = True
    except ImportError:
        logger.warning(
            "transformers not installed. Using lexicon rule-based fallback for RoBERTa sentiment."
        )


# --- Pydantic Validation Schema ---


class TextFeaturesRow(BaseModel):
    deal_id: int
    sbert_embedding: List[float] = Field(..., min_length=384, max_length=384)
    dominant_topic_id: int
    topic_drift_score: float = Field(..., ge=0.0, le=1.0)
    sentiment_mean: float = Field(..., ge=-1.0, le=1.0)
    sentiment_slope: float
    hedge_word_density: float = Field(..., ge=0.0, le=1.0)


def validate_text_features_df(df: pd.DataFrame) -> None:
    """Validates types and value ranges of the text features dataframe using Pydantic.

    Args:
        df: The text features dataframe.
    """
    records = df.reset_index().to_dict(orient="records")
    for r in records:
        cleaned = {}
        for k, v in r.items():
            if isinstance(v, (list, np.ndarray)):
                cleaned[k] = v.tolist() if isinstance(v, np.ndarray) else v
            else:
                cleaned[k] = None if pd.isna(v) else v
        try:
            TextFeaturesRow(**cleaned)
        except Exception as e:
            logger.error(
                f"Text feature validation failed for deal row: {cleaned}. Error: {e}"
            )
            raise ValueError(f"Text feature dataframe validation error: {e}")


# --- Fallback Implementations for Python 3.14 / Missing DL Libraries ---


class FallbackKMeansTopicModel:
    """A scikit-learn based KMeans topic modeling fallback."""

    def __init__(self, n_topics: int = 10):
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        self.kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init="auto")
        self.n_topics = n_topics

    def fit(self, documents: List[str]):
        """Fits TF-IDF and KMeans on a list of document strings."""
        if not documents:
            return self
        tfidf = self.vectorizer.fit_transform(documents)
        n_docs = len(documents)
        if n_docs < self.n_topics:
            from sklearn.cluster import KMeans

            self.kmeans = KMeans(
                n_clusters=max(1, n_docs), random_state=42, n_init="auto"
            )
        self.kmeans.fit(tfidf)
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Transforms document strings into cluster/topic assignments."""
        if not documents:
            return np.array([])
        tfidf = self.vectorizer.transform(documents)
        return self.kmeans.predict(tfidf)


def compute_fallback_sbert(text: str) -> np.ndarray:
    """Produces a 384-dimensional unit vector using deterministic feature hashing.

    Args:
        text: Input document string.

    Returns:
        A 384-dimensional L2-normalized numpy array.
    """
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(n_features=384, norm="l2", alternate_sign=True)
    dense_vec = vectorizer.transform([text]).toarray()[0]
    return dense_vec


# Pre-defined lexicon for sentiment fallback
POSITIVE_WORDS = {
    "great",
    "good",
    "excellent",
    "agree",
    "perfect",
    "success",
    "won",
    "happy",
    "pleased",
    "interested",
    "progress",
    "forward",
    "partnership",
    "thanks",
    "regards",
    "best",
    "deal",
}
NEGATIVE_WORDS = {
    "bad",
    "fail",
    "lost",
    "delay",
    "issue",
    "concern",
    "disagree",
    "reject",
    "sorry",
    "unfortunately",
    "problem",
    "difficult",
    "risk",
    "cancel",
    "tbd",
    "hedge",
    "not sure",
}


def compute_fallback_sentiment(text: str) -> float:
    """Computes lexicon-based sentiment score in [-1.0, 1.0].

    Args:
        text: Input document string.

    Returns:
        Sentiment score.
    """
    if not text:
        return 0.0
    words = re.findall(r"\b\w+\b", text.lower())
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / (total + 1.0)


# --- Text Feature Helper Functions ---


def load_hedge_words(path: str) -> List[str]:
    """Loads a list of hedge words/phrases from a flat text file.

    Args:
        path: Path to the hedge words text file.

    Returns:
        A list of cleaned hedge words.
    """
    if not os.path.exists(path):
        # Fallback if path is invalid
        logger.warning(f"Hedge words file not found at {path}. Using default list.")
        return ["might", "perhaps", "I think", "not sure", "TBD", "let me check"]

    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def compute_hedge_word_density(text: str, hedge_words: List[str]) -> float:
    """Computes hedge word density for a given text.

    Hedge word density is: (count of hedge word matches) / (total word count).

    Args:
        text: Input email body text.
        hedge_words: List of hedge words/phrases to match.

    Returns:
        A float density in range [0.0, 1.0].
    """
    if not text or not hedge_words:
        return 0.0

    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return 0.0

    # Compile pattern for all hedge phrases/words matching word boundaries case-insensitively
    escaped_terms = [re.escape(term) for term in hedge_words]
    pattern = re.compile(r"\b(" + "|".join(escaped_terms) + r")\b", re.IGNORECASE)

    matches = len(pattern.findall(text))
    return min(1.0, matches / total_words)


def compute_sentiment_slope(scores: List[float]) -> float:
    """Computes the linear regression slope of sentiment scores over message order.

    If the deal has fewer than 2 messages, the slope defaults to 0.0.

    Args:
        scores: Chronologically ordered list of sentiment scores for a deal's messages.

    Returns:
        The regression slope.
    """
    n = len(scores)
    if n < 2:
        return 0.0

    x = np.arange(n)
    y = np.array(scores)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Computes the cosine distance (1 - cosine_similarity) between two vectors.

    If either vector is zero, similarity is 1.0 (so distance is 0.0).

    Args:
        u: First vector.
        v: Second vector.

    Returns:
        Cosine distance in range [0.0, 1.0].
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0

    sim = np.dot(u, v) / (norm_u * norm_v)
    # Clip to avoid float inaccuracies out of [-1, 1] range
    sim = np.clip(sim, -1.0, 1.0)
    return float(1.0 - sim)


# --- Batched DL / Fallback Feature Extraction ---


def extract_sbert_embeddings_batched(
    texts: List[str], model_name: str, batch_size: int
) -> np.ndarray:
    """Extracts SBERT embeddings for a list of texts in batches.

    Args:
        texts: List of document strings.
        model_name: Name of the SentenceTransformer model.
        batch_size: Inference batch size.

    Returns:
        Numpy array of shape (len(texts), 384).
    """
    if not texts:
        return np.empty((0, 384))

    if HAS_SBERT:
        logger.info(
            f"Computing SBERT embeddings using '{model_name}' (batch_size={batch_size})..."
        )
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return np.array(embeddings)
    else:
        logger.info("Computing fallback SBERT embeddings using HashingVectorizer...")
        vectors = [compute_fallback_sbert(t) for t in texts]
        return np.vstack(vectors)


def extract_roberta_sentiment_batched(
    texts: List[str], model_name: str, batch_size: int
) -> List[float]:
    """Scores sentiment for a list of texts in batches using RoBERTa or fallback.

    Continuous sentiment score is: P(positive) - P(negative).

    Args:
        texts: List of document strings.
        model_name: Name of the transformers model.
        batch_size: Inference batch size.

    Returns:
        List of sentiment scores in range [-1.0, 1.0].
    """
    if not texts:
        return []

    if HAS_TRANSFORMERS:
        logger.info(
            f"Computing RoBERTa sentiment using '{model_name}' (batch_size={batch_size})..."
        )
        classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            top_k=None,
            device=-1,  # Force CPU for general environment compatibility
        )
        scores = []
        # Process batched inputs via pipeline generator/list
        results = classifier(texts, batch_size=batch_size)
        for res in results:
            # res is a list of score dicts: [{'label': 'negative', 'score': ...}, ...]
            probs = {d["label"].lower(): d["score"] for d in res}
            # Handle both name styles (positive/negative vs label_2/label_0)
            p_pos = probs.get("positive", probs.get("label_2", 0.0))
            p_neg = probs.get("negative", probs.get("label_0", 0.0))
            scores.append(float(p_pos - p_neg))
        return scores
    else:
        logger.info("Computing fallback sentiment using lexicon scorer...")
        return [compute_fallback_sentiment(t) for t in texts]


def train_or_load_topic_model(
    texts: List[str], embeddings: np.ndarray, model_dir: str, min_topic_size: int
) -> Tuple[Union[BERTopic, FallbackKMeansTopicModel], List[int]]:
    """Fits topic model (BERTopic or KMeans fallback) and saves it to disk if not cached.

    Args:
        texts: All corpus documents.
        embeddings: Pre-computed document SBERT embeddings.
        model_dir: Local path to save/load the topic model.
        min_topic_size: Minimum topic cluster size configuration.

    Returns:
        Tuple of (fitted model, document topic assignments).
    """
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "topic_model.pkl")

    if HAS_BERTOPIC:
        # Load from disk if it exists
        if os.path.exists(model_file) or os.path.exists(
            os.path.join(model_dir, "topics.json")
        ):
            logger.info(f"Loading cached BERTopic model from {model_dir}...")
            try:
                # Use BERTopic.load or pickle depending on how it was saved
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                topics = model.transform(texts, embeddings=embeddings)[0]
                return model, list(topics)
            except Exception as e:
                logger.warning(
                    f"Failed to load cached BERTopic model: {e}. Refitting..."
                )

        logger.info(f"Fitting BERTopic model with min_topic_size={min_topic_size}...")
        model = BERTopic(min_topic_size=min_topic_size, calculate_probabilities=False)
        topics, _ = model.fit_transform(texts, embeddings=embeddings)

        # Save model
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved BERTopic model to {model_file}")
        return model, list(topics)
    else:
        # Fallback custom KMeans clustering model
        if os.path.exists(model_file):
            logger.info(f"Loading cached FallbackKMeansTopicModel from {model_file}...")
            with open(model_file, "rb") as f:
                model = pickle.load(f)
            topics = model.transform(texts)
            return model, list(topics)

        logger.info(
            f"Fitting FallbackKMeansTopicModel with n_topics={min_topic_size}..."
        )
        model = FallbackKMeansTopicModel(n_topics=min_topic_size)
        model.fit(texts)
        topics = model.transform(texts)

        # Save model
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved FallbackKMeansTopicModel to {model_file}")
        return model, list(topics)


# --- Main Feature Extraction pipeline ---


def build_text_features(config: AppConfig) -> pd.DataFrame:
    """Loads deal timelines, extracts Stream A text features, and saves them to parquet.

    Args:
        config: Loaded AppConfig configurations containing path inputs/outputs.

    Returns:
        The validated text features DataFrame.
    """
    data_config = config.data
    features_config = config.features

    deals_dir = data_config.processed_deals_dir
    if not os.path.exists(deals_dir):
        logger.error(f"Processed deals directory does not exist: {deals_dir}")
        raise FileNotFoundError(f"Missing processed deals directory: {deals_dir}")

    timeline_files = [f for f in os.listdir(deals_dir) if f.endswith(".json")]
    if not timeline_files:
        logger.warning(f"No deal timeline files found in {deals_dir}")
        return pd.DataFrame()

    # Step 1: Parse all timeline documents and collect messages chronologically
    deal_timelines: List[DealTimelineModel] = []
    # Sort files to ensure deterministic ordering of deals
    for filename in sorted(timeline_files):
        filepath = os.path.join(deals_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        deal_timelines.append(DealTimelineModel(**data))

    # Flat list of messages mapped to deal identifiers
    all_message_texts: List[str] = []
    # Maps each message index back to deal_id & ordering
    message_metadata: List[Dict[str, Any]] = []

    for timeline in deal_timelines:
        emails = [e for e in timeline.events if e.type == "email"]
        for idx, email in enumerate(emails):
            all_message_texts.append(email.content)
            message_metadata.append(
                {
                    "deal_id": timeline.deal_id,
                    "msg_idx": idx,
                    "timestamp": email.timestamp,
                }
            )

    if not all_message_texts:
        logger.warning(
            "No email messages found in any deal timeline. Creating empty features DataFrame."
        )
        return pd.DataFrame()

    # Step 2: Compute Batched SBERT Embeddings (Stream A-1)
    embeddings = extract_sbert_embeddings_batched(
        all_message_texts, features_config.sbert_model_name, features_config.batch_size
    )

    # Step 3: Compute Batched Sentiment Scores (Stream A-3)
    sentiment_scores = extract_roberta_sentiment_batched(
        all_message_texts,
        features_config.roberta_sentiment_model_name,
        features_config.batch_size,
    )

    # Step 4: Fit/Transform Topic Modeling (Stream A-2)
    topic_model, topic_ids = train_or_load_topic_model(
        all_message_texts,
        embeddings,
        features_config.bertopic_model_dir,
        features_config.bertopic_min_topic_size,
    )

    # Step 5: Load hedge words
    hedge_words = load_hedge_words(features_config.hedge_words_resource_path)

    # Step 6: Map message-level features back to deals and aggregate
    # Add computed values to metadata list for easy grouping
    for i, meta in enumerate(message_metadata):
        meta["embedding"] = embeddings[i]
        meta["sentiment"] = sentiment_scores[i]
        meta["topic_id"] = topic_ids[i]
        meta["text"] = all_message_texts[i]

    # Convert to DataFrame of messages
    msg_df = pd.DataFrame(message_metadata)

    # Aggregate features per deal
    feature_rows = []
    # Identify unique topic space for probability vectors
    all_unique_topics = sorted(list(set(topic_ids)))
    topic_to_idx = {topic: idx for idx, topic in enumerate(all_unique_topics)}
    num_topics = len(all_unique_topics)

    for timeline in deal_timelines:
        deal_id = timeline.deal_id
        deal_msgs = msg_df[msg_df["deal_id"] == deal_id].sort_values("msg_idx")

        if deal_msgs.empty:
            # Handle deal with 0 email messages
            feature_rows.append(
                {
                    "deal_id": deal_id,
                    "sbert_embedding": [0.0] * 384,
                    "dominant_topic_id": -1,
                    "topic_drift_score": 0.0,
                    "sentiment_mean": 0.0,
                    "sentiment_slope": 0.0,
                    "hedge_word_density": 0.0,
                }
            )
            continue

        # A-1. SBERT embeddings: mean-pooling across all messages in timeline
        deal_embeddings = np.vstack(deal_msgs["embedding"].values)
        sbert_embedding = np.mean(deal_embeddings, axis=0)

        # A-2. Topic modeling (dominant topic and topic drift)
        deal_topics = deal_msgs["topic_id"].tolist()
        # Dominant topic for the deal (mode or most frequent topic)
        dominant_topic_id = int(max(set(deal_topics), key=deal_topics.count))

        # Compute Topic Drift Score: cosine distance between first half and second half topic vectors
        n_msgs = len(deal_topics)
        if n_msgs >= 2:
            mid = n_msgs // 2
            early_topics = deal_topics[:mid]
            late_topics = deal_topics[mid:]

            # Construct topic frequency histograms
            early_vec = np.zeros(num_topics)
            late_vec = np.zeros(num_topics)

            for t in early_topics:
                early_vec[topic_to_idx[t]] += 1
            for t in late_topics:
                late_vec[topic_to_idx[t]] += 1

            # Normalize to distributions
            early_vec = early_vec / len(early_topics)
            late_vec = late_vec / len(late_topics)

            topic_drift_score = compute_cosine_distance(early_vec, late_vec)
        else:
            topic_drift_score = 0.0

        # A-3. Sentiment Mean & Sentiment Slope
        deal_sentiments = deal_msgs["sentiment"].tolist()
        sentiment_mean = float(np.mean(deal_sentiments))
        sentiment_slope = compute_sentiment_slope(deal_sentiments)

        # A-4. Hedge-word density
        # Combine all texts in deal's messages and compute overall density
        combined_text = "\n".join(deal_msgs["text"].values)
        hedge_word_density = compute_hedge_word_density(combined_text, hedge_words)

        feature_rows.append(
            {
                "deal_id": deal_id,
                "sbert_embedding": sbert_embedding,
                "dominant_topic_id": dominant_topic_id,
                "topic_drift_score": topic_drift_score,
                "sentiment_mean": sentiment_mean,
                "sentiment_slope": sentiment_slope,
                "hedge_word_density": hedge_word_density,
            }
        )

    # Step 7: Build output DataFrame
    df = pd.DataFrame(feature_rows)
    df.set_index("deal_id", inplace=True)

    # Validate output schema
    validate_text_features_df(df)

    # Step 8: Save to Parquet format
    output_path = features_config.processed_text_features_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    logger.info(f"Successfully saved {len(df)} deal text feature rows to {output_path}")

    return df
