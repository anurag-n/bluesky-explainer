"""
Semantic similarity scoring using sentence embeddings.

Uses the `sentence-transformers` library with the all-MiniLM-L6-v2 model —
a lightweight, fast model that produces 384-dimensional embeddings suitable
for comparing short text outputs like bullet point lists.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class SimilarityScorer:
    """
    Computes cosine similarity between two text strings using sentence embeddings.

    The model is loaded once at construction and reused across calls.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model = SentenceTransformer(model_name)

    def score(self, actual: str, expected: str) -> float:
        """
        Compute the cosine similarity between two texts.

        Both texts are encoded into embedding vectors, then the cosine
        similarity is computed as: dot(a, b) / (||a|| * ||b||).

        Args:
            actual: The agent-generated output.
            expected: The manually written ground truth.

        Returns:
            A float in [0.0, 1.0] where 1.0 is identical meaning.
        """
        embeddings = self._model.encode([actual, expected], normalize_embeddings=True)
        # With normalized embeddings, cosine similarity = dot product
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        # Clamp to [0, 1] to handle any floating point edge cases
        return max(0.0, min(1.0, similarity))
