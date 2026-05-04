from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
STYLE_LABELS = {
    "__many_exclaims__": "many exclamation marks",
    "__upper_heavy__": "heavy uppercase styling",
    "__sensational__": "sensational wording",
    "__attributed__": "clear attribution language",
    "__contains_url__": "embedded URL",
    "__has_number__": "contains specific numbers",
    "__question_heavy__": "multiple question marks",
}


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    evidence: list[tuple[str, float]]


class NaiveBayesFakeNewsDetector:
    def __init__(self) -> None:
        self.labels = ("fake", "real")
        self.class_counts: dict[str, int] = {}
        self.token_totals: dict[str, int] = {}
        self.token_counts: dict[str, dict[str, int]] = {}
        self.class_log_priors: dict[str, float] = {}
        self.token_log_probs: dict[str, dict[str, float]] = {}
        self.vocabulary: set[str] = set()

    def fit(self, documents: Iterable[str], labels: Iterable[str]) -> "NaiveBayesFakeNewsDetector":
        docs = list(documents)
        y = [label.lower().strip() for label in labels]
        if len(docs) != len(y):
            raise ValueError("Document and label counts must match.")
        if not docs:
            raise ValueError("At least one training document is required.")
        if any(label not in self.labels for label in y):
            raise ValueError("Labels must be 'fake' or 'real'.")

        token_counts = {label: Counter() for label in self.labels}
        class_counts = Counter(y)

        for document, label in zip(docs, y):
            token_counts[label].update(self._extract_features(document))

        vocabulary = set()
        for counter in token_counts.values():
            vocabulary.update(counter.keys())

        if not vocabulary:
            raise ValueError("Training data did not produce any tokens.")

        total_docs = len(docs)
        token_totals = {label: sum(token_counts[label].values()) for label in self.labels}
        vocab_size = len(vocabulary)

        token_log_probs: dict[str, dict[str, float]] = {}
        for label in self.labels:
            denominator = token_totals[label] + vocab_size
            token_log_probs[label] = {
                token: math.log((token_counts[label][token] + 1) / denominator)
                for token in vocabulary
            }

        self.class_counts = dict(class_counts)
        self.token_totals = token_totals
        self.token_counts = {label: dict(counter) for label, counter in token_counts.items()}
        self.class_log_priors = {
            label: math.log(class_counts[label] / total_docs) for label in self.labels
        }
        self.token_log_probs = token_log_probs
        self.vocabulary = vocabulary
        return self

    def predict(self, text: str) -> PredictionResult:
        self._assert_is_trained()
        features = self._extract_features(text)
        feature_counts = Counter(feature for feature in features if feature in self.vocabulary)

        scores = {}
        for label in self.labels:
            score = self.class_log_priors[label]
            for token, count in feature_counts.items():
                score += count * self.token_log_probs[label][token]
            scores[label] = score

        probabilities = self._softmax(scores)
        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]
        evidence = self._top_evidence(feature_counts)

        return PredictionResult(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            evidence=evidence,
        )

    def save(self, path: Path) -> None:
        self._assert_is_trained()
        payload = {
            "labels": list(self.labels),
            "class_counts": self.class_counts,
            "token_totals": self.token_totals,
            "token_counts": self.token_counts,
            "class_log_priors": self.class_log_priors,
            "token_log_probs": self.token_log_probs,
            "vocabulary": sorted(self.vocabulary),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NaiveBayesFakeNewsDetector":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls()
        model.labels = tuple(payload["labels"])
        model.class_counts = {key: int(value) for key, value in payload["class_counts"].items()}
        model.token_totals = {key: int(value) for key, value in payload["token_totals"].items()}
        model.token_counts = {
            key: {token: int(count) for token, count in value.items()}
            for key, value in payload["token_counts"].items()
        }
        model.class_log_priors = {
            key: float(value) for key, value in payload["class_log_priors"].items()
        }
        model.token_log_probs = {
            key: {token: float(score) for token, score in value.items()}
            for key, value in payload["token_log_probs"].items()
        }
        model.vocabulary = set(payload["vocabulary"])
        return model

    def _top_evidence(self, feature_counts: Counter[str]) -> list[tuple[str, float]]:
        contributions = defaultdict(float)
        for token, count in feature_counts.items():
            fake_score = self.token_log_probs["fake"][token]
            real_score = self.token_log_probs["real"][token]
            contributions[token] += count * (fake_score - real_score)

        ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        formatted: list[tuple[str, float]] = []
        for token, score in ranked[:5]:
            label = STYLE_LABELS.get(token, token.replace("_", " "))
            formatted.append((label, score))
        return formatted

    def _softmax(self, scores: dict[str, float]) -> dict[str, float]:
        max_score = max(scores.values())
        exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
        total = sum(exp_scores.values())
        return {label: value / total for label, value in exp_scores.items()}

    def _assert_is_trained(self) -> None:
        if not self.vocabulary or not self.token_log_probs:
            raise RuntimeError("Model has not been trained yet.")

    def _extract_features(self, text: str) -> list[str]:
        lowered = text.lower()
        tokens = TOKEN_PATTERN.findall(lowered)
        bigrams = [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        features = list(tokens) + bigrams

        if text.count("!") >= 2:
            features.append("__many_exclaims__")
        if text.count("?") >= 2:
            features.append("__question_heavy__")
        if "http://" in lowered or "https://" in lowered or "www." in lowered:
            features.append("__contains_url__")
        if any(char.isdigit() for char in text):
            features.append("__has_number__")

        uppercase_chars = sum(1 for char in text if char.isupper())
        letter_chars = sum(1 for char in text if char.isalpha())
        if letter_chars and uppercase_chars / letter_chars > 0.3:
            features.append("__upper_heavy__")

        sensational_markers = (
            "miracle",
            "secret",
            "shocking",
            "exposed",
            "viral",
            "hidden truth",
            "they don't want you to know",
        )
        attribution_markers = (
            "according to",
            "officials said",
            "the department released",
            "documents show",
            "records submitted",
            "researchers said",
            "the report said",
            "the paper describes",
        )

        if any(marker in lowered for marker in sensational_markers):
            features.append("__sensational__")
        if any(marker in lowered for marker in attribution_markers):
            features.append("__attributed__")

        return features
