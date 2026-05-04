from __future__ import annotations

import unittest
from pathlib import Path
import shutil

from fake_news_detector.data import DEFAULT_DATA_PATH, load_examples
from fake_news_detector.model import NaiveBayesFakeNewsDetector


class FakeNewsDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        training_docs = [
            "Miracle cure exposed by secret insiders and viral posts",
            "Government hides shocking truth about moon power",
            "Officials said the department released audited budget records",
            "Researchers said the paper describes the test method and results",
        ]
        labels = ["fake", "fake", "real", "real"]
        self.model = NaiveBayesFakeNewsDetector().fit(training_docs, labels)

    def test_fake_like_text_scores_as_fake(self) -> None:
        result = self.model.predict("Viral insiders reveal miracle device that changes the weather")
        self.assertEqual(result.label, "fake")
        self.assertGreater(result.probabilities["fake"], result.probabilities["real"])

    def test_reported_text_scores_as_real(self) -> None:
        result = self.model.predict(
            "Officials said the agency released inspection data and supporting records on Monday."
        )
        self.assertEqual(result.label, "real")
        self.assertGreater(result.probabilities["real"], result.probabilities["fake"])

    def test_model_round_trip_save_and_load(self) -> None:
        temp_dir = Path(__file__).resolve().parent / ".tmp_model"
        temp_dir.mkdir(exist_ok=True)
        model_path = temp_dir / "detector.json"
        try:
            self.model.save(model_path)
            loaded = NaiveBayesFakeNewsDetector.load(model_path)
            result = loaded.predict("Miracle rumor spreads online without named sources")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(result.label, "fake")

    def test_seed_dataset_is_available_and_balanced(self) -> None:
        examples = load_examples(DEFAULT_DATA_PATH)
        labels = [example.label for example in examples]
        self.assertIn("fake", labels)
        self.assertIn("real", labels)
        self.assertGreaterEqual(len(examples), 20)


if __name__ == "__main__":
    unittest.main()
