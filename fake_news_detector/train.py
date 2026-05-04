from __future__ import annotations

import argparse
from pathlib import Path

from .data import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, load_training_corpus
from .model import NaiveBayesFakeNewsDetector


def train_model(input_path: Path = DEFAULT_DATA_PATH, output_path: Path = DEFAULT_MODEL_PATH) -> NaiveBayesFakeNewsDetector:
    documents, labels = load_training_corpus(input_path)
    model = NaiveBayesFakeNewsDetector().fit(documents, labels)
    model.save(output_path)
    return model


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH, data_path: Path = DEFAULT_DATA_PATH) -> NaiveBayesFakeNewsDetector:
    if model_path.exists():
        return NaiveBayesFakeNewsDetector.load(model_path)
    return train_model(input_path=data_path, output_path=model_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the fake-news detector.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="CSV training data path")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="Where to save the trained model")
    parser.add_argument("--force", action="store_true", help="Retrain even if the output file already exists")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"Model already exists at {args.output}. Use --force to retrain.")
        return

    model = train_model(input_path=args.input, output_path=args.output)
    print(
        f"Trained detector on {sum(model.class_counts.values())} samples and saved it to {args.output}"
    )


if __name__ == "__main__":
    main()
