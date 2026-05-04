from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "seed_training_data.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "model" / "detector.json"


@dataclass(frozen=True)
class NewsExample:
    label: str
    title: str
    text: str

    @property
    def combined_text(self) -> str:
        return " ".join(part for part in (self.title.strip(), self.text.strip()) if part).strip()


def load_examples(csv_path: Path) -> list[NewsExample]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        field_names = {name.strip().lower() for name in (reader.fieldnames or [])}
        if "label" not in field_names:
            raise ValueError("Training data must include a 'label' column.")
        if not ({"title", "text"} & field_names):
            raise ValueError("Training data must include at least a 'title' or 'text' column.")

        examples: list[NewsExample] = []
        for row_number, row in enumerate(reader, start=2):
            normalized = {key.strip().lower(): (value or "").strip() for key, value in row.items() if key}
            label = normalized.get("label", "").lower()
            if label not in {"fake", "real"}:
                raise ValueError(f"Row {row_number} has invalid label '{label}'. Use 'fake' or 'real'.")

            title = normalized.get("title", "")
            text = normalized.get("text", "") or normalized.get("body", "")
            combined = " ".join(part for part in (title, text) if part).strip()
            if not combined:
                raise ValueError(f"Row {row_number} is missing both title and text.")

            examples.append(NewsExample(label=label, title=title, text=text))

    if not examples:
        raise ValueError("Training data is empty.")

    return examples


def load_training_corpus(csv_path: Path) -> tuple[list[str], list[str]]:
    examples = load_examples(csv_path)
    documents = [example.combined_text for example in examples]
    labels = [example.label for example in examples]
    return documents, labels
