# Fake News Detector

A lightweight fake-news detector built in pure Python. It includes:

- a starter training dataset
- a reusable Naive Bayes text classifier
- a local web interface
- a small test suite

## What it does

This project classifies a news-like passage as `fake` or `real` using a simple bag-of-words and bigram model. It is designed as a practical starter project, not a production fact-checking system.

The detector works out of the box with the included seed dataset, and you can retrain it later on your own CSV data.

## Quick start

1. Train a model:

```powershell
python -m fake_news_detector.train --force
```

2. Start the local web app:

```powershell
python app.py
```

3. Open `http://127.0.0.1:8000`

## Train on your own data

Your CSV should contain:

- `label` with values `fake` or `real`
- `title` and/or `text`

Example header:

```csv
label,title,text
fake,Shocking claim spreads online,Post claims the moon is made of batteries.
real,City releases budget update,The finance department published the revised budget after a public meeting.
```

Then run:

```powershell
python -m fake_news_detector.train --input data\your_file.csv --output model\detector.json --force
```

## Run tests

```powershell
python -m unittest discover -s tests
```

## Project layout

- [app.py](\Projects\Fake-News-Detector\app.py)
- [fake_news_detector](\Projects\Fake-News-Detector\fake_news_detector)
- [data\seed_training_data.csv](\Projects\Fake-News-Detector\data\seed_training_data.csv)
- [tests\test_model.py](\Projects\Fake-News-Detector\tests\test_model.py)

## Notes

- The included dataset is intentionally small and synthetic so the app is easy to run locally.
- Confidence scores reflect how strongly the model leans on its training data. They are not proof that a claim is factually true or false.
