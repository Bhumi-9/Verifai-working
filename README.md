# VerifAI — Fake Review Detection System

A fully working ML-powered web app that detects fake product reviews in real-time.

## Quick Start (3 commands)

```bash
pip install flask scikit-learn numpy pandas

python3 train_model.py   # trains and saves model (run once)

python3 app.py           # starts server at http://localhost:5000
```

## How it works
1. Enter any product review in the demo box
2. Click "Analyze this review"
3. Get instant Fake/Genuine prediction with confidence % and explanation signals

## Project Structure
```
verifai-working/
├── app.py            ← Flask server (routes + API)
├── predict.py        ← ML inference module
├── train_model.py    ← Training script (run once)
├── model/            ← Saved model + vectorizer (auto-created)
│   ├── model.pkl
│   └── vectorizer.pkl
├── instance/         ← SQLite DB (auto-created)
│   └── verifai.db
└── templates/
    └── index.html    ← Full website (single file)
```

## API
`POST /api/predict` with `{"review": "your review text"}` returns:
```json
{
  "label": "Fake",
  "confidence": 85.7,
  "fake_prob": 85.7,
  "genuine_prob": 14.3,
  "signals": [
    {"text": "14 exclamation marks", "type": "bad"},
    {"text": "Promotional phrases detected", "type": "bad"}
  ]
}
```
