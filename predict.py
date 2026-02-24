"""
VerifAI — Prediction Module (no NLTK)
"""
import pickle, re, os

# ── INLINE STOPWORDS ──────────────────────────────────────
STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'is','it','its','was','are','be','been','has','have','had','do','does',
    'did','will','would','could','should','may','might','shall','can','need',
    'i','me','my','we','our','you','your','he','his','she','her','they','them',
    'this','that','these','those','what','which','who','how','when','where',
    'not','no','nor','so','yet','both','either','neither','just','very','too',
    'also','more','most','any','all','some','such','even','than','then','there',
    'here','about','above','after','before','between','into','through','during',
    'if','as','up','out','off','over','under','again','further'
}

BASE = os.path.dirname(__file__)
_model = _vec = None

def _load():
    global _model, _vec
    mp = os.path.join(BASE, 'model', 'model.pkl')
    vp = os.path.join(BASE, 'model', 'vectorizer.pkl')
    if os.path.exists(mp) and os.path.exists(vp):
        with open(mp, 'rb') as f: _model = pickle.load(f)
        with open(vp, 'rb') as f: _vec   = pickle.load(f)
        print("✅  Model loaded.")
    else:
        print("⚠   model/model.pkl not found — run train_model.py first.")

_load()

def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

def _signals(text: str) -> list:
    signals = []
    exclaims = len(re.findall(r'!', text))
    upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    lower = text.lower()

    if exclaims >= 3:
        signals.append({'text': f'{exclaims} exclamation marks', 'type': 'bad'})
    if upper_ratio > 0.12:
        signals.append({'text': 'Unusual capitalization', 'type': 'bad'})
    if re.search(r'\b(best ever|love love|must buy|buy now|buy it now|changed my life)\b', lower):
        signals.append({'text': 'Promotional phrases detected', 'type': 'bad'})
    if re.search(r'\b(everyone|everybody|all my friends|tell everyone)\b', lower):
        signals.append({'text': 'Over-generalizing language', 'type': 'bad'})
    if len(text) < 80 and exclaims >= 2:
        signals.append({'text': 'Very short with hype', 'type': 'bad'})
    if re.search(r'(\w+)\s+\1\s+\1', lower):
        signals.append({'text': 'Repeated words (e.g. "love love love")', 'type': 'bad'})

    if re.search(r'\b(however|although|but|despite|though|while|whereas)\b', lower):
        signals.append({'text': 'Balanced, nuanced language', 'type': 'good'})
    if len(text) > 180:
        signals.append({'text': f'Detailed review ({len(text)} chars)', 'type': 'good'})
    if re.search(r'\b(after (using|buying|washing|testing|trying)|been using|weeks?|months?|days?)\b', lower):
        signals.append({'text': 'Time-referenced experience', 'type': 'good'})
    if re.search(r'\b(delivery|packaging|customer service|zipper|stitching|material|size|colour|color|weight|battery|screen|cable)\b', lower):
        signals.append({'text': 'Specific product details mentioned', 'type': 'good'})
    if re.search(r'\b(issue|problem|flaw|complaint|concern|disappointed|not perfect|could be better)\b', lower):
        signals.append({'text': 'Acknowledges flaws (honest)', 'type': 'good'})

    return signals[:6]

def predict(text: str) -> dict:
    if _model is None or _vec is None:
        return {'error': 'Model not loaded. Run train_model.py first.'}

    cleaned  = _clean(text)
    features = _vec.transform([cleaned])
    label_id = _model.predict(features)[0]
    proba    = _model.predict_proba(features)[0]

    label        = 'Fake' if label_id == 1 else 'Genuine'
    fake_prob    = round(float(proba[1]) * 100, 1)
    genuine_prob = round(float(proba[0]) * 100, 1)
    confidence   = round(float(max(proba)) * 100, 1)

    return {
        'label':        label,
        'confidence':   confidence,
        'fake_prob':    fake_prob,
        'genuine_prob': genuine_prob,
        'signals':      _signals(text),
        'word_count':   len(text.split()),
        'char_count':   len(text),
    }
