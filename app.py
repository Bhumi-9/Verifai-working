"""
VerifAI — Flask Application
"""
import os, sqlite3, datetime
from flask import Flask, request, jsonify, render_template
from predict import predict

app = Flask(__name__)
DB  = os.path.join(os.path.dirname(__file__), 'instance', 'verifai.db')

def init_db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review TEXT, label TEXT, confidence REAL,
        fake_prob REAL, genuine_prob REAL, ts TEXT
    )''')
    conn.commit(); conn.close()

def log(review, label, confidence, fp, gp):
    try:
        conn = sqlite3.connect(DB)
        conn.execute('INSERT INTO predictions VALUES (NULL,?,?,?,?,?,?)',
            (review[:500], label, confidence, fp, gp, datetime.datetime.utcnow().isoformat()))
        conn.commit(); conn.close()
    except: pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({'error': 'Missing review field'}), 400
    text = data['review'].strip()
    if len(text) < 5:
        return jsonify({'error': 'Review too short (min 5 chars)'}), 422
    if len(text) > 5000:
        return jsonify({'error': 'Review too long (max 5000 chars)'}), 422
    result = predict(text)
    if 'error' not in result:
        log(text, result['label'], result['confidence'], result['fake_prob'], result['genuine_prob'])
    return jsonify(result)

@app.route('/api/stats')
def api_stats():
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM predictions'); total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE label='Fake'"); fake = c.fetchone()[0]
        c.execute('SELECT AVG(confidence) FROM predictions'); avg = c.fetchone()[0] or 0
        conn.close()
        return jsonify({'total': total, 'fake': fake, 'genuine': total-fake, 'avg_conf': round(avg,1)})
    except: return jsonify({'total':0,'fake':0,'genuine':0,'avg_conf':0})

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'version': '2.0'})

if __name__ == '__main__':
    init_db()
    print("✅  VerifAI running → http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
