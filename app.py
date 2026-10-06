from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from newspaper import Article
import os

app = Flask(__name__)
# Enable CORS so your frontend can communicate with this backend securely
CORS(app)

# Load the trained ML Model and TF-IDF Vectorizer
# (Instructions on how to generate these files are in the deployment steps below)
try:
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

def extract_text_from_url(url):
    """Scrapes the main article text from a provided URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title + " " + article.text
    except Exception as e:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model files not found on server."}), 500

    data = request.get_json()
    input_type = data.get('type')
    content = data.get('content')

    # Step 1: Process Input
    if input_type == 'url':
        text_to_analyze = extract_text_from_url(content)
        if not text_to_analyze:
            return jsonify({"error": "Failed to extract text from URL."}), 400
    else:
        text_to_analyze = content

    # Step 2: Vectorize and Predict (Maximized speed)
    try:
        vectorized_text = vectorizer.transform([text_to_analyze])
        prediction = model.predict(vectorized_text)[0]
        
        return jsonify({
            "prediction": prediction, # Expected to be "REAL" or "FAKE"
            "message": f"Successfully analyzed {len(text_to_analyze.split())} words."
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Binds to the port provided by the host environment (Required for free cloud hosting)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)