from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from newspaper import Article
import os

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()
        
        # Extract the exact keys your frontend is sending
        req_type = data.get('type')
        content = data.get('content', '').strip()
        
        text_to_analyze = ""

        if not content:
            return jsonify({'error': 'Please provide either a valid URL or text.'}), 400
        
        # Handle URL scraping
        if req_type == 'url':
            try:
                article = Article(content)
                article.download()
                article.parse()
                text_to_analyze = article.text
                
                if not text_to_analyze.strip():
                    return jsonify({'error': 'Website blocked the scraper or no text found. Try pasting the text manually.'}), 400
            except Exception as e:
                return jsonify({'error': f'Failed to read URL: {str(e)}'}), 400
                
        # Handle direct text input
        elif req_type == 'text':
            text_to_analyze = content
            
        else:
            return jsonify({'error': 'Invalid input type.'}), 400

        # Vectorize and predict
        vectorized_text = vectorizer.transform([text_to_analyze])
        prediction = model.predict(vectorized_text)
        
        # Map WELFake labels (1 = Real, 0 = Fake)
        final_verdict = "Fake" if prediction[0] == 1 else "Real"

        return jsonify({'result': final_verdict})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)