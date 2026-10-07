from flask import Flask, request, jsonify
from flask_cors import CORS
from newspaper import Article
import cloudscraper
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app)

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.8-flash')
else:
    model = None

scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if not model:
        return jsonify({'error': 'Server misconfiguration: Gemini API key missing.'}), 500

    try:
        data = request.get_json()
        req_type = data.get('type')
        content = data.get('content', '').strip()
        
        text_to_analyze = ""

        if not content:
            return jsonify({'error': 'Please provide either a valid URL or text.'}), 400
        
        # Extract text based on input type
        if req_type == 'url':
            try:
                response = scraper.get(content, timeout=15)
                if response.status_code != 200:
                    return jsonify({'error': f'Website blocked the scraper (Status {response.status_code}).'}), 400
                    
                article = Article(content)
                article.download(input_html=response.text)
                article.parse()
                text_to_analyze = article.text
                
                if not text_to_analyze.strip():
                    return jsonify({'error': 'No text found in the article.'}), 400
            except Exception as e:
                return jsonify({'error': f'Failed to read URL: {str(e)}'}), 400
                
        elif req_type == 'text':
            text_to_analyze = content
        else:
            return jsonify({'error': 'Invalid input type.'}), 400

        # Prompt Gemini directly without live web context
        prompt = f"""
        You are a strict fact-checking AI. Analyze the following news text for factual accuracy based on your extensive knowledge base. 
        Respond with EXACTLY ONE WORD: 'Real' if the core claims are factually true, or 'Fake' if the claims are false, heavily misleading, or satirical. Do not include any punctuation or explanations.

        Text to analyze:
        {text_to_analyze}
        """
        
        response = model.generate_content(prompt)
        verdict = response.text.strip().lower()

        # Map verdict to uppercase to match script.js
        prediction_val = "REAL" if "real" in verdict else "FAKE"

        return jsonify({
            'prediction': prediction_val,
            'result': "Real" if prediction_val == "REAL" else "Fake",
            'message': 'Fact-checked via Gemini analysis.'
        })

    except Exception as e:
        return jsonify({'error': f'Fact-checking failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)