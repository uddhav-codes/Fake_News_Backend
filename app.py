from flask import Flask, request, jsonify
from flask_cors import CORS
from newspaper import Article
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app)

# Configure Gemini using an environment variable (set securely in Render)
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    # Using flash for the fastest response times
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

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
        
        # 1. Extract text based on input type
        if req_type == 'url':
            try:
                article = Article(content)
                article.download()
                article.parse()
                text_to_analyze = article.text
                
                if not text_to_analyze.strip():
                    return jsonify({'error': 'Website blocked the scraper or no text found. Try pasting the text.'}), 400
            except Exception as e:
                return jsonify({'error': f'Failed to read URL: {str(e)}'}), 400
                
        elif req_type == 'text':
            text_to_analyze = content
        else:
            return jsonify({'error': 'Invalid input type.'}), 400

        # 2. Prompt Gemini to fact-check the text
        prompt = f"""
        You are a strict fact-checking AI. Analyze the following news text for factual accuracy. 
        Cross-check the claims against your knowledge base. 
        Respond with EXACTLY ONE WORD: 'Real' if the core claims are factually true, or 'Fake' if the claims are false, heavily misleading, or satirical. Do not include any punctuation or explanations.

        Text to analyze:
        {text_to_analyze}
        """
        
        response = model.generate_content(prompt)
        verdict = response.text.strip().lower()

        # 3. Format the result for the frontend
        final_verdict = "Real" if "real" in verdict else "Fake"

        return jsonify({'result': final_verdict})

    except Exception as e:
        return jsonify({'error': f'Fact-checking failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)