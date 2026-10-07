from flask import Flask, request, jsonify
from flask_cors import CORS
from newspaper import Article
import cloudscraper
import google.generativeai as genai
import os
from duckduckgo_search import DDGS

app = Flask(__name__)
CORS(app)

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    # Revert to the standard free model without the premium search tool
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
        search_query = ""

        if not content:
            return jsonify({'error': 'Please provide either a valid URL or text.'}), 400
        
        # 1. Extract text and headline
        if req_type == 'url':
            try:
                response = scraper.get(content, timeout=15)
                if response.status_code != 200:
                    return jsonify({'error': f'Website blocked the scraper (Status {response.status_code}).'}), 400
                    
                article = Article(content)
                article.download(input_html=response.text)
                article.parse()
                text_to_analyze = article.text
                search_query = article.title # Grab the title to search the web
                
                if not text_to_analyze.strip():
                    return jsonify({'error': 'No text found in the article.'}), 400
            except Exception as e:
                return jsonify({'error': f'Failed to read URL: {str(e)}'}), 400
                
        elif req_type == 'text':
            text_to_analyze = content
            # Grab the first 60 characters of the pasted text to use as a search query
            search_query = content[:60]
        else:
            return jsonify({'error': 'Invalid input type.'}), 400

        # 2. Perform a free DuckDuckGo Search to fetch live context
        live_context = ""
        try:
            results = DDGS().text(search_query, max_results=3)
            for res in results:
                live_context += f"- {res['title']}: {res['body']}\n"
        except Exception:
            live_context = "No live web results available."

        # 3. Prompt Gemini with the live web evidence
        prompt = f"""
        You are a strict fact-checking AI. Analyze the following news text for factual accuracy. 
        Because you have a knowledge cutoff, I have provided recent search results as 'Live Context' to help you verify breaking events.
        
        Respond with EXACTLY ONE WORD: 'Real' if the core claims are factually true according to the Live Context, or 'Fake' if the claims are false, heavily misleading, or satirical. Do not include any punctuation or explanations.

        Live Context:
        {live_context}

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
            'message': 'Fact-checked via real-time web search and Gemini analysis.'
        })

    except Exception as e:
        return jsonify({'error': f'Fact-checking failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)