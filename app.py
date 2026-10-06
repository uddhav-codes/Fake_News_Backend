from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from newspaper import Article
import os

app = Flask(__name__)
# Enable CORS so your GitHub Pages frontend can communicate with this Render backend
CORS(app)

# Load the WELFake-trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()
        text_to_analyze = ""
        
        # Handle URL scraping
        if 'url' in data and data['url'].strip() != "":
            url = data['url']
            try:
                article = Article(url)
                article.download()
                article.parse()
                text_to_analyze = article.text
                
                # Check if the website blocked the scraper
                if not text_to_analyze.strip():
                    return jsonify({'error': 'Website blocked the scraper or no text found. Try pasting the text manually.'}), 400
            except Exception as e:
                return jsonify({'error': f'Failed to read URL: {str(e)}'}), 400
                
        # Handle direct text input
        elif 'text' in data and data['text'].strip() != "":
            text_to_analyze = data['text']
            
        else:
            return jsonify({'error': 'Please provide either a valid URL or text.'}), 400

        # Vectorize the text
        vectorized_text = vectorizer.transform([text_to_analyze])
        
        # Make the prediction
        prediction = model.predict(vectorized_text)
        
        # Map WELFake labels (1 = Real, 0 = Fake)
        final_verdict = "Real" if prediction[0] == 1 else "Fake"

        return jsonify({'result': final_verdict})

    except Exception as e:
        # Catch internal server errors to prevent silent crashes
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Bind to the PORT environment variable required by Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)