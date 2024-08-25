from flask import Flask, request, jsonify
from flask_cors import CORS  # Importar CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Ativar CORS no Flask

# Função para pré-processar os sonhos
def preprocess_text(dream):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(dream.lower())
    filtered_words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

@app.route('/analyze', methods=['POST'])
def analyze_dreams():
    dreams = request.json.get('dreams', [])
    
    dreams_processed = [preprocess_text(dream) for dream in dreams]
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(dreams_processed)
    
    similarities = cosine_similarity(X, X)

    result = []
    for i in range(len(similarities)):
        for j in range(i + 1, len(similarities)):
            result.append({
                'dream1': i + 1,
                'dream2': j + 1,
                'similarity': similarities[i][j]
            })
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
