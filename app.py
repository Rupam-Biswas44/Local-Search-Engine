from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Load FAISS index
try:
    index = faiss.read_index('index/docs.index')
    print("FAISS index loaded successfully!")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    raise

# Load ID mapping
try:
    id_mapping = np.load('index/id_mapping.npy')
    print("ID mapping loaded successfully!")
except Exception as e:
    print(f"Error loading ID mapping: {e}")
    raise

# Load documents
try:
    with open('data/docs.json', 'r') as f:
        documents = json.load(f)
    print("Documents loaded successfully!")
except Exception as e:
    print(f"Error loading documents: {e}")
    raise

# Map id -> document
doc_map = {doc['id']: doc for doc in documents}


# Define search function
def search(query, top_k=3):
    try:
        # Encode query
        query_embedding = model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        print(f"Query embedding: {query_embedding}")

        # Search FAISS index
        distances, indices = index.search(query_embedding, top_k)
        print(f"Distances: {distances}, Indices: {indices}")

        # Collect results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            doc_id = id_mapping[idx]
            doc = doc_map[doc_id]
            results.append({
                'title': doc['title'],
                'content': doc['content'],
                'score': float(distance)
            })

        return results

    except Exception as e:
        print(f"Error during search: {e}")
        return []  # Return an empty list if an error occurs


# Define API route for search
@app.route('/search', methods=['POST'])
def search_api():
    try:
        # Get query from POST request
        data = request.get_json()
        query = data.get('query')
        top_k = data.get('top_k', 3)

        if not query:
            return jsonify({'error': 'Query field is required.'}), 400

        results = search(query, top_k)

        if not results:
            return jsonify({'error': 'No results found.'}), 404

        return jsonify(results)

    except Exception as e:
        print(f"Error in API request: {e}")
        return jsonify({'error': 'Internal server error'}), 500


# Home route (optional)
@app.route('/')
def home():
    return "Semantic Search API is running!"


# Run the Flask app
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, debug=True)
    except Exception as e:
        print(f"Error starting the app: {e}")
