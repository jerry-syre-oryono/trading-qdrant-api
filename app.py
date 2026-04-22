import os
import numpy as np
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient

app = Flask(__name__)

# Hardcoded for deployment context as provided; normally use environment variables
QDRANT_URL = "https://d0754f8d-a323-402c-8dba-4cc8079c15e5.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MThkZTVlZGYtYzA0MS00NDg5LTkxYmMtM2UwYTI0YjQ0YWMwIn0.9-lx_H6LELMk2NOLJzdXPWIBWPFZgRi3gqk8FwE0u8U"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def encode_window(window_list):
    arr = np.array(window_list).flatten()
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else arr.tolist()

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Qdrant Similarity Search API is running",
        "endpoints": {
            "health": "/health",
            "similar": "/similar (POST)"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/similar', methods=['POST'])
def similar():
    data = request.get_json()
    window = data.get('window')
    if not window:
        return jsonify({"error": "no window"}), 400
    query_vec = encode_window(window)
    try:
        results = client.search(
            collection_name="winning_patterns",
            query_vector=query_vec,
            limit=5,
            score_threshold=0.7
        )
        if not results:
            return jsonify({"similarity_score": 0.5, "matches": 0})
        avg_score = np.mean([r.score for r in results])
        avg_return = np.mean([r.payload.get('return_pct', 0) for r in results])
        return jsonify({
            "similarity_score": float(avg_score),
            "avg_return": float(avg_return),
            "matches": len(results)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
