import os
import json
from flask import Flask, jsonify, request
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer
from waitress import serve

app = Flask(__name__)
port_no = 5000

# モデルの初期化
model_name_or_path = "stsb-xlm-r-multilingual"
encoder = SentenceTransformer(model_name_or_path)


@app.route('/echo', methods=['GET'])
def echo():
    """Echo endpoint for testing."""
    return jsonify({"message": "Echo response"}), 200


@app.route('/encoding', methods=['POST'])
def encoding():
    """Encode sentences into embeddings."""
    data = request.json
    sentences = data.get("sentences", [])
    batch_size = int(data.get("batch_size", 8))
    embeddings = encoder.encode(sentences, convert_to_tensor=True)
    embeddings = [x.tolist() for x in embeddings]
    return jsonify({"embeddings": embeddings})


@app.route('/collections', methods=['GET'])
def get_collections():
    """Get list of collections."""
    storage_path = 'storage'
    if not os.path.exists(storage_path):
        return jsonify({"error": "Storage directory not found"}), 404
    collections = [d for d in os.listdir(storage_path) if os.path.isdir(os.path.join(storage_path, d))]
    return jsonify({"collections": collections}), 200


@app.route('/collections/<collection_name>', methods=['PUT'])
def create_collection(collection_name):
    """Create a new collection with given texts."""
    print(f"collections request at {collection_name}")
    try:
        data = request.json
        texts = data.get("texts", [])
        if not texts:
            return jsonify({"error": "Missing texts in request body"}), 400

        # textsをembeddingに変換
        text_embeddings = encoder.encode(texts, convert_to_tensor=True)
        text_embeddings_np = text_embeddings.cpu().numpy()

        # FAISSインデックスの構築と保存
        storage_path = os.path.join("storage", collection_name)
        os.makedirs(storage_path, exist_ok=True)
        index_path = os.path.join(storage_path, "index.faiss")

        faiss_index = faiss.IndexFlatL2(text_embeddings_np.shape[1])
        faiss_index.add(text_embeddings_np)
        faiss.write_index(faiss_index, index_path)

        # textを保存するためのファイルを作成
        text_file_path = os.path.join(storage_path, "texts.json")
        json_data = [{"id": i, "text": text} for i, text in enumerate(texts)]

        with open(text_file_path, "w", encoding="utf-8") as file:
            json.dump(json_data, file, indent=4)

        return jsonify({"message": "Collection created"}), 201
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/collections/<collection_name>/similarity', methods=['POST'])
def similarity_search(collection_name):
    """Search for similar texts in the collection."""
    print(f"similarity request at {collection_name}")
    storage_path = os.path.join("storage", collection_name)
    index_file_path = os.path.join(storage_path, "index.faiss")
    text_file_path = os.path.join(storage_path, "texts.json")

    if not os.path.exists(index_file_path):
        return jsonify({"error": "Index file not found"}), 404

    try:
        faiss_index = faiss.read_index(index_file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to load FAISS index: {str(e)}"}), 500

    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is missing in request body"}), 400

    query_embedding = encoder.encode([query], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().numpy()

    k = data.get("k", 1)
    distances, indices = faiss_index.search(query_embedding_np, k)
    # print(f"search {query} is distance {distances}")

    # 類似するインデックスが存在し、最初のインデックスが有効であるか確認
    if len(indices) > 0 and indices[0][0] != -1:
        # テキストファイルを読み込む
        with open(text_file_path, "r") as f:
            texts = json.load(f)

        # 最も類似するテキストのインデックスを取得
        index = int(indices[0][0])

        # 類似するテキストのリストを作成
        similar_texts = []
        for i in range(len(indices[0])):
            index = int(indices[0][i])
            distance = float(distances[0][i])
            text = None
            for item in texts:
                if item["id"] == index:
                    text = item["text"]
                    break
            if text:
                similar_texts.append({"text": text, "distance": distance})

        # 類似するテキストが見つかった場合
        if similar_texts:
            return jsonify({"similar_items": similar_texts}), 200
        else:
            return jsonify({"error": "No similar texts found"}), 404
    else:
        return jsonify({"error": "No similar points found"}), 404


if __name__ == '__main__':
    print(f"アプリを起動します。アクセスURL: http://localhost:{port_no}")
    serve(app, host="0.0.0.0", port=port_no)
