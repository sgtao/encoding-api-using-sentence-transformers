import os
from flask import Flask, jsonify, request
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer
from waitress import serve
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS


app = Flask(__name__)
port_no = 5000

@app.route('/echo', methods=['GET'])
def echo():
    return jsonify({"message": "Echo response"}), 200


@app.route('/encoding', methods=['POST'])
def encoding():
    data = request.json
    sentences = data.get("sentences", [])
    batch_size = int(data.get("batch_size", 8))

    # embeddings = encoder.encode(sentences, batch_size=batch_size)
    embeddings = encoder.encode(sentences, convert_to_tensor=True)
    embeddings = [x.tolist() for x in embeddings]

    return jsonify({"embeddings": embeddings})

@app.route('/collections', methods=['GET'])
def get_collections():
    storage_path = 'storage'
    if not os.path.exists(storage_path):
        return jsonify({"error": "Storage directory not found"}), 404

    collections = [d for d in os.listdir(storage_path) if os.path.isdir(os.path.join(storage_path, d))]
    return jsonify({"collections": collections}), 200

@app.route('/collections/<collection_name>', methods=['PUT'])
def create_collection(collection_name):
    try:
        print(f"receive request as {collection_name}")
        data = request.json
        embedding_reference = data.get("embedding_reference")
        if embedding_reference is None:
            return jsonify({"error": "Missing embedding_reference in request body"}), 400

        # FAISSインデックスの構築と保存
        storage_path = os.path.join("storage", collection_name)
        os.makedirs(storage_path, exist_ok=True)
        index_path = os.path.join(storage_path, "index.faiss")

        # embedding_referenceをNumPy配列に変換
        embedding_reference_np = np.array(embedding_reference, dtype=np.float32)

        # FAISSインデックスの初期設定
        faiss_index = faiss.IndexFlatL2(embedding_reference_np.shape[1])
        faiss_index.add(embedding_reference_np)

        # インデックスをファイルに保存
        faiss.write_index(faiss_index, index_path)

        return jsonify({"message": "Collection created"}), 201
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/collections/<collection_name>/similarity', methods=['POST'])
def similarity_search(collection_name):
    print(f"similarity request at {collection_name}")
    storage_path = os.path.join("storage", collection_name)
    index_file_path = os.path.join(storage_path, "index.faiss")

    # `index_file_path`のファイルがなければ 404応答する
    if not os.path.exists(index_file_path):
        return jsonify({"error": "Index file not found"}), 404

    # FAISSファイルからコレクションを取得
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

    # 類似検索の実行
    k = data.get("k", 1)  # デフォルトで1件の類似結果を返す
    distances, indices = faiss_index.search(query_embedding_np, k)

    # print(jsonify({"closest_point_id": int(indices[0][0]), "distance": float(distances[0][0])}))

    if len(indices) > 0 and indices[0][0] != -1:
        return jsonify({"closest_point_id": int(indices[0][0]), "distance": float(distances[0][0])}), 200
    else:
        return jsonify({"error": "No similar points found"}), 404


if __name__ == '__main__':
    # model_name_or_path = os.environ.get('model_name_or_path', "bert-base-nli-stsb-mean-tokens")
    model_name_or_path = "stsb-xlm-r-multilingual"
    # encoder = SentenceTransformer(model_name_or_path=model_name_or_path)
    encoder = SentenceTransformer(model_name_or_path)

    print(f"アプリを起動します。アクセスURL: http://localhost:{port_no}")
    serve(app, host="0.0.0.0", port=port_no)
