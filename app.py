# import os

from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
from waitress import serve

app = Flask(__name__)


@app.route('/encoding', methods=['POST'])
def encoding():
    data = request.json
    sentences = data.get("sentences", [])
    batch_size = int(data.get("batch_size", 8))

    # embeddings = encoder.encode(sentences, batch_size=batch_size)
    embeddings = encoder.encode(sentences, convert_to_tensor=True)
    embeddings = [x.tolist() for x in embeddings]

    return jsonify({"embeddings": embeddings})


if __name__ == '__main__':
    # model_name_or_path = os.environ.get('model_name_or_path', "bert-base-nli-stsb-mean-tokens")
    model_name_or_path = "stsb-xlm-r-multilingual"
    # encoder = SentenceTransformer(model_name_or_path=model_name_or_path)
    encoder = SentenceTransformer(model_name_or_path)

    serve(app, host="0.0.0.0", port=5000)
