# util_search_nearest_sentence.py
# !pip install -Uqq sentence-transformers faiss-gpu

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def search_nearest_sentence(
    sentence_refs,
    sentence_queries,
    model_name="intfloat/multilingual-e5-base",
    path_out_faiss_full_index_path=None,
    nearest_k=1,
):
    # モデルの選択と読み込み
    model = SentenceTransformer(model_name)

    # 英語ならば他にもよいモデルがある
    # https://huggingface.co/spaces/mteb/leaderboard など参照
    # model_name = "thenlper/gte-base"
    # model = SentenceTransformer(model_name)

    # データのエンコード
    embeddings_refrence = model.encode(
        sentence_refs,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_tensor=True
    )

    embeddings_query = model.encode(
        sentence_queries,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_tensor=True
    )

    # Faiss インデックスの構築と保存
    # faiss.IndexFlatL2の初期設定として、次元数を設定
    print(f"Type of embeddings_reference is {type(embeddings_refrence)}")
    print(f"embeddings_refrence[0]:{embeddings_refrence[0]}")
    faiss_index = faiss.IndexFlatL2(len(embeddings_refrence[0]))
    faiss_index.add(embeddings_refrence.detach().cpu().numpy())
    print(f"Tyoe of faiss_indes is {type(faiss_index)}")

    if path_out_faiss_full_index_path is not None:
        faiss.write_index(faiss_index, path_out_faiss_full_index_path)

    # 類似度検索と結果の保存
    # 今回は最も類似しているものを返すため1
    search_score, idx_list = \
        faiss_index.search(
            embeddings_query.detach().cpu().numpy().astype(np.float32),
            nearest_k,
        )

    df_out = pd.DataFrame([search_score.flatten(), idx_list.flatten()]).T
    df_out.columns = ["Score", "ID"]
    df_out["ID"] = df_out["ID"].astype(int)
    df_out["Sentence"] = [sentence_refs[idx] for idx in df_out["ID"]]

    return df_out

# 実行例
# 1つ目を正例として想定して入れた
article_titles = [
    "Vector Searchによる検索",
    "Firebase Authのリダイレクトログインを使っている人は今年の6月までに対応しないと大変ですよという注意喚起",
    "データ分析基盤まとめ(随時更新)",
]

sentence_queries="似た文書をベクトル検索で探し出したい ~SentenceTransformersとFaissで効率的にベクトル検索~"
print(f"search : {sentence_queries}")
result = search_nearest_sentence(
    article_titles,
    sentence_queries=[sentence_queries],
    model_name="stsb-xlm-r-multilingual",
    nearest_k=2,
)
print(result)
# => 以下のpandas dataframeが返ってくる
# Score	ID	Sentence
# 0.278252	0	Vector Searchによる検索