# semantic-trans-search
- [sentence-transformers](https://www.sbert.net/index.html)を使ってベクトル値計算・ベクトル検索を提供するAPIサービスを構築する

## 利用モデル
- 試したモデルをまとめる

Model Name | 次元数 | 制約条件\n(Max Token Sizeなど) | ダウンロードサイズ(GB) | 備考
--|--|--|--|--
[stsb-xlm-r-multilingual](https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual) | 768次元 | -- | 1.1GB | コメントアウト
[Multilingual-E5](https://huggingface.co/intfloat/multilingual-e5-base) | 1,024次元 | 最大512トークン | 2.2GB | 利用

## Setup
```sh
# after git clone
cd semantic-trans-search
poetry install
```

## Usage
```sh
poetry shell
python3 ./app.py
# アプリを起動します。アクセスURL: http://localhost:5000
```

### Access Point
- ローカル起動の場合、アクセスポイントは
  - オリジン：http://localhost:5000
  - 提供パス：（チェックが実装済み）
    - [x] GET、`/echo`：応答のみ
    - [x] POST、`/embedding`：テキスト情報からベクトル値を取得
    - [x] GET、`/collections`：DB名（collection_name）のリストを取得
    - [ ] GET、`/collections/<collection_name>`：DB名の基本情報を取得
    - [x] PUT、`/collections/<collection_name>`：DB名がない場合、`storage/<collection_name>/index.faiss`ファイルを作成
    - [ ] GET、`/collections/<collection_name>/points`：DBに格納されるpointのID一覧を取得
    - [ ] GET、`/collections/<collection_name>/points/payload?id=<id>`：DBのIDに関する情報を取得
    - [x] PUT、`/collections/<collection_name>/point?id=<id>`：DB名に対して新たにIDでアイテムを作成・上書き
    - [x] POST、`/collections/<collection_name>/similarity`：DBに対して類似検索を実施（最も近いIDの情報を取得）

## References
- Zenn : 『[似た文書をベクトル検索で探し出したい](https://zenn.dev/nishimoto/articles/0c2ac8c061e597)』
- note : 『[Sentence Transformers の使い方](https://note.com/npaka/n/n82d058c68172)』
- 記事：『[nikkie-ftnextの日記 @ 2023-07-08](https://nikkie-ftnext.hatenablog.com/entry/sentence-transformers-embeddings-introduction-en-ja)』
  - 多言語を扱えるモデル [stsb-xlm-r-multilingual](https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual) を紹介してる
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- Zenn : 『[Embeddingsを使ってローカルでテキストをクラスタリングする](https://zenn.dev/libratech/articles/afe9c5b30668bb)』
  - 多言語・高精度モデル [Multilingual-E5](https://huggingface.co/intfloat/multilingual-e5-base) を紹介している
## LICENSE
- Apache-2.0 license
