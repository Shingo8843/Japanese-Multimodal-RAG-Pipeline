# 日本語マルチモーダルRAGパイプライン

日本語のテキストと画像を処理し、正確で文脈を考慮した応答を提供する強力なRetrieval-Augmented Generationシステム。

## 📌 プロジェクト概要

### 目的
日本語のテキストと画像を処理し、正確で文脈を考慮した応答を提供するRetrieval-Augmented Generationシステムの開発。

### 機能
- テキストと画像によるクエリ処理
- 効率的な文書検索と再ランキング
- 日本語対応
- マルチモーダル応答生成
- 永続的なベクトルストレージ

## 🧩 アーキテクチャ概要

### コンポーネント
- **埋め込みモデル**: cl-nagoya/ruri-large（日本語テキスト埋め込み用）
- **再ランキング**: cl-nagoya/ruri-reranker-large（検索結果の精度向上用）
- **ベクトルストア**: ChromaDB（埋め込みの保存と検索用）
- **LLM**: Qwen-VL（検索された文脈に基づく応答生成用）

### ワークフロー
1. 入力: ユーザーが日本語のクエリ（テキストおよび/または画像）を送信
2. 埋め込み: Ruriを使用してクエリを埋め込み
3. 検索: ChromaDBから関連する文書/画像を取得
4. 再ランキング: 検索結果を関連性で再ランキング
5. 生成: 上位の結果をQwen-VLに渡して応答を生成

## 🛠️ セットアップとインストール

### 前提条件
- Python 3.8以上
- CUDA対応GPU（推奨）
- 仮想環境（推奨）

### 環境セットアップ
```bash
# 仮想環境の作成と有効化
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 依存関係のインストール
pip install -r requirements.txt

# CUDAサポート付きPyTorchのインストール
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 設定
プロジェクトは`config.py`ファイルで設定を管理します：
```python
# モデル設定
EMBEDDING_MODEL = "cl-nagoya/ruri-large"
RERANKER_MODEL = "cl-nagoya/ruri-reranker-large"
LLM_MODEL = "Qwen/Qwen-VL"

# ChromaDB設定
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "japanese_documents"

# RAG設定
TOP_K_RETRIEVAL = 5
TOP_N_RERANK = 3

# キャッシュ設定
CACHE_DIR = "cache"
```

## 📁 プロジェクト構造

```
.
├── app.py                 # メインアプリケーションファイル
├── config.py             # 設定ファイル
├── rag_pipeline.py       # コアRAGパイプライン実装
├── add_more_data.py      # データベースに文書を追加するスクリプト
├── clear_db.py          # データベースをクリア/リセットするユーティリティ
├── test_pipeline.py     # パイプラインのテストスクリプト
├── requirements.txt     # プロジェクトの依存関係
└── README.md           # プロジェクトドキュメント
```

## 💻 使用方法

### 文書の追加
```python
from rag_pipeline import JapaneseRAGPipeline

pipeline = JapaneseRAGPipeline()
pipeline.add_document(
    text="日本語のテキスト",
    image_data=base64_encoded_image,  # オプション
    metadata={"source": "wikipedia"}   # オプション
)
```

### クエリの処理
```python
response = pipeline.process_query(
    query="質問文",
    image=PIL_image  # オプション
)
print(response)
```

### データベース管理
```python
# データベースのクリアと再初期化
python clear_db.py

# サンプル文書の追加
python add_more_data.py

# テストの実行
python test_pipeline.py
```

## 🔍 実装の詳細

### 埋め込み処理
- テキスト埋め込みはRuriモデルを使用して生成
- 画像埋め込みはQwen-VLの隠れ状態から抽出
- マルチモーダルクエリのために埋め込みを正規化して結合

### 文書ストレージ
- 文書は埋め込みとともにChromaDBに保存
- テキストのみとマルチモーダルの両方の文書をサポート
- 文書管理のためのメタデータを含む

### クエリ処理
1. クエリ埋め込みの生成
2. ChromaDBでの類似検索
3. 検索文書の再ランキング
4. LLM用の文脈準備
5. Qwen-VLによる応答生成

## 📈 パフォーマンス

システムは以下のような様々なクエリでテストされ、良好なパフォーマンスを示しています：
- ランドマークの認識と説明
- 料理と食文化の情報
- 文化的トピック
- 技術関連のクエリ

## 🤝 貢献

問題の報告、リポジトリのフォーク、改善のためのプルリクエストの作成を歓迎します。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています - 詳細はLICENSEファイルを参照してください。

## 📄 参考文献

- [Ruri埋め込みモデル](https://huggingface.co/cl-nagoya/ruri-large)
- [Ruri再ランキング](https://huggingface.co/cl-nagoya/ruri-reranker-large)
- [Qwen-VLモデル](https://github.com/QwenLM/Qwen-VL)
- [ChromaDB](https://github.com/chroma-core/chroma) 