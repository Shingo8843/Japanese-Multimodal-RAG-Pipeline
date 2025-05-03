# Japanese Multimodal RAG Pipeline

A powerful Retrieval-Augmented Generation system for processing Japanese text and images, providing accurate and context-aware responses.

## 📌 Project Overview

### Objective
Develop a Retrieval-Augmented Generation system capable of processing Japanese text and images to provide accurate, context-aware responses.

### Features
- Text and image-based querying
- Efficient document retrieval and reranking
- Japanese language support
- Multimodal response generation
- Persistent vector storage

## 🧩 Architecture Overview

### Components
- **Embedding Model**: cl-nagoya/ruri-large for Japanese text embeddings
- **Reranker**: cl-nagoya/ruri-reranker-large to refine retrieved results
- **Vector Store**: ChromaDB for storing and querying embeddings
- **LLM**: Qwen-VL for generating responses based on retrieved context

### Workflow
1. Input: User submits a Japanese query (text and/or image)
2. Embedding: Query is embedded using Ruri
3. Retrieval: Relevant documents/images are retrieved from ChromaDB
4. Reranking: Retrieved results are reranked for relevance
5. Generation: Top results are passed to Qwen-VL to generate a response

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Virtual environment (recommended)

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Configuration
The project uses a `config.py` file to manage settings:
```python
# Model settings
EMBEDDING_MODEL = "cl-nagoya/ruri-large"
RERANKER_MODEL = "cl-nagoya/ruri-reranker-large"
LLM_MODEL = "Qwen/Qwen-VL"

# ChromaDB settings
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "japanese_documents"

# RAG settings
TOP_K_RETRIEVAL = 5
TOP_N_RERANK = 3

# Cache settings
CACHE_DIR = "cache"
```

## 📁 Project Structure

```
.
├── app.py                 # Main application file
├── config.py             # Configuration settings
├── rag_pipeline.py       # Core RAG pipeline implementation
├── add_more_data.py      # Script to add documents to the database
├── clear_db.py          # Utility to clear/reset the database
├── test_pipeline.py     # Test script for the pipeline
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## 💻 Usage

### Adding Documents
```python
from rag_pipeline import JapaneseRAGPipeline

pipeline = JapaneseRAGPipeline()
pipeline.add_document(
    text="日本語のテキスト",
    image_data=base64_encoded_image,  # Optional
    metadata={"source": "wikipedia"}   # Optional
)
```

### Processing Queries
```python
response = pipeline.process_query(
    query="質問文",
    image=PIL_image  # Optional
)
print(response)
```

### Database Management
```python
# Clear and reinitialize the database
python clear_db.py

# Add sample documents
python add_more_data.py

# Run tests
python test_pipeline.py
```

## 🔍 Implementation Details

### Embedding Handling
- Text embeddings are generated using the Ruri model
- Image embeddings are extracted from Qwen-VL's hidden states
- Embeddings are normalized and combined for multimodal queries

### Document Storage
- Documents are stored in ChromaDB with their embeddings
- Supports both text-only and multimodal documents
- Includes metadata for better document management

### Query Processing
1. Query embedding generation
2. Similarity search in ChromaDB
3. Reranking of retrieved documents
4. Context preparation for LLM
5. Response generation with Qwen-VL

## 📈 Performance

The system has been tested with various queries and shows good performance in:
- Landmark recognition and description
- Food and cuisine information
- Cultural topics
- Technology-related queries

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📄 References

- [Ruri Embedding Model](https://huggingface.co/cl-nagoya/ruri-large)
- [Ruri Reranker](https://huggingface.co/cl-nagoya/ruri-reranker-large)
- [Qwen-VL Model](https://github.com/QwenLM/Qwen-VL)
- [ChromaDB](https://github.com/chroma-core/chroma)

