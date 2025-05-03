from rag_pipeline import JapaneseRAGPipeline
import torch
from tqdm import tqdm
import os
from huggingface_hub import snapshot_download

def download_models():
    """Download all required models"""
    models = {
        "cl-nagoya/ruri-large": "Ruri Embedding Model",
        "cl-nagoya/ruri-reranker-large": "Ruri Reranker",
        "Qwen/Qwen-VL": "Qwen-VL Model"
    }
    
    for model_id, model_name in models.items():
        print(f"\nDownloading {model_name} ({model_id})...")
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=f"models/{model_id.split('/')[-1]}",
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"✓ {model_name} downloaded successfully!")
        except Exception as e:
            print(f"✗ Error downloading {model_name}: {str(e)}")

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download models
    download_models()
    
    print("\nInitializing RAG pipeline...")
    pipeline = JapaneseRAGPipeline()
    print("Models loaded successfully!")

if __name__ == "__main__":
    main() 