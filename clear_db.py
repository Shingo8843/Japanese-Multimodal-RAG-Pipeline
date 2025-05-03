import os
import shutil
import config
from rag_pipeline import JapaneseRAGPipeline

def clear_database():
    """Clear the ChromaDB directory and reinitialize it"""
    try:
        # Remove the ChromaDB directory if it exists
        if os.path.exists(config.CHROMA_PERSIST_DIR):
            print(f"Removing ChromaDB directory: {config.CHROMA_PERSIST_DIR}")
            shutil.rmtree(config.CHROMA_PERSIST_DIR)
            print("ChromaDB directory removed successfully")
        
        # Initialize pipeline (this will create a new ChromaDB)
        print("\nInitializing pipeline with fresh database...")
        pipeline = JapaneseRAGPipeline()
        print("Pipeline initialized with demo data")
        
    except Exception as e:
        print(f"Error clearing database: {str(e)}")

if __name__ == "__main__":
    clear_database() 