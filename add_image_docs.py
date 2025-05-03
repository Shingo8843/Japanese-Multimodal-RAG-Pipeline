import os
from rag_pipeline import JapaneseRAGPipeline
from PIL import Image
import base64
from io import BytesIO

def add_image_documents():
    """Add image-based documents to the ChromaDB collection"""
    try:
        # Initialize the pipeline
        print("Initializing pipeline...")
        pipeline = JapaneseRAGPipeline()
        
        # Sample image documents with descriptions
        image_docs = [
            {
                "text": "東京タワーは、東京都港区芝公園にある電波塔です。高さは333メートルで、1958年に完成しました。",
                "image_path": "images/tokyo_tower.jpg",
                "metadata": {"source": "wikipedia", "type": "landmark"}
            },
            {
                "text": "富士山は日本で最も高い山で、高さは3,776メートルです。静岡県と山梨県にまたがっています。",
                "image_path": "images/mount_fuji.jpg",
                "metadata": {"source": "wikipedia", "type": "landmark"}
            },
            {
                "text": "新幹線は日本の高速鉄道システムで、最高速度は320km/hに達します。",
                "image_path": "images/shinkansen.jpg",
                "metadata": {"source": "wikipedia", "type": "transportation"}
            }
        ]
        
        # Add each document to the collection
        for doc in image_docs:
            if os.path.exists(doc["image_path"]):
                # Read and encode the image
                with open(doc["image_path"], "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Add to collection
                pipeline.add_document(
                    text=doc["text"],
                    image_data=img_data,
                    metadata=doc["metadata"]
                )
                print(f"Added document: {doc['text'][:30]}...")
            else:
                print(f"Warning: Image file not found: {doc['image_path']}")
        
        print("Image documents added successfully!")
        
    except Exception as e:
        print(f"Error adding image documents: {str(e)}")

if __name__ == "__main__":
    add_image_documents() 