import os
from rag_pipeline import JapaneseRAGPipeline
from PIL import Image
import base64
from io import BytesIO

def add_more_data():
    """Add more diverse Japanese content to the database"""
    try:
        # Initialize the pipeline
        print("Initializing pipeline...")
        pipeline = JapaneseRAGPipeline()
        
        # Sample documents with various topics
        documents = [
            # Landmarks and Places
            {
                "text": "東京タワーは、東京都港区芝公園にある電波塔です。高さは333メートルで、1958年に完成しました。赤と白の塗装が特徴的な観光名所です。",
                "image_path": "images/tokyo_tower.jpg",
                "metadata": {"source": "wikipedia", "type": "landmark", "location": "tokyo"}
            },
            {
                "text": "富士山は、静岡県と山梨県に跨る活火山です。標高3,776メートルで、日本で最も高い山です。2013年に世界文化遺産に登録されました。",
                "image_path": "images/mount_fuji.jpg",
                "metadata": {"source": "wikipedia", "type": "landmark", "location": "shizuoka"}
            },
            {
                "text": "金閣寺は、京都市北区にある臨済宗相国寺派の寺院です。正式名称は鹿苑寺で、1994年に世界遺産に登録されました。金箔で覆われた舎利殿が有名です。",
                "image_path": "images/kinkakuji.jpeg",
                "metadata": {"source": "wikipedia", "type": "landmark", "location": "kyoto"}
            },
            
            # Food and Cuisine
            {
                "text": "ラーメンは、中国から伝わった麺料理で、日本で独自の発展を遂げました。醤油、味噌、塩、豚骨など様々なスープがあり、地域によって特徴が異なります。",
                "image_path": "images/ramen.jpg",
                "metadata": {"source": "wikipedia", "type": "food", "category": "noodles"}
            },
            {
                "text": "寿司は、酢飯と生魚を組み合わせた日本の伝統料理です。江戸前寿司や関西寿司など、地域によって様々なスタイルがあります。",
                "image_path": "images/sushi.jpg",
                "metadata": {"source": "wikipedia", "type": "food", "category": "seafood"}
            },
            
            # Culture and Traditions
            {
                "text": "相撲は、日本の国技として知られる格闘技です。土俵と呼ばれる円形の競技場で、力士が勝負を競います。",
                "image_path": "images/sumo.jpeg",
                "metadata": {"source": "wikipedia", "type": "sport", "category": "traditional"}
            },
            
            # Technology and Innovation
            {
                "text": "新幹線は、日本の高速鉄道システムで、1964年に世界で初めて開業しました。最高速度は320km/hに達し、安全性と正確性が特徴です。",
                "image_path": "images/shinkansen.jpg",
                "metadata": {"source": "wikipedia", "type": "transportation", "category": "railway"}
            },
            {
                "text": "ロボット技術は、日本の得意分野の一つです。産業用ロボットから人型ロボットまで、様々な分野で活躍しています。",
                "image_path": "images/robot.jpg",
                "metadata": {"source": "wikipedia", "type": "technology", "category": "robotics"}
            }
        ]
        
        # Add each document to the collection
        for doc in documents:
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
        
        print("Documents added successfully!")
        
    except Exception as e:
        print(f"Error adding documents: {str(e)}")

if __name__ == "__main__":
    add_more_data() 