from rag_pipeline import JapaneseRAGPipeline
from PIL import Image
import os

def test_pipeline():
    """Test the pipeline with various types of queries"""
    try:
        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = JapaneseRAGPipeline()
        
        # Test cases
        test_cases = [
            # Landmark queries
            {
                "query": "東京タワーについて教えてください。",
                "image_path": None,
                "expected_topics": ["高さ", "完成年", "電波塔"]
            },
            {
                "query": "富士山の特徴は何ですか？",
                "image_path": None,
                "expected_topics": ["標高", "活火山", "世界遺産"]
            },
            {
                "query": "金閣寺の特徴は何ですか？",
                "image_path": None,
                "expected_topics": ["金箔", "世界遺産", "舎利殿"]
            },
            
            # Food queries
            {
                "query": "ラーメンの種類について教えてください。",
                "image_path": None,
                "expected_topics": ["醤油", "味噌", "塩", "豚骨"]
            },
            {
                "query": "寿司の種類について教えてください。",
                "image_path": None,
                "expected_topics": ["江戸前", "関西", "酢飯"]
            },
            
            # Culture queries
            {
                "query": "相撲の競技場は何と呼ばれますか？",
                "image_path": None,
                "expected_topics": ["土俵", "力士"]
            },
            
            # Technology queries
            {
                "query": "新幹線の最高速度はどれくらいですか？",
                "image_path": None,
                "expected_topics": ["320km/h", "1964年"]
            },
            {
                "query": "日本のロボット技術について教えてください。",
                "image_path": None,
                "expected_topics": ["産業用", "人型"]
            }
        ]
        
        # Run tests
        for i, test in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test['query']}")
            
            # Load image if specified
            image = None
            if test["image_path"] and os.path.exists(test["image_path"]):
                image = Image.open(test["image_path"])
            
            # Process query
            response = pipeline.process_query(test["query"], image)
            print(f"Response: {response}")
            
            # Check if expected topics are mentioned
            for topic in test["expected_topics"]:
                if topic in response:
                    print(f"✓ Found expected topic: {topic}")
                else:
                    print(f"✗ Missing expected topic: {topic}")
            
            print("-" * 80)
        
        print("\nTesting completed!")
        
    except Exception as e:
        print(f"Error in testing: {str(e)}")

if __name__ == "__main__":
    test_pipeline() 