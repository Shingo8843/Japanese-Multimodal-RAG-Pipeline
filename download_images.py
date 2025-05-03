import os
import requests
from PIL import Image
from io import BytesIO

def download_images():
    """Download required images for testing"""
    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)
    
    # Image URLs and filenames (using Unsplash for reliable access)
    images = {
        "skytree.jpg": "https://images.unsplash.com/photo-1542051841857-5f90071e7989",
        "kinkakuji.jpg": "https://images.unsplash.com/photo-1624727828489-a1e03b79bba8",
        "ramen.jpg": "https://images.unsplash.com/photo-1569718212165-3a8278d5f624",
        "tempura.jpg": "https://images.unsplash.com/photo-1581781870027-04212e231e96",
        "sado.jpg": "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d",
        "sumo.jpg": "https://images.unsplash.com/photo-1605457212477-6cf0462f38fe",
        "shinkansen.jpg": "https://images.unsplash.com/photo-1524413840807-0c3cb6fa808d",
        "robot.jpg": "https://images.unsplash.com/photo-1485827404703-89b55fcc595e"
    }
    
    # Add headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Download each image
    for filename, url in images.items():
        try:
            print(f"Downloading {filename}...")
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                # Open and save the image
                img = Image.open(BytesIO(response.content))
                img.save(f"images/{filename}")
                print(f"✓ Successfully downloaded {filename}")
            else:
                print(f"✗ Failed to download {filename}: HTTP {response.status_code}")
        except Exception as e:
            print(f"✗ Error downloading {filename}: {str(e)}")
    
    print("\nDownload completed!")

if __name__ == "__main__":
    download_images() 