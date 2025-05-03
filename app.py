from flask import Flask, render_template, request, jsonify
from rag_pipeline import JapaneseRAGPipeline
import os
import tempfile
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
pipeline = JapaneseRAGPipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('query', '')
        image_data = data.get('image', None)
        
        # Process image if provided
        image = None
        if image_data:
            # Remove the data URL prefix
            image_data = image_data.split(',')[1]
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
        
        # Process query
        response = pipeline.process_query(query, image)
        
        return jsonify({
            'status': 'success',
            'response': response
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 