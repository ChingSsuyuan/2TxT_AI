
"""
This is the main application file for the Image Caption Generator web app.
"""
from flask import Flask, request, render_template, jsonify
import os
import json
import subprocess
import uuid
import shutil

app = Flask(__name__)
 
# Configuration
UPLOAD_FOLDER = 'test_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CAPTION_OUTPUT = 'generated_captions.json'
MODEL_WEIGHTS = './checkpoints/clip_pro_prefix-001.pt'  
CLIP_MODEL = 'RN50x4'  
USE_CPU = True  
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and generate caption"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        for f in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        unique_filename = f"{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[1].lower()}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        try:
            import sys
            cmd = [
                sys.executable, '03_predict.py',
                '--img_dir', UPLOAD_FOLDER,
                '--weights', MODEL_WEIGHTS,
                '--clip_model', CLIP_MODEL,
                '--output_file', CAPTION_OUTPUT
            ]
            
            if USE_CPU:
                cmd.append('--use_cpu')
                
            subprocess.run(cmd, check=True)
            with open(CAPTION_OUTPUT, 'r', encoding='utf-8') as f:
                captions = json.load(f)
            if unique_filename in captions:
                caption = captions[unique_filename]
            else:
                caption = "Caption not generated for this image."
            
            return jsonify({
                'success': True,
                'filename': unique_filename,
                'caption': caption
            })
            
        except Exception as e:
            return jsonify({'error': f'Error generating caption: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the server is running"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)