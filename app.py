import os
import io
import sys
import base64
import threading
import webbrowser
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

template_folder = os.path.join(base_path, 'templates')
app = Flask(__name__, template_folder=template_folder)
CORS(app)

model_path = os.path.join(base_path, 'runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)

class_map = {
    'Ripe': 'Cà chua chín',
    'near_ripe': 'Cà chua gần chín',
    'unripe': 'Cà chua xanh',
    'blossom_unfert': 'Hoa chưa thụ phấn',
    'blossom_fert': 'Hoa đã thụ phấn'
}

def img_to_base64(img_arr):
    # Encode image array to base64
    _, buffer = cv2.imencode('.jpg', img_arr)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/list_folder', methods=['POST'])
def list_folder():
    data = request.json
    folder_path = data.get('folder_path', '')
    p = Path(folder_path)
    if not p.exists() or not p.is_dir():
        return jsonify({"error": "Invalid folder path"}), 400
    
    images = [f.name for f in p.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    return jsonify({"images": sorted(images)})

@app.route('/api/count_folder', methods=['POST'])
def count_folder():
    data = request.json
    folder_path = data.get('folder_path', '')
    p = Path(folder_path)
    if not p.exists() or not p.is_dir():
        return jsonify({"error": "Invalid folder path"}), 400
    
    images = [f for f in p.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    
    total_counts = {v: 0 for v in class_map.values()}
    for img in images:
        results = model(img)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        for c in classes:
            orig_name = model.names[c]
            inter_name = class_map.get(orig_name, orig_name)
            total_counts[inter_name] += 1
            
    return jsonify({"counts": total_counts})

@app.route('/api/visualize', methods=['POST'])
def visualize():
    data = request.json
    folder_path = data.get('folder_path', '')
    image_name = data.get('image_name', '')
    
    img_path = Path(folder_path) / image_name
    if not img_path.exists():
        return jsonify({"error": "Image not found"}), 400
        
    results = model(img_path)
    counts = {v: 0 for v in class_map.values()}
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    for c in classes:
        orig_name = model.names[c]
        inter_name = class_map.get(orig_name, orig_name)
        counts[inter_name] += 1
        
    res_plotted = results[0].plot()
    
    # Resize for web visualization if too large
    h, w = res_plotted.shape[:2]
    max_h = 600
    if h > max_h:
        scale = max_h / float(h)
        res_plotted = cv2.resize(res_plotted, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
    b64_img = img_to_base64(res_plotted)
    
    return jsonify({
        "image": f"data:image/jpeg;base64,{b64_img}",
        "counts": counts
    })

@app.route('/api/visualize_upload', methods=['POST'])
def visualize_upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    # Read image directly from memory
    in_memory_file = file.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(img)
    counts = {v: 0 for v in class_map.values()}
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    for c in classes:
        orig_name = model.names[c]
        inter_name = class_map.get(orig_name, orig_name)
        counts[inter_name] += 1
        
    res_plotted = results[0].plot()
    
    h, w = res_plotted.shape[:2]
    max_h = 600
    if h > max_h:
        scale = max_h / float(h)
        res_plotted = cv2.resize(res_plotted, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
    b64_img = img_to_base64(res_plotted)
    
    return jsonify({
        "image": f"data:image/jpeg;base64,{b64_img}",
        "counts": counts
    })

@app.route('/api/count_single_upload', methods=['POST'])
def count_single_upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    in_memory_file = file.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(img)
    counts = {v: 0 for v in class_map.values()}
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    for c in classes:
        orig_name = model.names[c]
        inter_name = class_map.get(orig_name, orig_name)
        counts[inter_name] += 1
        
    return jsonify({
        "counts": counts,
        "filename": file.filename
    })

@app.route('/api/export_excel', methods=['POST'])
def export_excel():
    import pandas as pd
    data = request.json.get('data', [])
    headers = request.json.get('headers', [])
    
    df = pd.DataFrame(data, columns=headers)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Ket Qua')
    output.seek(0)
    
    from flask import send_file
    return send_file(
        output,
        download_name="Ket_Qua_Ca_Chua.xlsx",
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5123')

if __name__ == '__main__':
    threading.Timer(1.5, open_browser).start()
    app.run(port=5123, debug=False)
