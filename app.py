import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch

# הגדרת Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# הגדרת YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# יצירת תיקיית העלאות אם היא לא קיימת
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# מסלול לטעינת דף הנחיתה (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# מסלול להעלאת תמונה
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # הפעלת YOLO על התמונה שהועלתה
        results = model(filepath)
        output = results.pandas().xyxy[0].to_dict(orient="records")  # קבלת תוצאות בפורמט JSON

        return jsonify(output)

# הפעלת השרת
if __name__ == '__main__':
    app.run(debug=True)
