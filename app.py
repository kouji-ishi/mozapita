# app.py
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import os
import uuid
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output/predict'
MARK_FOLDER = 'static/marks'
TMP_FOLDER = 'static/tmp_dl'
DOWNLOAD_FOLDER = str(Path.home() / "Downloads")

model = YOLO(os.path.join("weights", "best.pt"))  # 必要に応じてモデルパスを変更

# フォルダ準備
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, MARK_FOLDER, TMP_FOLDER, DOWNLOAD_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def add_mark_image(image_path, results, save_path, mark_file):
    image = Image.open(image_path).convert("RGBA")
    mark = Image.open(os.path.join(MARK_FOLDER, mark_file)).convert("RGBA")

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        w, h = x2 - x1, y2 - y1

        # 🔽 縮小率（例: 0.6 = 60%サイズ）
        scale = 1.5
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 🔽 中央配置のためのオフセットを計算
        offset_x = x1 + (w - new_w) // 2
        offset_y = y1 + (h - new_h) // 2

        resized_mark = mark.resize((new_w, new_h), Image.LANCZOS)
        image.paste(resized_mark, (offset_x, offset_y), resized_mark)

    # JPEG保存対応（RGBA → RGB）
    if save_path.lower().endswith('.jpg') or save_path.lower().endswith('.jpeg'):
        image = image.convert("RGB")

    image.save(save_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist("images")
    image_filename_pairs = []

    for f in uploaded_files:
        uid = str(uuid.uuid4())[:8]
        filename = uid + '_' + f.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(save_path)

        results = model.predict(source=save_path, conf=0.1, save=False)
        
        summary = f"nipple:{sum(1 for b in results[0].boxes if b.cls[0] == 0)} / genitals:{sum(1 for b in results[0].boxes if b.cls[0] == 1)}"
        result_path = os.path.join(OUTPUT_FOLDER, filename)
        add_mark_image(save_path, results, result_path, 'mark01.png')

        image_filename_pairs.append((f"/static/output/predict/{filename}", filename, summary))

    return render_template('result.html', image_filename_pairs=image_filename_pairs)

@app.route('/result')
def result():
    image_filename_pairs = []
    for file in os.listdir(OUTPUT_FOLDER):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_filename_pairs.append((f"/static/output/predict/{file}", file, "nipple:? / genitals: ?"))
    return render_template('result.html', image_filename_pairs=image_filename_pairs)

@app.route('/apply_mark', methods=['POST'])
def apply_mark():
    data = request.get_json()
    filename = data.get('filename')
    mark_file = data.get('mark', 'mark01.png')

    original_path = os.path.join(UPLOAD_FOLDER, filename)
    result_path = os.path.join(OUTPUT_FOLDER, filename)
    results = model.predict(source=original_path, conf=0.1, save=False)
    add_mark_image(original_path, results, result_path, mark_file)

    return jsonify({'new_path': f'/static/output/predict/{filename}'})

@app.route('/download_selected_zip', methods=['POST'])
def download_selected():
    selected = request.form.get("selected_images", "").split(',')

    # TMP_FOLDER内のファイルだけ削除（フォルダ自体は残す）
    if os.path.exists(TMP_FOLDER):
        for f in os.listdir(TMP_FOLDER):
            try:
                os.remove(os.path.join(TMP_FOLDER, f))
            except Exception as e:
                print(f"ファイル削除失敗: {f} - {e}")
    else:
        os.makedirs(TMP_FOLDER, exist_ok=True)

    for path in selected:
        filename = os.path.basename(path)
        shutil.copy2(os.path.join(OUTPUT_FOLDER, filename), os.path.join(TMP_FOLDER, filename))

    zip_base_name = os.path.join(DOWNLOAD_FOLDER, 'selected')
    shutil.make_archive(zip_base_name, 'zip', TMP_FOLDER)
    return send_file(zip_base_name + '.zip', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
