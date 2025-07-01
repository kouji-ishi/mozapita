from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import os
import uuid
import shutil
import zipfile
import io
from collections import Counter
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

app = Flask(__name__)

def add_blocks_to_image(image_path, results, save_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # 黄色の塗りつぶし矩形（不透明な■）
        draw.rectangle([x1, y1, x2, y2], fill="yellow")

    pil_img.save(save_path)

# モデルロード（← ご自身のモデルパスに変更済み）
model = YOLO("C:/Users/石渡浩二/runs/detect/train5/weights/best.pt")

# フォルダ設定
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output/predict'

# フォルダ作成（最初だけ）
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # predict フォルダの中身を削除（ファイルだけ消す）
    if os.path.exists(OUTPUT_FOLDER):
        for filename in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'ファイル削除エラー: {e}')

    files = request.files.getlist('image')  # 複数ファイル対応
    result_image_paths = []
    filenames = []
    summary_texts = []

    for file in files:
        if file:
            filename = file.filename
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            # YOLO 推論実行
            results = model.predict(
            source=upload_path,
            conf=0.1,
            save=False  # YOLO側の保存はOFF
            )

            # ★マーク付き画像として保存
            result_image_name = os.path.basename(upload_path)
            result_image_path = os.path.join(OUTPUT_FOLDER, result_image_name)
            add_blocks_to_image(upload_path, results, result_image_path)
            
            # 表示用のパスはURLパスに変換
            result_image_web_path = '/static/output/predict/' + result_image_name

            # 検出ラベルをカウントして文字列化
            labels = model.names  # ← モデル定義のラベル辞書を直接参照
            cls_array = results[0].boxes.cls.cpu().numpy()
            detected_labels = [labels[int(cls)] for cls in cls_array]
            counts = Counter(detected_labels)
            label_order = ["nipple", "genitals", "penis"]
            summary = " / ".join([f"{k}:{counts[k]}" for k in label_order if k in counts])

            labels = results[0].names
            print("🔍 ラベル辞書:", labels)  # 追加デバッグ

            cls_array = results[0].boxes.cls.cpu().numpy()
            print("🔍 検出クラス:", cls_array)  # 追加デバッグ

            detected_labels = [labels[int(cls)] for cls in cls_array]
            print("🔍 検出ラベル:", detected_labels)  # 追加デバッグ

            # 各リストに追加
            result_image_paths.append(result_image_web_path)
            filenames.append(filename)
            summary_texts.append(summary)

    # 結果データをテンプレートに渡す
    image_filename_pairs = list(zip(result_image_paths, filenames, summary_texts))

    return render_template("result.html", image_filename_pairs=image_filename_pairs)

@app.route('/download_zip')
def download_zip():
    # ZIPをメモリ上に作成
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            if os.path.isfile(file_path):
                zipf.write(file_path, arcname=filename)

    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='mosaic_results.zip'
    )

@app.route('/download_selected_zip', methods=['POST'])
def download_selected_zip():
    selected_images = request.form.getlist('selected_images')

    if not selected_images:
        return 'No images selected.', 400

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for img_path in selected_images:
            filename = os.path.basename(img_path)
            file_full_path = os.path.join(OUTPUT_FOLDER, filename)
            if os.path.isfile(file_full_path):
                zf.write(file_full_path, arcname=filename)

    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='selected_images.zip'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)