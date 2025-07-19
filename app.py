from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import sqlite3
import json

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from openpyxl import Workbook

app = Flask(__name__)

# Загружаем модель один раз при запуске сервера
model = YOLO('yolov8x.pt')

TARGET_CLASSES = ['hair drier', 'scissors']
names_dict = model.names

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            result TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'Нет файла'}), 400
    
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    results = model(img)
    
    # Отрисовка рамок для целевых классов - ножниц и фенов
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = names_dict[class_id]
            if class_name in TARGET_CLASSES:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                # Рисуем рамку и подпись
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(img, label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    # Сохранение результата
    output_path = os.path.join('static', 'result.jpg')
    cv2.imwrite(output_path, img)
    
    counts = {}
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = names_dict[class_id]
            if class_name in TARGET_CLASSES:
                counts[class_name] = counts.get(class_name, 0) + 1
    
    # Сохранение в базу данных
    try:
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            'INSERT INTO requests (timestamp, result) VALUES (?, ?)',
            (timestamp_str, json.dumps(counts))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Ошибка при сохранении в базу данных: {e}")
    
    return jsonify(counts)

# генераци отчета pdf
def generate_pdf_report():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, result FROM requests ORDER BY id DESC')
    records = cursor.fetchall()
    conn.close()

    filename = os.path.join('static', 'report.pdf')
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    c.setFont("Arial", 14)
    c.drawString(50, height - 50, "Control of equipment (scissors and hair driers")
    
    y_position = height - 80
    c.setFont("Helvetica", 10)

    for record in records:
        timestamp, result_json = record
        c.drawString(50, y_position, f"Time: {timestamp}")
        y_position -= 15

        counts = json.loads(result_json)
        for obj_type, count in counts.items():
            c.drawString(70, y_position, f"{obj_type}: {count}")
            y_position -= 12

        y_position -= 10
        if y_position < 50:
            c.showPage()
            y_position = height - 50

    c.save()
    return 'report.pdf'

# Функция для генерации Excel отчёта
def generate_excel_report():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, result FROM requests ORDER BY id DESC')
    records = cursor.fetchall()
    conn.close()

    wb = Workbook()
    ws = wb.active
    ws.title="Report"

    ws.append(["Time", "Equipment"])

    for record in records:
        timestamp, result_json=record
        counts=json.loads(result_json)
        counts_str=', '.join([f"{k}: {v}" for k,v in counts.items()])
        ws.append([timestamp, counts_str])

    filename=os.path.join('static','report.xlsx')
    wb.save(filename)
    return 'report.xlsx'

# Маршрут для скачивания PDF отчёта
@app.route('/download/pdf')
def download_pdf():
    filename=generate_pdf_report()
    return send_from_directory('static', filename , as_attachment=True)

# Маршрут для скачивания Excel отчёта
@app.route('/download/excel')
def download_excel():
    filename=generate_excel_report()
    return send_from_directory('static', filename , as_attachment=True)

if __name__=='__main__':
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
