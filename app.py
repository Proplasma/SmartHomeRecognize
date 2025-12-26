from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading 
import requests 
import numpy as np 
import datetime 
import csv      
import os
import json 
from ai_core import SmartHomeAI 

app = Flask(__name__)
ai_system = SmartHomeAI()
camera = cv2.VideoCapture(0)

# --- CẤU HÌNH IFTTT ---
IFTTT_KEY = "Dán_Mã_Key_Của_Em_Vào_Đây" 
IFTTT_URL = "https://maker.ifttt.com/trigger/{event}/with/key/" + IFTTT_KEY

# --- FILE DỮ LIỆU ---
DEVICE_FILE = "devices.json"
USER_PREF_FILE = "user_prefs.json" 
HISTORY_FILE = "history_log.csv" # Định nghĩa tên file log cho chuẩn
last_log = "" 

# --- QUẢN LÝ THIẾT BỊ (GLOBAL) ---
def load_devices():
    if not os.path.exists(DEVICE_FILE):
        default_data = [
            {"id": "light", "name": "Đèn Chính", "status": "OFF", "on_gesture": "OPEN_HAND", "off_gesture": "FIST", "icon": "fa-lightbulb"},
            {"id": "fan", "name": "Quạt Trần", "status": "OFF", "on_gesture": "POINTING", "off_gesture": "VICTORY", "icon": "fa-fan"}
        ]
        with open(DEVICE_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_data, f, ensure_ascii=False, indent=4)
        return default_data
    try:
        with open(DEVICE_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except: return []

def save_devices(data):
    with open(DEVICE_FILE, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=4)

# --- QUẢN LÝ SỞ THÍCH CÁ NHÂN (PERSONAL) ---
def load_user_prefs():
    if not os.path.exists(USER_PREF_FILE): return {}
    try:
        with open(USER_PREF_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except: return {}

def save_user_prefs(data):
    with open(USER_PREF_FILE, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=4)

# Load dữ liệu vào RAM
devices_list = load_devices()
user_prefs = load_user_prefs()

# --- LOG & IFTTT ---
def save_history(user, action, method="AI"):
    global last_log
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        file_exists = os.path.isfile(HISTORY_FILE)
        with open(HISTORY_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(["Thời Gian", "Người Dùng", "Hành Động", "Phương Thức"])
            writer.writerow([now, user, action, method])
            last_log = f"{action} bởi {user} ({method})"
            print(f">>> [LOG] {last_log}")
    except: pass

def send_ifttt_command(event_name):
    try: 
        full_url = IFTTT_URL.format(event=event_name)
        requests.post(full_url)
    except: pass

# --- HÀM ĐIỀU KHIỂN THIẾT BỊ ---
def control_device_by_id(dev_id, action, user, method="AI"):
    global devices_list
    for dev in devices_list:
        if dev["id"] == dev_id:
            if dev["status"] == action: return 
            
            dev["status"] = action
            save_devices(devices_list) 
            
            event_name = f"{dev_id}_{'on' if action=='ON' else 'off'}"
            threading.Thread(target=send_ifttt_command, args=(event_name,)).start()
            
            save_history(user, f"{'BẬT' if action=='ON' else 'TẮT'} {dev['name']}", method)
            return

# --- XỬ LÝ VIDEO ---
def generate_frames():
    global devices_list, user_prefs
    while True:
        success, frame = camera.read()
        if not success: break
        
        processed_frame, user, gesture = ai_system.process_frame(frame)
        
        if user != "Unknown" and gesture != "None":
            command_executed = False
            
            # 1. ƯU TIÊN: Kiểm tra sở thích cá nhân
            if user in user_prefs:
                user_rules = user_prefs[user]
                for dev in devices_list:
                    dev_id = dev["id"]
                    if dev_id in user_rules:
                        prefs = user_rules[dev_id]
                        if gesture == prefs.get("on"):
                            control_device_by_id(dev_id, "ON", user, "Personal_Gesture")
                            command_executed = True
                        elif gesture == prefs.get("off"):
                            control_device_by_id(dev_id, "OFF", user, "Personal_Gesture")
                            command_executed = True
            
            # 2. MẶC ĐỊNH: Dùng luật chung
            if not command_executed:
                for dev in devices_list:
                    if gesture == dev["on_gesture"]:
                        control_device_by_id(dev["id"], "ON", user, "Global_Gesture")
                    elif gesture == dev["off_gesture"]:
                        control_device_by_id(dev["id"], "OFF", user, "Global_Gesture")
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- API ENDPOINTS ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status(): return jsonify({"devices": devices_list, "last_log": last_log})

# --- API MỚI: PHÂN TÍCH DỮ LIỆU (ANALYTICS) ---
@app.route('/get_analytics')
def get_analytics():
    """Đọc file log, đếm dữ liệu và trả về JSON cho biểu đồ"""
    if not os.path.exists(HISTORY_FILE):
        return jsonify({"users": {"labels": [], "data": []}, "hours": {"labels": [], "data": []}})
    
    user_counts = {}
    hour_counts = {str(i): 0 for i in range(24)} # Tạo khung 0h -> 23h
    
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # Bỏ qua dòng tiêu đề
            
            for row in reader:
                # row = [Time, User, Action, Method]
                if len(row) < 3: continue
                
                # 1. Đếm theo User
                user = row[1]
                user_counts[user] = user_counts.get(user, 0) + 1
                
                # 2. Đếm theo Giờ
                try:
                    time_str = row[0] # VD: 2023-12-20 14:30:00
                    dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                    hour_counts[str(dt.hour)] += 1
                except: pass
                
    except Exception as e:
        print(f"Lỗi đọc thống kê: {e}")

    # Chuẩn bị dữ liệu trả về
    return jsonify({
        "users": {
            "labels": list(user_counts.keys()),
            "data": list(user_counts.values())
        },
        "hours": {
            "labels": [f"{i}h" for i in range(24)],
            "data": [hour_counts[str(i)] for i in range(24)]
        }
    })

# --- CÁC API KHÁC (GIỮ NGUYÊN) ---
@app.route('/set_user_pref', methods=['POST'])
def set_user_pref():
    global user_prefs
    user = request.form.get('user')
    dev_id = request.form.get('device_id')
    
    if user not in user_prefs: user_prefs[user] = {}
    user_prefs[user][dev_id] = { "on": request.form.get('on_gesture'), "off": request.form.get('off_gesture') }
    save_user_prefs(user_prefs)
    return jsonify({"status": "success", "message": f"Đã lưu cho {user}!"})

@app.route('/get_user_pref', methods=['POST'])
def get_user_pref():
    user = request.form.get('user')
    return jsonify(user_prefs.get(user, {}))

@app.route('/add_device', methods=['POST'])
def add_device():
    global devices_list
    dev_id = request.form.get('id')
    for dev in devices_list:
        if dev["id"] == dev_id: return jsonify({"status": "fail", "message": "ID tồn tại!"})
    
    new_dev = {"id": dev_id, "name": request.form.get('name'), "status": "OFF", 
               "on_gesture": request.form.get('on_gesture'), "off_gesture": request.form.get('off_gesture'), 
               "icon": request.form.get('icon')}
    devices_list.append(new_dev)
    save_devices(devices_list)
    return jsonify({"status": "success", "message": "Đã thêm!"})

@app.route('/delete_device', methods=['POST'])
def delete_device():
    global devices_list
    dev_id = request.form.get('id')
    devices_list = [d for d in devices_list if d["id"] != dev_id]
    save_devices(devices_list)
    return jsonify({"status": "success", "message": "Đã xóa!"})

@app.route('/toggle_device', methods=['POST'])
def toggle_device():
    control_device_by_id(request.form.get('device_id'), request.form.get('action'), "Web_Admin", "Manual")
    return jsonify({"status": "success"})

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    success, frame = camera.read()
    if success and ai_system.register_user(frame, name):
        return jsonify({"status": "success", "message": f"Đã đăng ký: {name}"})
    return jsonify({"status": "fail"})

@app.route('/register_upload', methods=['POST'])
def register_upload():
    if 'file' not in request.files: return jsonify({"status": "error"})
    try:
        file_bytes = np.frombuffer(request.files['file'].read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if ai_system.register_user(img, request.form.get('name')): 
            return jsonify({"status": "success", "message": "Upload OK"})
        return jsonify({"status": "fail"})
    except: return jsonify({"status": "error"})

@app.route('/get_users')
def get_users(): return jsonify({"users": list(ai_system.face_db.keys()), "count": len(ai_system.face_db)})

@app.route('/delete_user', methods=['POST'])
def delete_user():
    name = request.form.get('name')
    if name in ai_system.face_db:
        del ai_system.face_db[name]
        ai_system.save_database()
        global user_prefs
        if name in user_prefs:
            del user_prefs[name]
            save_user_prefs(user_prefs)
        return jsonify({"status": "success", "message": "Đã xóa user"})
    return jsonify({"status": "fail"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)