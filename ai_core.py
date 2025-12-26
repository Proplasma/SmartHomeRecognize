"""
Module xử lý AI cho Smart Home
Nâng cấp: Thêm LOVE, ROCK, THREE vào bộ nhận diện
"""
import cv2
import numpy as np
import mediapipe as mp
import os
import json

class SmartHomeAI:
    def __init__(self):
        # ============= KHỞI TẠO MEDIAPIPE =============
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ============= KHỞI TẠO YUNET & SFACE =============
        path_detect = "models/face_detection_yunet_2023mar.onnx"
        path_recog = "models/face_recognition_sface_2021dec.onnx"
        
        if not os.path.exists(path_detect) or not os.path.exists(path_recog):
            print("LỖI: Thiếu file model trong thư mục models/!")
            
        self.detector = cv2.FaceDetectorYN.create(
            model=path_detect, config="", input_size=(320, 320), 
            score_threshold=0.8, nms_threshold=0.3, top_k=5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(model=path_recog, config="")

        # ============= DATABASE =============
        self.db_file = "face_db.json"
        self.face_db = self.load_database()
        self.threshold_cosine = 0.30 

    def load_database(self):
        if not os.path.exists(self.db_file): return {}
        try:
            with open(self.db_file, 'r') as f:
                data = json.load(f)
            db_converted = {}
            for name, feature_list in data.items():
                db_converted[name] = np.array(feature_list, dtype=np.float32)
            return db_converted
        except: return {}

    def save_database(self):
        data_to_save = {}
        for name, feature_array in self.face_db.items():
            data_to_save[name] = feature_array.tolist()
        with open(self.db_file, 'w') as f:
            json.dump(data_to_save, f)

    def check_face_quality(self, frame, face_box, landmarks):
        x, y, w, h = list(map(int, face_box[:4]))
        if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]: return False, "Sat le"
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0: return False, "Loi cat"
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 20: return False, f"Mo ({int(blur_score)})"

        nose_x = landmarks[2][0]
        right_eye_x = landmarks[0][0]
        left_eye_x = landmarks[1][0]
        dist_left = nose_x - right_eye_x
        dist_right = left_eye_x - nose_x
        if dist_right == 0: ratio = 0
        else: ratio = dist_left / dist_right

        if ratio < 0.3 or ratio > 3.0: return False, f"Nghieng ({ratio:.2f})"
        return True, f"OK ({int(blur_score)})"

    # --- HÀM NHẬN DIỆN CỬ CHỈ (UPDATE 3 CỬ CHỈ MỚI) ---
    def detect_gesture(self, frame_rgb):
        results = self.hands.process(frame_rgb)
        gesture = "None"
        
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                fingers = [] 
                # Tips ID: Index(8), Middle(12), Ring(16), Pinky(20)
                tips = [8, 12, 16, 20]
                
                # 1. Check ngón cái (ngang)
                if hand_lm.landmark[4].x < hand_lm.landmark[3].x: fingers.append(1)
                else: fingers.append(0)
                
                # 2. Check 4 ngón còn lại (dọc)
                for tip in tips:
                    if hand_lm.landmark[tip].y < hand_lm.landmark[tip-2].y: fingers.append(1)
                    else: fingers.append(0)
                
                # fingers = [Cái, Trỏ, Giữa, Áp Út, Út]
                total = fingers.count(1)
                
                # Lấy tọa độ Y để check Like/Dislike
                thumb_tip_y = hand_lm.landmark[4].y
                thumb_ip_y = hand_lm.landmark[3].y
                
                # --- PHÂN LOẠI CỬ CHỈ ---
                
                # 1. Bàn tay mở (5 ngón)
                if total == 5: gesture = "OPEN_HAND"
                
                # 2. Nắm đấm (0 ngón) hoặc Dislike
                elif total == 0: 
                    if thumb_tip_y > thumb_ip_y + 0.05: gesture = "THUMB_DOWN"
                    else: gesture = "FIST"
                
                # 3. Các cử chỉ 1 ngón
                elif total == 1:
                    if fingers[1] == 1: gesture = "POINTING" # Chỉ tay
                    elif fingers[0] == 1: # Ngón cái
                        if thumb_tip_y < thumb_ip_y: gesture = "THUMB_UP"

                # 4. Các cử chỉ 2 ngón
                elif total == 2:
                    if fingers[1] == 1 and fingers[2] == 1: gesture = "VICTORY" # Chữ V
                    elif fingers[1] == 1 and fingers[4] == 1: gesture = "ROCK"  # Rock (MỚI)

                # 5. Các cử chỉ 3 ngón
                elif total == 3:
                    if fingers[1]==1 and fingers[2]==1 and fingers[3]==1: gesture = "THREE" # Số 3 (MỚI)
                    elif fingers[0]==1 and fingers[1]==1 and fingers[4]==1: gesture = "LOVE"  # Love (MỚI)
                    elif fingers[2]==1 and fingers[3]==1 and fingers[4]==1: # OK Sign
                        dist_thumb_index = abs(hand_lm.landmark[4].x - hand_lm.landmark[8].x)
                        if dist_thumb_index < 0.05: gesture = "OK_SIGN"
                
        return gesture

    def process_frame(self, frame):
        display_frame = frame.copy()
        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(frame)
        user_name = "Unknown"
        
        if faces is not None:
            for face in faces:
                box = list(map(int, face[:4]))
                landmarks = face[4:14].reshape((5, 2))
                is_good, msg = self.check_face_quality(frame, face[:4], landmarks)
                if is_good:
                    face_align = self.recognizer.alignCrop(frame, face)
                    face_feature = self.recognizer.feature(face_align)
                    max_score = 0.0
                    for name, db_feature in self.face_db.items():
                        score = self.recognizer.match(face_feature, db_feature, cv2.FaceRecognizerSF_FR_COSINE)
                        if score > max_score:
                            max_score = score
                            if score > self.threshold_cosine: user_name = name
                    color = (0, 255, 0) if user_name != "Unknown" else (255, 255, 0)
                    cv2.rectangle(display_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
                    cv2.putText(display_frame, f"{user_name} ({max_score:.2f})", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                else:
                    cv2.rectangle(display_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gesture = self.detect_gesture(frame_rgb)
        if gesture != "None":
            cv2.putText(display_frame, f"CMD: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        return display_frame, user_name, gesture

    def register_user(self, frame, name):
        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is not None:
            face = faces[0]
            landmarks = face[4:14].reshape((5, 2))
            is_good, msg = self.check_face_quality(frame, face[:4], landmarks)
            if not is_good: return False
            face_align = self.recognizer.alignCrop(frame, face)
            face_feature = self.recognizer.feature(face_align)
            self.face_db[name] = face_feature
            self.save_database()
            return True
        return False