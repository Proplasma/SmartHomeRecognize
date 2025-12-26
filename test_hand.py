"""
Script test nhận diện cử chỉ tay độc lập (không cần nhận diện khuôn mặt)

Chức năng:
- Mở camera
- Phát hiện bàn tay
- Đếm số ngón tay duỗi
- Hiển thị cử chỉ (OPEN_HAND = 5 ngón, FIST = 0 ngón)
- Hiển thị các điểm khớp tay trên ảnh (skeleton)
"""

import cv2
import mediapipe as mp

# ============= KHỞI TẠO MEDIAPIPE =============
# Khởi tạo MediaPipe (Chuyên gia về tay)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Dùng để vẽ skeleton tay

# ===== CẤU HÌNH MEDIAPIPE =====
# Cấu hình: model_complexity=0 là chế độ "Siêu Nhẹ" cho máy yếu
hands = mp_hands.Hands(
    static_image_mode=False,          # False = Video mode (tối ưu cho video stream)
    max_num_hands=1,                  # Chỉ nhận diện 1 tay cho đỡ lag
    model_complexity=0,               # QUAN TRỌNG: 0 = Lite, 1 = Full
    min_detection_confidence=0.5,     # Ngưỡng tin cậy phát hiện
    min_tracking_confidence=0.5       # Ngưỡng tin cậy theo dõi
)


def count_fingers(landmarks):
    """Đếm số ngón tay đang duỗi lên
    
    Cách hoạt động:
    - MediaPipe cung cấp 21 điểm khớp tay
    - Mỗi ngón tay có 4 điểm (khớp + đầu)
    - So sánh vị trí đầu ngón với khớp dưới để xác định duỗi hay gập
    
    Args:
        landmarks: 21 điểm khớp tay từ MediaPipe
        
    Returns:
        int: Số ngón tay duỗi (0-5)
    """
    # Các điểm đầu ngón tay: [Ngón Trỏ, Giữa, Áp Út, Út]
    # Lưu ý: Ngón cái check riêng vì nó chuyển động ngang
    finger_tips = [8, 12, 16, 20] 
    count = 0
    
    # ===== CHECK 4 NGÓN DÀI =====
    # Nếu đầu ngón tay cao hơn khớp dưới -> là đang duỗi
    # Lưu ý: Trục Y trong ảnh tính từ trên xuống dưới (trên cùng y=0)
    for tip in finger_tips:
        # tip = điểm đầu ngón tay
        # tip-2 = khớp dưới ngón tay
        if landmarks[tip].y < landmarks[tip - 2].y:
            # Đầu ngón cao hơn khớp = duỗi
            count += 1
            
    # ===== CHECK NGÓN CÁI =====
    # Ngón cái so sánh ngang (x): nếu đưa sang ngang xa hơn khớp -> duỗi
    # Cái này check cho tay phải, tay trái sẽ ngược lại xíu (tạm thời test tay phải)
    if landmarks[4].x < landmarks[3].x:
        count += 1
        
    return count


# ============= MỞ CAMERA =============
cap = cv2.VideoCapture(0)

print("Đưa tay lên trước camera nào...")

# ============= VÒNG LẶP CHÍNH =============
while True:
    success, frame = cap.read()
    if not success:
        break  # Thoát nếu camera bị lỗi
    
    # ===== BƯỚC 1: CHUẨN BỊ ẢNH =====
    # MediaPipe cần ảnh màu RGB, còn OpenCV dùng BGR -> Phải đổi màu
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ===== BƯỚC 2: PHÁT HIỆN TÂY =====
    # Xử lý tìm bàn tay bằng MediaPipe
    results = hands.process(frame_rgb)
    
    # ===== BƯỚC 3: XỬ LÝ KẾT QUẢ =====
    # Nếu thấy bàn tay
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # ===== VẼ SKELETON TÂY =====
            # Vẽ các đốt xương tay lên hình (21 điểm + các đường nối)
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS  # Vẽ các đường nối giữa các khớp
            )
            
            # ===== ĐẾM NGÓN TAY =====
            # Gọi hàm đếm số ngón duỗi
            fingers = count_fingers(hand_landmarks.landmark)
            
            # ===== ĐẶT TRẠNG THÁI =====
            # Xác định cử chỉ dựa vào số ngón duỗi
            status_text = f"Ngon tay: {fingers}"  # Mặc định hiển thị số ngón
            color = (0, 0, 255)  # Đỏ
            
            if fingers == 5:
                # 5 ngón duỗi = MỞ TÂY
                status_text = "MO TAY -> BAT DEN"
                color = (0, 255, 0)  # Xanh
            elif fingers == 0:
                # 0 ngón duỗi = NẮM TẶNG
                status_text = "NAM TAY -> TAT DEN"
                color = (0, 255, 255)  # Vàng

            # ===== HIỂN THỊ TEXT LÊN ẢNH =====
            cv2.putText(
                frame, 
                status_text, 
                (10, 50),  # Vị trí (x, y)
                cv2.FONT_HERSHEY_SIMPLEX, 
                1,        # Font size
                color,    # Màu
                2         # Độ dày
            )

    # ===== BƯỚC 4: HIỂN THỊ =====
    cv2.imshow("Smart Home AI - Hand Gesture Test", frame)
    
    # ===== BƯỚC 5: THOÁT =====
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============= ĐÓNG =============
cap.release()
cv2.destroyAllWindows()