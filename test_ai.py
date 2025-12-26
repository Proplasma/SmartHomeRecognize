"""
Script test tích hợp AI - Kiểm tra toàn bộ chức năng trong chế độ Desktop
(Không qua web server)

Chức năng:
- Mở camera
- Nhận diện khuôn mặt + cử chỉ tay
- Đăng ký người dùng mới (Nút 's')
- Hiển thị lệnh điều khiển khi phát hiện cử chỉ
- Thoát (Nút 'q')
"""

import cv2
from ai_core import SmartHomeAI  # Import file em vừa tạo

# ============= KHỞI TẠO =============
my_ai = SmartHomeAI()  # Khởi tạo bộ não AI
cap = cv2.VideoCapture(0)  # Mở camera

print("Nhấn 's' để lưu mặt Admin. Nhấn 'q' để thoát.")

# ============= VÒNG LẶP CHÍNH =============
while True:
    # Đọc frame từ camera
    ret, frame = cap.read()
    if not ret:
        break  # Thoát nếu camera bị lỗi

    # ===== BƯỚC 1: GỌI AI XỬ LÝ FRAME =====
    # Gọi hàm xử lý từ ai_core
    # Nó trả về 3 thứ: Ảnh đã vẽ, Tên người, Cử chỉ
    processed_frame, user, gesture = my_ai.process_frame(frame)
    
    # ===== BƯỚC 2: XỬ LÝ LOGIC ĐIỀU KHIỂN =====
    # Kiểm tra nếu có người được nhận diện và phát hiện cử chỉ
    if user != "Unknown":
        if gesture == "OPEN_HAND":
            # Tay mở = BẬT ĐÈN
            print(f"!!! Gửi lệnh BẬT ĐÈN cho {user}")
        elif gesture == "FIST":
            # Tay gập = TẮT ĐÈN
            print(f"!!! Gửi lệnh TẮT ĐÈN cho {user}")

    # ===== BƯỚC 3: HIỂN THỊ ẢNH =====
    cv2.imshow("Test Integration", processed_frame)
    
    # ===== BƯỚC 4: NHẬN TỪ PHÍM BẤM =====
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        # Nhấn 'q' = Thoát chương trình
        break
    elif key & 0xFF == ord('s'):
        # Nhấn 's' = Đăng ký khuôn mặt hiện tại
        # Thử đăng ký khuôn mặt
        if my_ai.register_user(frame, "Admin_Do"):
            print(">>> Đã lưu Admin_Do thành công!")

# ============= ĐÓNG =============
cap.release()
cv2.destroyAllWindows()