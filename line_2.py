import cv2
import numpy as np

def detect_black_regions(frame):
    height, width, _ = frame.shape
    section_height = height // 8  # แบ่งภาพออกเป็น 8 ส่วนแนวนอน
    centroids = []  # ลิสต์เก็บ centroid ที่พบ
    edges = []  # ลิสต์เก็บขอบซ้ายและขวา

    # กำหนดช่องตรงกลาง (3 ช่องในแนวนอน แต่ละช่องกว้างไม่เกิน 40px)
    middle_width = 70  # ความกว้างของช่องตรงกลางในแนวนอน
    center_x_min = (width // 2) - middle_width  # ขอบซ้ายของช่องตรงกลาง
    center_x_max = (width // 2) + middle_width  # ขอบขวาของช่องตรงกลาง
    
    # วาดกรอบช่องตรงกลาง
    cv2.rectangle(frame, (center_x_min, 0), (center_x_max, height), (0, 255, 255), 2)  # วาดกรอบช่องตรงกลางเป็นสีเหลือง

    # วาดจุด centroid ของพื้นที่สีดำ และจุดขอบซ้ายขวา
    for i in range(8):
        y_start = i * section_height
        y_end = (i + 1) * section_height if i < 7 else height  # ส่วนสุดท้ายอาจไม่พอดี
        section = frame[y_start:y_end, :]
        
        gray = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, 85, 100, cv2.THRESH_BINARY_INV)
        
        # หา contour ของพื้นที่สีดำ
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame[y_start:y_end, :], contours, -1, (0, 255, 0), 2)  # วาดเส้นสีเขียว
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(black_mask)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"]) + y_start
                centroids.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (255, 255, ), -1)  # วาดจุดสีน้ำเงินที่ centroid
                left = x
                right = x + w
                cv2.circle(frame, (left, cy), 5, (0, 0, 255), -1)  # แสดงจุดขอบซ้าย (สีแดง)
                cv2.circle(frame, (right, cy), 5, (0, 0, 255), -1)  # แสดงจุดขอบขวา (สีแดง)
                cv2.line(frame, (left, cy), (right, cy), (0, 0, 255), 2)
                distance = right - left
                

    Axis = [0, 0, 0]
    if len(centroids) >= 5:
        centroids_sorted = sorted(centroids, key=lambda c: (c[0] - width//2)**2 + (c[1] - height//2)**2)
        closest_centroids = centroids_sorted[:3]
        avg_x = np.mean([c[0] for c in closest_centroids])
        

        # คำนวณค่า z
        if abs(avg_x - width // 2) <= middle_width:
            Axis[0] = 1
        elif distance > 250 :
            Axis[0] = "Stop"
        elif avg_x < width // 2:
            Axis[2] = -1
        else:
            Axis[2] = 1
        
    # แสดงค่า x, y, z
    cv2.putText(frame, f"x = {Axis[0]}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"y = {Axis[1]}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"z = {Axis[2]}", (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame

def main():
    cap = cv2.VideoCapture(1)  # ใช้กล้องเว็บแคม
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = detect_black_regions(frame)
        cv2.imshow('Black Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
