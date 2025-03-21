import rclpy
from rclpy.node import Node
import cv2
import socket
import struct
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from std_msgs.msg import Int32
from ultralytics import YOLO
from geometry_msgs.msg import Vector3
model = YOLO("/home/jonut/dentist/src/robot/train2/weights/best.pt")

class RegionDetector(Node):
    def __init__(self):
        super().__init__('region_detector')
        self.bridge = CvBridge()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("0.0.0.0", 8080))
        self.server_socket.listen(5)
        self.get_logger().info("Waiting for connection...")

        self.cnt_vel_pub = self.create_publisher(Vector3, 'cnt_vel', 10)
        self.timer = self.create_timer(0.05, self.cnt_vel_pub.publish)
        self.last_axis = Vector3()  # Store last Axis value   

        self.camera_switch_publisher = self.create_publisher(Int32, 'camera_switch', 10)  # Use Int32 here
        self.image_publisher = self.create_publisher(Image, 'processed_image', 10)
        self.black_state = True
        self.color_state = False
        self.turnaround_state = False
        self.pick_state = False
        self.current_state = " "
        self.receive_video() 

    def filter_noise(self, contours, min_area=500):
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:  # Only keep contours with an area above the threshold
                filtered_contours.append(contour)
        return filtered_contours

    def detect_blue_regions(self, frame):
        blue_data = {
            'centroids': [],
            'areas': [],
            'bounding_boxes': []
        }
        height, width, _ = frame.shape
        section_height = height // 8

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([130, 255, 255])


        blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        for i in range(8):
            y_start = i * section_height
            y_end = (i + 1) * section_height if i < 7 else height
            blue_mask_section = blue_mask[y_start:y_end, :]

            contours, _ = cv2.findContours(blue_mask_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = self.filter_noise(contours, min_area=500)

            if contours:
                cv2.drawContours(frame[y_start:y_end, :], contours, -1, (255, 0, 0), 2)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                moments = cv2.moments(contour)
                area = cv2.contourArea(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"]) + y_start
                    blue_data['centroids'].append((cx, cy))
                    blue_data['areas'].append(area)
                    blue_data['bounding_boxes'].append((x, y + y_start, w, h))
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        return frame, blue_data

    def detect_green_regions(self, frame):
        green_data = {
            'centroids': [],
            'areas': [],
            'bounding_boxes': []
        }
        height, width, _ = frame.shape
        section_height = height // 8

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 150, 150])

        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        for i in range(8):
            y_start = i * section_height
            y_end = (i + 1) * section_height if i < 7 else height
            green_mask_section = green_mask[y_start:y_end, :]

            contours, _ = cv2.findContours(green_mask_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = self.filter_noise(contours, min_area=500)

            if contours:
                cv2.drawContours(frame[y_start:y_end, :], contours, -1, (0, 255, 0), 2)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                moments = cv2.moments(contour)
                area = cv2.contourArea(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"]) + y_start
                    green_data['centroids'].append((cx, cy))
                    green_data['areas'].append(area)
                    green_data['bounding_boxes'].append((x, y + y_start, w, h))
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        return frame, green_data

    def detect_red_regions(self, frame):
        red_data = {
            'centroids': [],
            'areas': [],
            'bounding_boxes': []
        }
        height, width, _ = frame.shape
        section_height = height // 8

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red_1 = np.array([0, 100, 100])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 100, 100])
        upper_red_2 = np.array([180, 255, 255])

        red_mask_1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
        red_mask_2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        for i in range(8):
            y_start = i * section_height
            y_end = (i + 1) * section_height if i < 7 else height
            red_mask_section = red_mask[y_start:y_end, :]

            contours, _ = cv2.findContours(red_mask_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = self.filter_noise(contours, min_area=500)
            if contours:
                cv2.drawContours(frame[y_start:y_end, :], contours, -1, (0, 0, 255), 2)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                moments = cv2.moments(contour)
                area = cv2.contourArea(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"]) + y_start
                    red_data['centroids'].append((cx, cy))
                    red_data['areas'].append(area)
                    red_data['bounding_boxes'].append((x, y + y_start, w, h))
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        return frame, red_data

    def detect_black_regions(self, frame):
        black_data = {
            'centroids': [],
            'areas': [],
            'bounding_boxes': []
        }
        height, width, _ = frame.shape
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 110])

        for i in range(8):
            y_start = i * (height // 8)
            y_end = (i + 1) * (height // 8) if i < 7 else height
            section = frame[y_start:y_end, :]
            hsv_section = hsv_frame[y_start:y_end, :]

            black_mask = cv2.inRange(hsv_section, lower_black, upper_black)
            contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = self.filter_noise(contours, min_area=500)
            cv2.drawContours(frame[y_start:y_end, :], contours, -1, (0, 255, 0), 2)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                moments = cv2.moments(contour)
                area = cv2.contourArea(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"]) + y_start
                    black_data['centroids'].append((cx, cy))
                    black_data['areas'].append(area)
                    black_data['bounding_boxes'].append((x, y_start + y, w, h))
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        return frame, black_data
    
    def detect_objects(self, frame):
        results = model(frame)
        processed_frame = frame.copy()

        # คำนวณตำแหน่งของ crosshair (จุดกึ่งกลางภาพ)
        h, w, _ = processed_frame.shape
        center_x, center_y = w // 2, h // 2

        # วาด crosshair
        cv2.line(processed_frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 0), 2)  # เส้นแนวนอน
        cv2.line(processed_frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 0), 2)  # เส้นแนวตั้ง

        # วาด Bounding Box และ Centroid
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names[cls]} {conf:.2f}"

                # คำนวณ Centroid
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # คำนวณค่าความคลาดเคลื่อน (error) ในแกน X
                error_x = center_x - cx  # ค่าความต่างระหว่าง crosshair และ centroid

                # กำหนดค่า Axis ตามเงื่อนไขที่กำหนด
                if abs(error_x) <= 10:
                    axis = 0  # อยู่ในช่วง error ±10
                elif center_x > cx:
                    axis = -1  # crosshair อยู่ทางขวาของ centroid
                else:
                    axis = 1  # crosshair อยู่ทางซ้ายของ centroid

                axis_label = f"Axis: {axis}"

                # วาดกรอบสี่เหลี่ยมและ Centroid
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(processed_frame, (cx, cy), 5, (0, 0, 255), -1)  # จุดสีแดง (Centroid)
                cv2.putText(processed_frame, f"({cx},{cy})", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # วาดเส้นจาก Centroid ไปยัง Crosshair
                cv2.line(processed_frame, (cx, cy), (center_x, center_y), (0, 255, 255), 2)

                # แสดงค่าของ Axis
                cv2.putText(processed_frame, axis_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return processed_frame , axis
    
    def state(self, black_data, color_data, frame):
        self.current_state = "black_state"
        black_centroid = black_data['centroids']
        black_areas = black_data['areas']
        black_bounding_boxes = black_data['bounding_boxes']
        
        color_centroid = color_data['centroids']
        color_areas = color_data['areas']
        color_bounding_boxes = color_data['bounding_boxes']
        switch_msg = Int32()
        height, width, _ = frame.shape
        Axis = [0, 0, 0]
        total_area = 0
        middle_width = 20
        center_x_min = (width // 2) - middle_width
        center_x_max = (width // 2) + middle_width
        cv2.line(frame, (center_x_min, 0), (center_x_min, height), (0, 255, 255), 2)  # เส้นซ้าย
        cv2.line(frame, (center_x_max, 0), (center_x_max, height), (0, 255, 255), 2)  # เส้นขวา

        section_height = height // 8
        lower_region_start = height - (3 * section_height)
        red_in_lower_region = any(cy >= lower_region_start for _, cy in color_centroid)
        
        if self.black_state:
            if not black_centroid :
                        Axis[2] = 1  
                        return Axis, frame

            if len(black_centroid) >= 3:
                if red_in_lower_region:
                    self.current_state = "color_state"
                    self.color_state = True
                    self.black_state = False   
                else:
                    centroids_sorted = sorted(black_centroid, key=lambda c: c[1], reverse=True)
                    closest_centroids = centroids_sorted[:3]
                    total_area = sum([black_areas[i] for i in range(3)])
                    avg_x = np.mean([c[0] for c in closest_centroids])

                    if total_area > 24000:
                        Axis[0] = 0
                    elif abs(avg_x - width // 2) <= middle_width:
                        Axis[0] = 1
                    elif avg_x < width // 2:
                        Axis[2] = -1
                    else:
                        Axis[2] = 1

                    for i, centroid in enumerate(closest_centroids):
                        cx, cy = centroid
                        x, y, w, h = black_bounding_boxes[i]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        elif self.color_state:
            if not color_centroid :
                        Axis[2] = 1  
                        return Axis, frame
            if len(color_centroid) >= 3:
                centroids_sorted = sorted(color_centroid, key=lambda c: c[1], reverse=True)
                closest_centroids = centroids_sorted[:3]
                total_area = sum([color_areas[i] for i in range(3)])
                avg_x = np.mean([c[0] for c in closest_centroids])

                if total_area > 24000:
                    self.current_state = "pick_state"
                    switch_msg.data = 1  # Set camera ID to 1
                    self.camera_switch_publisher.publish(switch_msg)
                    self.wait_for_publish_success()
                    self.pick_state = True
                    self.color_state = False
                elif abs(avg_x - width // 2) <= middle_width:
                    Axis[0] = 1
                elif avg_x < width // 2:
                    Axis[2] = -1
                else:
                    Axis[2] = 1

                for i, centroid in enumerate(closest_centroids):
                    cx, cy = centroid
                    x, y, w, h = color_bounding_boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)  # วาดกรอบสีขาว
        
        self.last_axis = Vector3(x=float(Axis[0]), y=float(Axis[1]), z=float(Axis[2]))
        self.cnt_vel_pub.publish(self.last_axis)
        cv2.putText(frame, f"x = {Axis[0]}", (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"y = {Axis[1]}", (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"z = {Axis[2]}", (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"area = {total_area}", (5, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"state = {self.current_state}", (5, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"camera = {switch_msg.data}", (5, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Processed Video", frame)
        cv2.waitKey(1)
        
        return Axis, frame
    
    def detect_objects(self, state ,frame):
        results = model(frame)
        processed_frame = frame.copy()

        # คำนวณตำแหน่งของ crosshair (จุดกึ่งกลางภาพ)
        h, w, _ = processed_frame.shape
        center_x, center_y = w // 2, h // 2

        # วาด crosshair
        cv2.line(processed_frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 0), 2)  # เส้นแนวนอน
        cv2.line(processed_frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 0), 2)  # เส้นแนวตั้ง

        # วาด Bounding Box และ Centroid
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names[cls]} {conf:.2f}"

                # คำนวณ Centroid
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # คำนวณค่าความคลาดเคลื่อน (error) ในแกน X
                error_x = center_x - cx  # ค่าความต่างระหว่าง crosshair และ centroid

                # กำหนดค่า Axis ตามเงื่อนไขที่กำหนด
                if abs(error_x) <= 10:
                    axis = 0  # อยู่ในช่วง error ±10
                elif center_x > cx:
                    axis = -1  # crosshair อยู่ทางขวาของ centroid
                else:
                    axis = 1  # crosshair อยู่ทางซ้ายของ centroid

                axis_label = f"Axis: {axis}"

                # วาดกรอบสี่เหลี่ยมและ Centroid
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(processed_frame, (cx, cy), 5, (0, 0, 255), -1)  # จุดสีแดง (Centroid)
                cv2.putText(processed_frame, f"({cx},{cy})", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # วาดเส้นจาก Centroid ไปยัง Crosshair
                cv2.line(processed_frame, (cx, cy), (center_x, center_y), (0, 255, 255), 2)

                # แสดงค่าของ Axis
                cv2.putText(processed_frame, axis_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return processed_frame

    def wait_for_publish_success(self):
        rclpy.spin_once(self, timeout_sec=2.0)

    def receive_video(self):
        while rclpy.ok():
            conn, addr = self.server_socket.accept()
            self.get_logger().info(f"Connected to {addr}")
            data = b""
            payload_size = struct.calcsize("Q")

            try:
                while True:
                    while len(data) < payload_size:
                        packet = conn.recv(4096)
                        if not packet:
                            self.get_logger().info("Client disconnected.")
                            return
                        data += packet

                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack("Q", packed_msg_size)[0]

                    while len(data) < msg_size:
                        data += conn.recv(4096)

                    frame_data = data[:msg_size]
                    data = data[msg_size:]

                    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if self.pick_state:
                        processed_frame = self.detect_objects(self.state, frame)
                    else:
                        processed_frame, blue_data = self.detect_blue_regions(frame)
                        processed_frame, green_data = self.detect_green_regions(processed_frame)
                        processed_frame, red_data = self.detect_red_regions(processed_frame)
                        processed_frame, black_data = self.detect_black_regions(processed_frame)
                        Axis = self.state(black_data, red_data, processed_frame)

                    ros_image = self.bridge.cv2_to_imgmsg(processed_frame, encoding='bgr8')
                    self.image_publisher.publish(ros_image)

                    cv2.imshow("Processed Video", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.get_logger().info("Stopping stream...")
                        return
            except (ConnectionResetError, BrokenPipeError):
                self.get_logger().info("Connection lost. Waiting for new connection...")
            finally:
                conn.close()
                cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = RegionDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
