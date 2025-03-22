import cv2
import socket
import struct
import time
import rclpy
from rclpy.node import Node
import os
import threading
from std_msgs.msg import Int32

class VideoStreamClient(Node):
    def __init__(self):
        super().__init__("video_stream_client")
        self.server_ip = "192.168.234.66"  # เปลี่ยนเป็น IP ของเซิร์ฟเวอร์
        self.server_port = 8080
        self.camera_list = []
        
        self.camera_index = 0  # Set default camera index to 0 (or any valid index)
        self.find_available_camera()
        print(self.camera_list)

        self.cap = cv2.VideoCapture(self.camera_list[self.camera_index])
        
        # Set resolution and frame rate for all cameras
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.lock = threading.Lock()
        self.client_socket = self.connect_to_server()
        
        # ROS2 Subscriber to change camera index
        self.subscription = self.create_subscription(
            Int32,
            '/camera_switch',
            self.camera_switch_callback,
            10
        )
        
        # Use a timer to send video frames instead of a blocking loop
        self.timer = self.create_timer(1/30, self.send_video_frame)  # 30 FPS

    def find_available_camera(self, max_index=5):
        for index in range(max_index):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.release()
                self.camera_list.append(index) 
        self.get_logger().info(str(self.camera_list))

    def connect_to_server(self):
        while rclpy.ok():
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((self.server_ip, self.server_port))
                self.get_logger().info("Connected to server.")
                return client_socket
            except (socket.error, ConnectionRefusedError):
                self.get_logger().warn("Connection failed. Retrying in 3 seconds...")
                time.sleep(3)

    def camera_switch_callback(self, msg):
        new_index = msg.data
        self.get_logger().info(f"Switching to camera index: {new_index}")
        with self.lock:  # Ensure thread safety
            if new_index != self.camera_index:
                # Release the current camera if it's open
                if self.cap.isOpened():
                    self.cap.release()
                
                # Attempt to open the new camera
                self.cap = cv2.VideoCapture(self.camera_list[new_index])
                
                if not self.cap.isOpened():
                    self.get_logger().error(f"Failed to open camera {new_index}")
                else:
                    # Set resolution and frame rate for the new camera
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    self.camera_index = new_index
                    
                    self.get_logger().info(f"Successfully switched to camera {new_index}")
            else:
                self.get_logger().info(f"Requested index {new_index} is already active")

    def send_video_frame(self):
        with self.lock:
            ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().error("Failed to capture frame.")
            return
        
        _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = encoded_frame.tobytes()
        size = struct.pack("Q", len(data))
        
        try:
            self.client_socket.sendall(size + data)
        except (socket.error, BrokenPipeError):
            self.get_logger().warn("Connection lost. Reconnecting...")
            self.client_socket.close()
            self.client_socket = self.connect_to_server()

    def destroy_node(self):
        with self.lock:
            if self.cap.isOpened():
                self.cap.release()
        self.client_socket.close()
        super().destroy_node()

def main(args=None):
    try:
        rclpy.init(args=args)
        print("ROS 2 initialized")
    except Exception as e:
        print(f"Error initializing ROS 2: {e}")
        return

    node = VideoStreamClient()
    print("Node created, spinning...")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down client.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
