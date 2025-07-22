#!/usr/bin/env python3
# filepath: /home/robotica/agro_ws/src/summit_agriculture_fork/furrow_following/furrow_following/furow_follower_green.py

"""
Green Line Follower Node for agricultural robots.
This ROS2 node detects and follows green lines (crop rows) using computer vision.
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class LineFollowerNode(Node):
    """
    A ROS2 node that detects and follows green lines using a camera.
    Publishes visualization images and velocity commands.
    """
    
    def __init__(self):
        """Initialize the line follower node with parameters and publishers/subscribers."""
        super().__init__('line_follower_node')
        
        # Initialize the CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        self.cmd_pub = self.create_publisher(Twist, '/robot/robotnik_base_control/cmd_vel', 10)
        self.visual_pub = self.create_publisher(Image, '/line_follower/visualization', 10)
        
        # Set control frequency to 20Hz for better responsiveness
        self.timer = self.create_timer(0.05, self.control_loop)

        # State variables
        self.cx = None  # Current detected line center x position
        self.image_width = 640
        
        # Declare adjustable control parameters - VELOCIDADES REDUCIDAS
        self.declare_parameter('linear_speed', 0.3)     
        self.declare_parameter('angular_gain', 0.005)   
        self.declare_parameter('search_speed', 0.2)      
        
        # Parámetros para recorte de imagen al centro - MÁS RECORTADO
        self.declare_parameter('crop_width_percent', 0.2)  
        self.declare_parameter('crop_height_percent', 0.7)

    def fit_line_to_green(self, mask, roi_height, roi_width):
        """
        Encuentra y ajusta una línea a los puntos verdes detectados
        """
        # Encontrar todos los puntos no-cero (píxeles verdes)
        green_points = np.column_stack(np.where(mask > 0))
        
        if len(green_points) < 10:  # Necesitamos al menos 10 puntos
            return None, None
            
        # Convertir a formato (x, y) para cv2.fitLine
        points = green_points[:, [1, 0]].astype(np.float32)  # Intercambiar x,y
        
        # Ajustar línea usando método de mínimos cuadrados
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calcular puntos de la línea en los extremos de la ROI
        # Punto superior (y = 0)
        t_top = -y0 / vy if vy != 0 else 0
        x_top = int(x0 + vx * t_top)
        
        # Punto inferior (y = roi_height)
        t_bottom = (roi_height - y0) / vy if vy != 0 else 0
        x_bottom = int(x0 + vx * t_bottom)
        
        # Asegurar que los puntos estén dentro de la ROI
        x_top = max(0, min(roi_width - 1, x_top))
        x_bottom = max(0, min(roi_width - 1, x_bottom))
        
        # Punto central de la línea (en la mitad vertical de la ROI)
        y_center = roi_height // 2
        t_center = (y_center - y0) / vy if vy != 0 else 0
        x_center = int(x0 + vx * t_center)
        x_center = max(0, min(roi_width - 1, x_center))
        
        return (x_top, 0), (x_bottom, roi_height - 1), x_center

    def find_green_endpoints(self, mask, roi_height, roi_width):
        """
        Find two points - one at the bottom section and one at the top section of the green line,
        preferring points closer to the center of the image.
        
        Args:
            mask: Binary mask with green pixels
            roi_height: Height of the ROI
            roi_width: Width of the ROI
            
        Returns:
            top_point, bottom_point, center_x: Coordinates of top point, bottom point and center x
        """
        # Define top and bottom sections (divide the ROI into sections)
        top_section_height = int(roi_height * 0.3)  # Top 30% of the ROI
        bottom_section_height = int(roi_height * 0.3)  # Bottom 30% of the ROI
        
        # Get the masks for each section
        top_mask = mask[:top_section_height, :]
        bottom_mask = mask[roi_height - bottom_section_height:, :]
        
        # Find non-zero pixels in each section
        top_points = np.column_stack(np.where(top_mask > 0))
        bottom_points = np.column_stack(np.where(bottom_mask > 0))
        
        # If we don't have enough points in either section, fallback to original method
        if len(top_points) < 10 or len(bottom_points) < 10:
            return self.fit_line_to_green(mask, roi_height, roi_width)
        
        # Find center x coordinate of the ROI
        center_x = roi_width // 2
        
        # Find points closest to the center in top section
        top_x_coords = top_points[:, 1]  # x coordinates
        top_distances = np.abs(top_x_coords - center_x)
        top_center_idx = np.argmin(top_distances)
        top_center_point = (int(top_points[top_center_idx, 1]), int(top_points[top_center_idx, 0]))
        
        # Find points closest to the center in bottom section
        bottom_x_coords = bottom_points[:, 1]  # x coordinates
        bottom_distances = np.abs(bottom_x_coords - center_x)
        bottom_center_idx = np.argmin(bottom_distances)
        # Adjust y coordinates to the full ROI
        bottom_y = bottom_points[bottom_center_idx, 0] + (roi_height - bottom_section_height)
        bottom_center_point = (int(bottom_points[bottom_center_idx, 1]), int(bottom_y))
        
        # Use the average of the x coordinates for control
        control_center_x = int((top_center_point[0] + bottom_center_point[0]) / 2)
        
        return top_center_point, bottom_center_point, control_center_x
        
    def image_callback(self, msg):
        """
        Process camera images to detect green lines.
        
        Args:
            msg: ROS Image message from camera
        """
        # Convert ROS image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Save a copy for visualization
        visual_frame = frame.copy()
        
        # RECORTAR IMAGEN MÁS AL CENTRO
        height, width = frame.shape[:2]
        
        # Calcular dimensiones del recorte (más pequeñas)
        crop_width_percent = self.get_parameter('crop_width_percent').value
        crop_height_percent = self.get_parameter('crop_height_percent').value
        
        crop_width = int(width * crop_width_percent)
        crop_height = int(height * crop_height_percent)
        
        # Calcular coordenadas del recorte (centrado horizontalmente, parte inferior)
        start_x = (width - crop_width) // 2
        end_x = start_x + crop_width
        start_y = height - crop_height  # Desde la parte inferior
        end_y = height
        
        # Extraer región de interés recortada
        roi = frame[start_y:end_y, start_x:end_x]
        
        # Marcar la región de interés en la imagen de visualización
        cv2.rectangle(visual_frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
        cv2.putText(visual_frame, f"ROI {crop_width}x{crop_height}", (start_x + 5, start_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define multiple green color ranges for better detection
        # Create and combine multiple masks to catch different shades of green
        # Light yellowish-green
        lower_green1 = np.array([25, 40, 30])
        upper_green1 = np.array([45, 255, 255])
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # Medium green
        lower_green2 = np.array([45, 40, 30])
        upper_green2 = np.array([75, 255, 255])
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        
        # Bluish-green
        lower_green3 = np.array([75, 40, 30])
        upper_green3 = np.array([95, 255, 255])
        mask3 = cv2.inRange(hsv, lower_green3, upper_green3)
        
        # Combine all masks
        mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
        
        # Apply morphological operations to improve detection
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)  # Increased closing to better connect vertical lines
        
        # Visualize the mask in the ROI area
        visual_roi = visual_frame[start_y:end_y, start_x:end_x]
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.addWeighted(mask_colored, 0.3, visual_roi, 0.7, 0, visual_roi)

        # Centro de la ROI
        roi_center_x = crop_width // 2
        roi_height, roi_width = mask.shape
        
        # TRAZAR LÍNEA A TRAVÉS DE LA ZONA VERDE
        # Replace the fit_line_to_green call with find_green_endpoints
        line_points = self.find_green_endpoints(mask, roi_height, roi_width)
        
        if line_points[0] is not None and line_points[1] is not None:
            top_point, bottom_point, center_x = line_points
            
            # Ajustar coordenadas a la imagen completa para visualización
            full_top = (start_x + top_point[0], start_y + top_point[1])
            full_bottom = (start_x + bottom_point[0], start_y + bottom_point[1])
            full_center_x = start_x + center_x
            full_center_y = start_y + roi_height // 2
            
            # Dibujar la línea ajustada
            cv2.line(visual_frame, full_top, full_bottom, (255, 0, 255), 3)
            
            # Dibujar punto central de la línea
            cv2.circle(visual_frame, (full_center_x, full_center_y), 8, (255, 0, 255), -1)
            
            # Dibujar línea desde centro de ROI al centro de la línea detectada
            roi_center_full = (start_x + roi_center_x, start_y + roi_height // 2)
            cv2.line(visual_frame, roi_center_full, (full_center_x, full_center_y), (255, 255, 0), 2)
            
            # Update control point
            self.cx = center_x
            self.roi_center = roi_center_x
            
            # Display error
            error = center_x - roi_center_x
            cv2.putText(visual_frame, f"Line Error: {error}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(visual_frame, f"Line Following", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # No se pudo ajustar línea, intentar con contornos como respaldo
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 30]
                
                if valid_contours:
                    # Encontrar contorno más cercano al centro
                    best_cx = None
                    best_distance = float('inf')
                    
                    for cnt in valid_contours:
                        M = cv2.moments(cnt)
                        if M['m00'] > 0:
                            cx = int(M['m10'] / M['m00'])
                            distance = abs(cx - roi_center_x)
                            if distance < best_distance:
                                best_distance = distance
                                best_cx = cx
                    
                    if best_cx is not None:
                        self.cx = best_cx
                        self.roi_center = roi_center_x
                        
                        # Dibujar contornos
                        for cnt in valid_contours:
                            adjusted_cnt = cnt.copy()
                            adjusted_cnt[:, :, 0] += start_x
                            adjusted_cnt[:, :, 1] += start_y
                            cv2.drawContours(visual_frame, [adjusted_cnt], -1, (0, 255, 0), 2)
                        
                        error = best_cx - roi_center_x
                        cv2.putText(visual_frame, f"Contour Error: {error}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        self.cx = None
                else:
                    self.cx = None
            else:
                self.cx = None
        
        if self.cx is None:
            cv2.putText(visual_frame, "No green line detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Dibujar línea central de referencia
        roi_center_full = start_x + roi_center_x
        cv2.line(visual_frame, (roi_center_full, start_y), (roi_center_full, end_y), (128, 128, 128), 1)
        
        # Publish visualization image
        visual_msg = self.bridge.cv2_to_imgmsg(visual_frame, encoding='bgr8')
        self.visual_pub.publish(visual_msg)

    def control_loop(self):
        """Calculate and publish control commands based on line detection."""
        twist = Twist()
        
        # Get parameters from ROS
        linear_speed = self.get_parameter('linear_speed').value
        angular_gain = self.get_parameter('angular_gain').value
        search_speed = self.get_parameter('search_speed').value
        
        if self.cx is not None:
            # Line is detected - calculate error relative to ROI center
            error = self.cx - getattr(self, 'roi_center', self.image_width // 2)
            twist.linear.x = linear_speed
            twist.angular.z = -float(error) * angular_gain
        else:
            # No line detected - search pattern (velocidades reducidas)
            twist.linear.x = 0.1  # Muy lento durante búsqueda
            twist.angular.z = search_speed

        # Publish command velocity
        self.cmd_pub.publish(twist)


def main(args=None):
    """Main entry point for the line follower node."""
    rclpy.init(args=args)
    node = LineFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()