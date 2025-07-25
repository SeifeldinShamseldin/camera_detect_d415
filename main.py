#!/usr/bin/env python3

import sys
import os
import logging
import signal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import pyrealsense2 as rs
    import cv2
    import numpy as np
    from loguru import logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)


class CameraDetectionSystem:
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.running = False
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging system"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            log_dir / "camera_detection.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO"
        )
        logger.info("Camera Detection System initialized")
    
    def initialize_camera(self):
        """Initialize Intel RealSense D415 camera"""
        try:
            logger.info("Initializing Intel RealSense D415 camera...")
            
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Get device info
            device = profile.get_device()
            logger.info(f"Camera initialized: {device.get_info(rs.camera_info.name)}")
            logger.info(f"Serial number: {device.get_info(rs.camera_info.serial_number)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def process_frames(self):
        """Main processing loop"""
        logger.info("Starting frame processing...")
        
        # Create alignment object
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        try:
            while self.running:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                
                # Align depth frame to color frame
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Apply colormap to depth image for visualization
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # Stack images horizontally
                images = np.hstack((color_image, depth_colormap))
                
                # Display
                cv2.namedWindow('Intel RealSense D415 - RGB & Depth', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Intel RealSense D415 - RGB & Depth', images)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("Exit requested by user")
                    break
                    
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        if self.pipeline:
            self.pipeline.stop()
            
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")
    
    def signal_handler(self, signum, frame):
        """Handle system signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def run(self):
        """Main application entry point"""
        logger.info("Starting Camera Detection System...")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Initialize camera
        if not self.initialize_camera():
            logger.error("Failed to initialize camera. Exiting.")
            return 1
        
        # Set running flag
        self.running = True
        
        try:
            # Start processing
            self.process_frames()
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return 1
        
        logger.info("Camera Detection System stopped")
        return 0


def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'pyrealsense2',
        'cv2',
        'numpy',
        'loguru'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"Missing required modules: {', '.join(missing_modules)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    return True


def check_camera_connection():
    """Check if RealSense camera is connected"""
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("No RealSense devices found!")
            print("Please ensure Intel RealSense D415 is connected via USB 3.0")
            return False
        
        for device in devices:
            print(f"Found device: {device.get_info(rs.camera_info.name)}")
            print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
        
        return True
        
    except Exception as e:
        print(f"Error checking camera connection: {e}")
        return False


def main():
    """Main entry point"""
    print("=" * 60)
    print("Intel RealSense D415 Camera Detection System")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check camera connection
    if not check_camera_connection():
        return 1
    
    # Create and run application
    app = CameraDetectionSystem()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())