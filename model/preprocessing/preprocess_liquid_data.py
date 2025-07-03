import os
import cv2
import numpy as np
import json
from pathlib import Path
import pytesseract
from PIL import Image
import re
from tqdm import tqdm
import argparse
from datetime import datetime
import logging

# python preprocessing/preprocess_liquid_data.py . ./processed_data --frame-interval 5

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiquidVolumePreprocessor:
    def __init__(self, input_dir, output_dir, frame_interval=5):
        """
        Initialize the preprocessor.
        
        Args:
            input_dir: Path to model folder containing data subfolders
            output_dir: Path to output processed data
            frame_interval: Extract every nth frame (default=5)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata storage
        self.metadata = {
            'processed_date': datetime.now().isoformat(),
            'frame_interval': frame_interval,
            'samples': []
        }
    
    def find_video_files(self, data_folder):
        """Find RGB and grey camera video files in a data folder."""
        video_files = {
            'rgb': None,
            'grey_left': None,
            'grey_right': None
        }
        
        # Look for specific file names based on your naming convention
        cam_a_path = data_folder / 'CAM_A_video.mp4'
        cam_b_path = data_folder / 'CAM_B_mono.mp4'
        cam_c_path = data_folder / 'CAM_C_mono.mp4'
        
        if cam_a_path.exists():
            video_files['rgb'] = cam_a_path
        if cam_b_path.exists():
            video_files['grey_left'] = cam_b_path
        if cam_c_path.exists():
            video_files['grey_right'] = cam_c_path
        
        # Log found files
        logger.info(f"Found videos in {data_folder.name}:")
        for camera_type, path in video_files.items():
            if path:
                logger.info(f"  {camera_type}: {path.name}")
        
        return video_files
    
    def detect_scale_region(self, frame):
        """
        Detect the region containing the digital scale display.
        This is a simple implementation - you may need to adjust based on your setup.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get bright regions (assuming digital display is bright)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio
        potential_displays = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Digital displays are typically rectangular with aspect ratio > 2
            if 1000 < area < 50000 and 2 < aspect_ratio < 6:
                potential_displays.append((x, y, w, h))
        
        # Return the largest potential display region
        if potential_displays:
            return max(potential_displays, key=lambda r: r[2] * r[3])
        
        # Default region if detection fails (adjust based on your camera setup)
        h, w = frame.shape[:2]
        return (int(w*0.6), int(h*0.7), int(w*0.3), int(h*0.1))
    
    def save_scale_detection_sample(self, frame, scale_region, output_folder, data_id):
        """Save a sample image showing the detected scale region for debugging."""
        x, y, w, h = scale_region
        
        # Create a copy and draw the scale region
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_frame, "Scale Region", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save debug image
        debug_path = output_folder / f"{data_id}_scale_detection_sample.jpg"
        cv2.imwrite(str(debug_path), debug_frame)
        logger.info(f"Saved scale detection sample to: {debug_path.name}")
    
    def read_weight_from_frame(self, frame, scale_region=None):
        """
        Extract weight reading from the scale display using OCR.
        """
        if scale_region is None:
            scale_region = self.detect_scale_region(frame)
        
        x, y, w, h = scale_region
        
        # Extract scale region
        scale_roi = frame[y:y+h, x:x+w]
        
        # Preprocess for OCR
        # Convert to grayscale
        gray = cv2.cvtColor(scale_roi, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
        
        # Apply threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.medianBlur(thresh, 3)
        
        # Resize for better OCR (optional)
        scale_factor = 3
        resized = cv2.resize(denoised, None, fx=scale_factor, fy=scale_factor, 
                           interpolation=cv2.INTER_CUBIC)
        
        # OCR configuration for digits
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
        
        try:
            # Perform OCR
            text = pytesseract.image_to_string(resized, config=custom_config).strip()
            
            # Extract numeric value
            weight_match = re.search(r'(\d+\.?\d*)', text)
            if weight_match:
                weight = float(weight_match.group(1))
                return weight, scale_region
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
        
        return None, scale_region
    
    def extract_frames(self, video_path, output_folder, camera_type, data_id):
        """Extract frames from a video and save them."""
        if video_path is None or not video_path.exists():
            logger.error(f"Video path does not exist: {video_path}")
            return []
            
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_data = []
        frame_count = 0
        scale_region = None
        
        logger.info(f"Processing {camera_type} video: {video_path.name} ({total_frames} frames @ {fps:.2f} fps)")
        
        with tqdm(total=total_frames//self.frame_interval, desc=f"{camera_type} frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_interval == 0:
                    # Read weight from scale (only for RGB camera)
                    weight = None
                    if camera_type == 'rgb':
                        weight, scale_region = self.read_weight_from_frame(frame, scale_region)
                        if weight is not None:
                            logger.debug(f"Frame {frame_count}: weight = {weight}g")
                        
                        # Save scale detection sample on first frame
                        if frame_count == 0 and scale_region is not None:
                            self.save_scale_detection_sample(frame, scale_region, output_folder, data_id)
                    
                    # Save frame
                    frame_filename = f"{data_id}_{camera_type}_frame_{frame_count:06d}.jpg"
                    frame_path = output_folder / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    
                    # Store frame metadata
                    frame_info = {
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'filename': frame_filename,
                        'camera_type': camera_type,
                        'weight': weight
                    }
                    frame_data.append(frame_info)
                    
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        return frame_data
    
    def synchronize_frames(self, rgb_frames, grey_left_frames, grey_right_frames):
        """
        Synchronize frames from all three cameras and interpolate weights.
        """
        synchronized_data = []
        
        # Create lookup for RGB frames with weights
        rgb_weights = {}
        for frame in rgb_frames:
            if frame['weight'] is not None:
                rgb_weights[frame['frame_number']] = frame['weight']
        
        # If no weights were detected, log warning
        if not rgb_weights:
            logger.warning("No weights detected via OCR. Using default value of 0.0")
            # Set all frames to 0.0 weight
            for frame in rgb_frames:
                rgb_weights[frame['frame_number']] = 0.0
        else:
            # Interpolate missing weights
            frame_numbers = sorted(rgb_weights.keys())
            for i in range(len(frame_numbers) - 1):
                start_frame = frame_numbers[i]
                end_frame = frame_numbers[i + 1]
                start_weight = rgb_weights[start_frame]
                end_weight = rgb_weights[end_frame]
                
                # Linear interpolation for frames in between
                for frame_num in range(start_frame + self.frame_interval, end_frame, self.frame_interval):
                    if frame_num not in rgb_weights:
                        interpolated_weight = start_weight + (end_weight - start_weight) * \
                                            (frame_num - start_frame) / (end_frame - start_frame)
                        rgb_weights[frame_num] = interpolated_weight
        
        # Combine synchronized frames
        for i, rgb_frame in enumerate(rgb_frames):
            frame_num = rgb_frame['frame_number']
            
            # Find corresponding frames from other cameras
            grey_left_frame = next((f for f in grey_left_frames if f['frame_number'] == frame_num), None)
            grey_right_frame = next((f for f in grey_right_frames if f['frame_number'] == frame_num), None)
            
            if grey_left_frame and grey_right_frame:
                synchronized_entry = {
                    'frame_number': frame_num,
                    'timestamp': rgb_frame['timestamp'],
                    'weight': rgb_weights.get(frame_num, 0.0),  # Default to 0 if no weight
                    'rgb_frame': rgb_frame['filename'],
                    'grey_left_frame': grey_left_frame['filename'],
                    'grey_right_frame': grey_right_frame['filename']
                }
                synchronized_data.append(synchronized_entry)
        
        return synchronized_data
    
    def process_data_folder(self, data_folder):
        """Process a single data folder containing videos."""
        data_id = data_folder.name
        logger.info(f"\nProcessing data folder: {data_id}")
        
        # Find video files
        video_files = self.find_video_files(data_folder)
        
        # Check if all videos are found
        missing_videos = [k for k, v in video_files.items() if v is None]
        if missing_videos:
            logger.warning(f"Missing videos for {data_id}: {missing_videos}")
            return None
        
        # Create output folder for this data
        data_output_folder = self.output_dir / data_id
        data_output_folder.mkdir(exist_ok=True)
        
        # Extract frames from each video
        rgb_frames = self.extract_frames(video_files['rgb'], data_output_folder, 'rgb', data_id)
        grey_left_frames = self.extract_frames(video_files['grey_left'], data_output_folder, 'grey_left', data_id)
        grey_right_frames = self.extract_frames(video_files['grey_right'], data_output_folder, 'grey_right', data_id)
        
        # Synchronize frames and interpolate weights
        synchronized_data = self.synchronize_frames(rgb_frames, grey_left_frames, grey_right_frames)
        
        # Save metadata for this data folder
        data_metadata = {
            'data_id': data_id,
            'source_videos': {
                'rgb': str(video_files['rgb'].name),
                'grey_left': str(video_files['grey_left'].name),
                'grey_right': str(video_files['grey_right'].name)
            },
            'total_frames': len(synchronized_data),
            'frames': synchronized_data
        }
        
        # Save metadata JSON
        metadata_path = data_output_folder / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(data_metadata, f, indent=2)
        
        logger.info(f"Processed {len(synchronized_data)} synchronized frames for {data_id}")
        
        return data_metadata
    
    def process_all_data(self):
        """Process all data folders in the input directory."""
        # Look for data folder
        data_dir = self.input_dir / 'data' if (self.input_dir / 'data').exists() else self.input_dir
        
        # Find all numbered data folders (1-xxx, 2-xxx, etc.)
        data_folders = []
        for folder in data_dir.iterdir():
            if folder.is_dir() and re.match(r'^\d+-\d+', folder.name):
                data_folders.append(folder)
        
        # Sort folders by the numeric prefix
        data_folders.sort(key=lambda x: int(x.name.split('-')[0]))
        
        logger.info(f"Found {len(data_folders)} data folders to process")
        if len(data_folders) > 5:
            logger.info(f"Data folders: {[f.name for f in data_folders[:5]]}... and {len(data_folders) - 5} more")
        else:
            logger.info(f"Data folders: {[f.name for f in data_folders]}")
        
        # Process each data folder
        for i, data_folder in enumerate(data_folders, 1):
            logger.info(f"\n[{i}/{len(data_folders)}] Processing: {data_folder.name}")
            try:
                data_metadata = self.process_data_folder(data_folder)
                if data_metadata:
                    self.metadata['samples'].append(data_metadata)
            except Exception as e:
                logger.error(f"Failed to process {data_folder.name}: {str(e)}")
                continue
        
        # Save global metadata
        global_metadata_path = self.output_dir / 'dataset_metadata.json'
        with open(global_metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"\nProcessing complete! Output saved to: {self.output_dir}")
        logger.info(f"Total samples processed: {len(self.metadata['samples'])}")
        logger.info(f"Failed samples: {len(data_folders) - len(self.metadata['samples'])}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess liquid volume detection dataset')
    parser.add_argument('input_dir', help='Path to model folder containing data subfolders')
    parser.add_argument('output_dir', help='Path to output processed data')
    parser.add_argument('--frame-interval', type=int, default=5, 
                       help='Extract every nth frame (default: 5)')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_dir).resolve()
    
    # Log paths for debugging
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    
    # Check if input directory exists
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return
    
    # Create preprocessor and run
    preprocessor = LiquidVolumePreprocessor(
        input_path,
        output_path,
        args.frame_interval
    )
    
    preprocessor.process_all_data()


if __name__ == "__main__":
    main()