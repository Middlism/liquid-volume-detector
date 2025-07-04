# python preprocessing/preprocess_liquid_data.py . ./processed_data --frame-interval 5

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

# Setup logging to track processing progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiquidVolumePreprocessor:
    def __init__(self, input_dir, output_dir, frame_interval=5, debug_mode=True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval
        self.debug_mode = debug_mode
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset metadata structure
        self.metadata = {
            'processed_date': datetime.now().isoformat(),
            'frame_interval': frame_interval,
            'debug_mode': debug_mode,
            'samples': []
        }
    
    def find_video_files(self, data_folder):
        video_files = {
            'rgb': None,
            'grey_left': None,
            'grey_right': None
        }

        cam_a_path = data_folder / 'CAM_A_video.mp4'
        cam_b_path = data_folder / 'CAM_B_mono.mp4'
        cam_c_path = data_folder / 'CAM_C_mono.mp4'
        
        if cam_a_path.exists():
            video_files['rgb'] = cam_a_path
        if cam_b_path.exists():
            video_files['grey_left'] = cam_b_path
        if cam_c_path.exists():
            video_files['grey_right'] = cam_c_path
        
        logger.info(f"Found videos in {data_folder.name}:")
        for camera_type, path in video_files.items():
            if path:
                logger.info(f"  {camera_type}: {path.name}")
        
        return video_files
    
    def detect_scale_region(self, frame):
        return []
    
    def detect_glass_cup_region(self, frame):
        return []
    
    def create_annotated_frame(self, frame, scale_regions, glass_regions, frame_number, weight=None):
        return frame.copy()
    
    def read_weight_from_frame(self, frame, scale_regions):
        return None, None, {"error": "OCR not implemented"}
    
    def extract_frames(self, video_path, output_folder, camera_type, data_id):
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
        
        logger.info(f"Processing {camera_type} video: {video_path.name} ({total_frames} frames @ {fps:.2f} fps)")
        
        with tqdm(total=total_frames//self.frame_interval, desc=f"{camera_type} frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_interval == 0:
                    # Initialize placeholder values
                    weight = None
                    scale_regions = []
                    glass_regions = []
                    ocr_debug = None
                    
                    if camera_type == 'rgb':
                        # Call placeholder detection functions
                        scale_regions = self.detect_scale_region(frame)
                        glass_regions = self.detect_glass_cup_region(frame)
                        
                        # Try to read weight
                        weight, best_scale_region, ocr_debug = self.read_weight_from_frame(frame, scale_regions)
                        
                        # Create annotated frame if in debug mode (currently just returns copy)
                        if self.debug_mode:
                            annotated_frame = self.create_annotated_frame(
                                frame, scale_regions, glass_regions, frame_count, weight
                            )
                            
                            # Save annotated frame
                            debug_filename = f"{data_id}_{camera_type}_debug_{frame_count:06d}.jpg"
                            debug_path = output_folder / debug_filename
                            cv2.imwrite(str(debug_path), annotated_frame)
                    
                    # Save original frame
                    frame_filename = f"{data_id}_{camera_type}_frame_{frame_count:06d}.jpg"
                    frame_path = output_folder / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    
                    # Store frame metadata
                    frame_info = {
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'filename': frame_filename,
                        'camera_type': camera_type,
                        'weight': weight,
                        'scale_regions': scale_regions,
                        'glass_regions': glass_regions,
                        'ocr_debug': ocr_debug
                    }
                    
                    if self.debug_mode and camera_type == 'rgb':
                        frame_info['debug_filename'] = debug_filename
                    
                    frame_data.append(frame_info)
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        return frame_data
    
    def synchronize_frames(self, rgb_frames, grey_left_frames, grey_right_frames):
        synchronized_data = []
        
        detection_stats = {
            "total_frames": len(rgb_frames), 
            "successful_detections": 0,
            "detection_rate": 0.0
        }
        
        # Create synchronized data
        for rgb_frame in rgb_frames:
            frame_num = rgb_frame['frame_number']
            
            # Find corresponding frames
            grey_left_frame = next((f for f in grey_left_frames if f['frame_number'] == frame_num), None)
            grey_right_frame = next((f for f in grey_right_frames if f['frame_number'] == frame_num), None)
            
            if grey_left_frame and grey_right_frame:
                synchronized_entry = {
                    'frame_number': frame_num,
                    'timestamp': rgb_frame['timestamp'],
                    'weight': 0.0,  # Default weight since OCR not implemented
                    'rgb_frame': rgb_frame['filename'],
                    'grey_left_frame': grey_left_frame['filename'],
                    'grey_right_frame': grey_right_frame['filename'],
                    'scale_regions_detected': len(rgb_frame.get('scale_regions', [])),
                    'glass_regions_detected': len(rgb_frame.get('glass_regions', [])),
                    'weight_detected': False  # Always False since OCR not implemented
                }
                
                if self.debug_mode:
                    synchronized_entry['debug_frame'] = rgb_frame.get('debug_filename')
                    synchronized_entry['ocr_debug'] = rgb_frame.get('ocr_debug')
                
                synchronized_data.append(synchronized_entry)
        
        return synchronized_data, detection_stats
    
    def process_data_folder(self, data_folder):
        data_id = data_folder.name
        logger.info(f"\nProcessing data folder: {data_id}")
        
        # Find video files
        video_files = self.find_video_files(data_folder)
        
        # Check if all required videos are present
        missing_videos = [k for k, v in video_files.items() if v is None]
        if missing_videos:
            logger.warning(f"Missing videos for {data_id}: {missing_videos}")
            return None
        
        # Create output folder
        data_output_folder = self.output_dir / data_id
        data_output_folder.mkdir(exist_ok=True)
        
        # Extract frames
        rgb_frames = self.extract_frames(video_files['rgb'], data_output_folder, 'rgb', data_id)
        grey_left_frames = self.extract_frames(video_files['grey_left'], data_output_folder, 'grey_left', data_id)
        grey_right_frames = self.extract_frames(video_files['grey_right'], data_output_folder, 'grey_right', data_id)
        
        # Synchronize frames
        synchronized_data, detection_stats = self.synchronize_frames(rgb_frames, grey_left_frames, grey_right_frames)
        
        # Create metadata
        data_metadata = {
            'data_id': data_id,
            'source_videos': {
                'rgb': str(video_files['rgb'].name),
                'grey_left': str(video_files['grey_left'].name),
                'grey_right': str(video_files['grey_right'].name)
            },
            'total_frames': len(synchronized_data),
            'detection_statistics': detection_stats,
            'frames': synchronized_data
        }
        
        # Save metadata
        metadata_path = data_output_folder / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(data_metadata, f, indent=2)
        
        logger.info(f"Processed {len(synchronized_data)} synchronized frames for {data_id}")
        logger.info(f"Note: Weight detection not implemented - using default values")
        
        return data_metadata
    
    def process_all_data(self):
        # Look for data folders
        data_dir = self.input_dir / 'data' if (self.input_dir / 'data').exists() else self.input_dir
        
        # Find all numbered data folders
        data_folders = []
        for folder in data_dir.iterdir():
            if folder.is_dir() and re.match(r'^\d+-\d+', folder.name):
                data_folders.append(folder)
        
        data_folders.sort(key=lambda x: int(x.name.split('-')[0]))
        
        logger.info(f"Found {len(data_folders)} data folders to process")
        logger.info(f"Debug mode: {'ENABLED' if self.debug_mode else 'DISABLED'}")
        logger.info("Note: Scale detection and OCR are not implemented - placeholder functions in use")
        
        # Process each folder
        total_stats = {
            "total_samples": 0,
            "successful_samples": 0,
            "total_frames": 0,
            "total_detections": 0
        }
        
        for i, data_folder in enumerate(data_folders, 1):
            logger.info(f"\n[{i}/{len(data_folders)}] Processing: {data_folder.name}")
            try:
                data_metadata = self.process_data_folder(data_folder)
                if data_metadata:
                    self.metadata['samples'].append(data_metadata)
                    total_stats["successful_samples"] += 1
                    total_stats["total_frames"] += data_metadata['total_frames']
                
                total_stats["total_samples"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {data_folder.name}: {str(e)}")
                continue
        
        # Add global statistics
        self.metadata['global_statistics'] = total_stats
        
        # Save global metadata
        global_metadata_path = self.output_dir / 'dataset_metadata.json'
        with open(global_metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info(f"PROCESSING COMPLETE!")
        logger.info(f"{'='*50}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Total samples processed: {total_stats['successful_samples']}/{total_stats['total_samples']}")
        logger.info(f"Total frames extracted: {total_stats['total_frames']}")
        logger.info(f"Note: Detection features not implemented - frames extracted only")


def main():
    parser = argparse.ArgumentParser(description='Liquid volume detection preprocessor (frame extraction only)')
    parser.add_argument('input_dir', help='Path to model folder containing data subfolders')
    parser.add_argument('output_dir', help='Path to output processed data')
    parser.add_argument('--frame-interval', type=int, default=5, 
                       help='Extract every nth frame (default: 5)')
    parser.add_argument('--no-debug', action='store_true', 
                       help='Disable debug mode (currently has no effect)')
    
    args = parser.parse_args()
    
    # Convert to paths
    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_dir).resolve()
    
    # Log configuration
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Frame interval: {args.frame_interval}")
    logger.info(f"Debug mode: {'DISABLED' if args.no_debug else 'ENABLED'}")
    
    # Validate input
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return
    
    # Create and run preprocessor
    preprocessor = LiquidVolumePreprocessor(
        input_path,
        output_path,
        args.frame_interval,
        debug_mode=not args.no_debug
    )
    
    preprocessor.process_all_data()


if __name__ == "__main__":
    main()