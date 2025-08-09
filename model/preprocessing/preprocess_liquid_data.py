import cv2
import json
import logging
import argparse
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
from segments import Segments
import matplotlib.pyplot as plt
import subprocess
import tempfile

# python preprocessing/preprocess_liquid_data.py data processed_data --frame-interval 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self):
        self.segments = Segments()
        self.scale_region = [548, 658, 669, 720] # [x1,x2,y1,y2] for scale region dataset 1-20
        self.scale_region_2 = [575, 685, 650,701] # [x1,x2,y1,y2] for scale region dataset after 20 (21-45)
        
    def detect_weight(self, edges, first_part):
        # Load digit templates (0-9)
        templates = []
        template_path = Path('preprocessing/templates')
        
        for digit in range(10):
            template_file = template_path / f'template_{digit}.png'
            if template_file.exists():
                template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    # Resize template to 17x39 if needed
                    template = cv2.resize(template, (17, 39))
                    templates.append(template)
                else:
                    templates.append(None)
            else:
                templates.append(None)
        
        # Define digit regions (x1, x2, y1, y2)
        digit_regions_1 = [
            (45, 62, 8, 48),  # First digit
            (62, 79, 8, 48),  # Second digit  
            (79, 98, 8, 48)   # Third digit
        ]

        digit_regions_2 = [
            (38, 57, 9, 50),
            (57, 75, 9, 50),
            (75, 93, 9, 50)
        ]
        
        detected_digits = []
        digit_started = False
        
        # Process each digit position
        for x1, x2, y1, y2 in digit_regions_1 if first_part else digit_regions_2:
            # Extract digit region
            digit_region = edges[y1:y2, x1:x2]
            
            # Find best matching digit
            best_score = -1
            best_digit = None
            
            for digit, template in enumerate(templates):
                if template is None:
                    continue
                    
                # Perform template matching
                result = cv2.matchTemplate(digit_region, template, cv2.TM_CCOEFF_NORMED)
                score = result.max()
                
                if score > best_score:
                    best_score = score
                    best_digit = digit
            
            # Threshold for valid detection
            if best_score > 0.5:  # Adjust threshold as needed
                detected_digits.append(str(best_digit))
                digit_started = True
            else:
                # If we've started detecting digits and encounter None, stop
                if digit_started:
                    break
        
        # Convert digits to weight value
        if detected_digits:  # If at least one digit was detected
            weight = int(''.join(detected_digits))
            return weight
        
        return None

    def process_folder(self, input_dir, output_dir, frame_interval=3):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        data_folders = sorted([f for f in input_path.iterdir() 
                        if f.is_dir() and re.match(r'^(\d+)-(\d+)', f.name)],
                        key=lambda x: int(re.match(r'^(\d+)-', x.name).group(1)))

        logger.info(f"Found {len(data_folders)} data folders")
        
        for folder in data_folders:
            video_path = folder / 'CAM_A_video.mp4'
            if not video_path.exists():
                logger.warning(f"No video in {folder.name}")
                continue
            
            folder_output = output_path / folder.name
            folder_output.mkdir(exist_ok=True)
            
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS for time calculation
            
            frame_count = 0
            weights = []
            weight_timeline = []
            
            logger.info(f"Processing {folder.name}")
            
            with tqdm(total=total_frames//frame_interval) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        x1, x2, y1, y2 = self.scale_region if int(folder.name.split('-')[0]) <= 20 else self.scale_region_2
                        scale_frame = frame[y1:y2, x1:x2]
                        gray_frame = cv2.cvtColor(scale_frame, cv2.COLOR_BGR2GRAY)
                        enhanced = cv2.convertScaleAbs(gray_frame, alpha=2, beta=-50)
                        # Method 1: Canny Edge Detection
                        # edges = cv2.Canny(enhanced, 100, 200)
                        # Method 2: Adaptive Thresholding
                        # adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        
                        background = cv2.GaussianBlur(enhanced, (101, 101), 0)
                        flat = cv2.divide(enhanced, background, scale=255)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        clahe_enhanced = clahe.apply(flat)
                        pre_thresh = cv2.GaussianBlur(clahe_enhanced, (5, 5), 0)

                        threshold_value, edges = cv2.threshold(clahe_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        hist = cv2.calcHist([enhanced], [0], None, [256], [0, 256])
                        hist_norm = hist.ravel() / hist.sum()
                        cdf = hist_norm.cumsum()
                        plt.figure(figsize=(10, 6))
                        plt.bar(range(256), hist_norm, color='gray', alpha=0.6, label='Histogram')
                        ax2 = plt.gca().twinx()
                        ax2.plot(range(256), cdf, 'r-', linewidth=2, label='CDF')
                        ax2.set_ylabel('CDF', color='r')
                        ax2.tick_params(axis='y', labelcolor='r')
                        ax2.set_ylim([0, 1])
                        plt.axvline(x=threshold_value, color='b', linestyle='--', linewidth=2, label=f'Otsu threshold: {threshold_value}')
                        plt.xlabel('Pixel Value')
                        plt.ylabel('Normalized Frequency')
                        plt.title(f'Histogram & CDF - Frame {frame_count}')
                        plt.legend(loc='upper left')
                        plt.grid(True, alpha=0.3)
                        plt.xlim([0, 255])
                        hist_filename = f"histogram_frame_{frame_count:06d}.png"
                        plt.savefig(str(folder_output / hist_filename), dpi=100, bbox_inches='tight')
                        plt.close()

                        # Correct the call by passing the first_part parameter
                        first_part = int(folder.name.split('-')[0]) <= 20
                        weight = self.detect_weight(edges, first_part)

                        if weight is not None:
                            weights.append(weight)
                            try:
                                weight_timeline.append((frame_count, int(weight)))
                            except:
                                pass
                            filename = f"frame_{frame_count:06d}_weight_{weight}g.jpg"
                        else:
                            filename = f"frame_{frame_count:06d}.jpg"
                        
                        cv2.imwrite(str(folder_output / filename), edges)
                        pbar.update(1)
                    
                    frame_count += 1

            cap.release()

            with open(folder_output / 'metadata.json', 'w') as f:
                json.dump({
                    'folder': folder.name,
                    'frames_processed': frame_count // frame_interval,
                    'weights_detected': len(weights),
                    'unique_weights': sorted(list(set(weights))),
                    'detection_rate': len(weights) / (frame_count // frame_interval) if frame_count > 0 else 0
                }, f, indent=2)
            logger.info(f"  Detected: {len(weights)} weights, Unique: {sorted(set(weights))}")
            
            if weight_timeline:
                frames, weights_values = zip(*weight_timeline)
                times = [frame / fps for frame in frames]  # Convert frames to time in seconds
                
                plt.figure(figsize=(12, 6))
                plt.scatter(times, weights_values, alpha=0.6, s=20)
                
                z = np.polyfit(times, weights_values, 1)
                p = np.poly1d(z)
                plt.plot(times, p(times), "r-", alpha=0.8, label=f'Linear fit: y={z[0]:.4f}x+{z[1]:.2f}')
                
                plt.xlabel('Time (seconds)')
                plt.ylabel('Weight (g)')
                plt.title(f'Weight Detection - {folder.name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(str(folder_output / 'weight_graph.png'), dpi=100, bbox_inches='tight')
                plt.close()
                
        logger.info(f"\nProcessing complete. Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Input directory containing data folders')
    parser.add_argument('output_dir', help='Output directory for processed frames')
    parser.add_argument('--frame-interval', type=int, default=5, help='Process every nth frame')
    args = parser.parse_args()
    
    processor = VideoProcessor()
    processor.process_folder(args.input_dir, args.output_dir, args.frame_interval)


if __name__ == '__main__':
    main()