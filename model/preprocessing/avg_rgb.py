import cv2
import numpy as np

def calculate_average_color(img, x1, y1, x2, y2):
    """Calculate average RGB color of pixels in rectangle"""
    # Extract the region of interest
    roi = img[y1:y2, x1:x2]
    
    # Calculate average color
    avg_color = np.mean(roi.reshape(-1, 3), axis=0)
    return avg_color.astype(int)

def main():
    # Define the 21 segment positions
    seg_pos = [
        # First digit (segments 0-6)
        (593, 678, 601, 684),  # 0: top
        (588, 684, 592, 692),  # 1: top-left
        (601, 684, 607, 694),  # 2: top-right
        (592, 696, 599, 701),  # 3: middle
        (587, 703, 591, 711),  # 4: bottom-left
        (599, 703, 606, 712),  # 5: bottom-right
        (591, 713, 598, 719),  # 6: bottom
        
        # Second digit (segments 7-13)
        (613, 678, 619, 684),  # 7: top
        (608, 684, 613, 694),  # 8: top-left
        (618, 684, 625, 696),  # 9: top-right
        (611, 695, 617, 702),  # 10: middle
        (607, 702, 612, 713),  # 11: bottom-left
        (617, 703, 624, 713),  # 12: bottom-right
        (611, 713, 617, 719),  # 13: bottom
        
        # Third digit (segments 14-20)
        (630, 678, 637, 684),  # 14: top
        (625, 684, 632, 695),  # 15: top-left
        (636, 685, 644, 695),  # 16: top-right
        (630, 695, 637, 703),  # 17: middle
        (625, 703, 630, 713),  # 18: bottom-left
        (635, 703, 641, 714),  # 19: bottom-right
        (628, 714, 636, 718),  # 20: bottom
    ]
    
    # Load the image
    img = cv2.imread("sample_frame.jpg")
    if img is None:
        print("Error: Could not load 'sample_frame.jpg'")
        return None
    
    # Calculate average colors for each segment
    avg_colors = []
    
    print("Average RGB Colors for Each Segment:")
    print("=" * 50)
    
    for i, (x1, y1, x2, y2) in enumerate(seg_pos):
        # Calculate average color (OpenCV uses BGR, so convert to RGB)
        avg_bgr = calculate_average_color(img, x1, y1, x2, y2)
        avg_rgb = [avg_bgr[2], avg_bgr[1], avg_bgr[0]]  # Convert BGR to RGB
        
        avg_colors.append(avg_rgb)
        
        # Determine which digit and segment
        if i < 7:
            digit = 1
            segment = i
        elif i < 14:
            digit = 2
            segment = i - 7
        else:
            digit = 3
            segment = i - 14
        
        segment_names = ["top", "top-left", "top-right", "middle", 
                        "bottom-left", "bottom-right", "bottom"]
        
        print(f"Segment {i:2d} (Digit {digit}, {segment_names[segment]:>11}): "
              f"RGB({avg_rgb[0]:3d}, {avg_rgb[1]:3d}, {avg_rgb[2]:3d})")
    
    # Optional: Create visualization
    create_visualization = input("\nCreate visualization image? (y/n): ").lower() == 'y'
    
    if create_visualization:
        # Create a copy for visualization
        vis_img = img.copy()
        
        for i, (x1, y1, x2, y2) in enumerate(seg_pos):
            # Draw rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Add segment number
            cv2.putText(vis_img, str(i), (x1-10, y1-2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            
            # Draw color swatch below each segment
            color_bgr = [avg_colors[i][2], avg_colors[i][1], avg_colors[i][0]]
            cv2.rectangle(vis_img, (x1, y2+2), (x2, y2+8), color_bgr, -1)
        
        cv2.imwrite("segment_analysis.jpg", vis_img)
        print("\nVisualization saved as 'segment_analysis.jpg'")
    
    return avg_colors

def get_brightness_analysis(avg_colors):
    """Analyze brightness to determine which segments are 'on'"""
    print("\nBrightness Analysis:")
    print("=" * 50)
    
    # Calculate brightness for each segment
    brightnesses = []
    for i, rgb in enumerate(avg_colors):
        brightness = np.mean(rgb)  # Simple average of R, G, B
        brightnesses.append(brightness)
        
        # Determine digit and segment
        if i < 7:
            digit = 1
            segment = i
        elif i < 14:
            digit = 2
            segment = i - 7
        else:
            digit = 3
            segment = i - 14
        
        print(f"Segment {i:2d} (Digit {digit}): Brightness = {brightness:.1f}")
    
    # Find threshold (could be more sophisticated)
    threshold = (max(brightnesses) + min(brightnesses)) / 2
    print(f"\nThreshold: {threshold:.1f}")
    print("\nSegments ON (brightness > threshold):")
    
    for i, brightness in enumerate(brightnesses):
        if brightness > threshold:
            if i < 7:
                digit = 1
            elif i < 14:
                digit = 2
            else:
                digit = 3
            print(f"  Segment {i} (Digit {digit})")

if __name__ == "__main__":
    # Run the analysis
    avg_colors = main()
    
    if avg_colors:
        # Additional brightness analysis
        print("\n" + "="*50)
        get_brightness_analysis(avg_colors)