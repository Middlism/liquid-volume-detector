import cv2
import numpy as np

# Globals
drawing = False
ix, iy = -1, -1
rectangles = []
rectangle_colors = []  # Store average colors for each rectangle

def calculate_average_color(img, x1, y1, x2, y2):
    """Calculate average color value of pixels in rectangle"""
    roi = img[y1:y2, x1:x2]
    
    if len(img.shape) == 3:  # Color image
        avg_color = np.mean(roi.reshape(-1, 3), axis=0)
        return avg_color.astype(int)
    else:  # Grayscale
        avg_gray = np.mean(roi)
        return int(avg_gray)

def update_display():
    """Update the displayed image with rectangles"""
    global display_img, img_copy, rectangles, rectangle_colors
    
    display_img = img_copy.copy()
    
    # Draw rectangles
    for i, (x1, y1, x2, y2) in enumerate(rectangles):
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw rectangle number - adjust position for small images
        label_x = x1 + 2
        label_y = y1 + 10 if y1 < 10 else y1 - 5
        cv2.putText(display_img, str(i+1), (label_x, label_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Draw help text - adjust font size and position for small images
    font_scale = 0.35 if img_copy.shape[0] < 100 else 0.5
    text_y = max(15, display_img.shape[0] - 5)
    cv2.putText(display_img, "a:Analysis | s:Save | u:Undo | c:Clear | ESC:Exit", 
               (5, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, display_img, rectangles, rectangle_colors
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_temp = display_img.copy()
            cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select ROI", img_temp)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex, ey = x, y
        
        # Only add rectangle if it has some area
        if ix != ex and iy != ey:
            # Normalize coordinates
            x1, y1 = min(ix, ex), min(iy, ey)
            x2, y2 = max(ix, ex), max(iy, ey)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_copy.shape[1]-1))
            y1 = max(0, min(y1, img_copy.shape[0]-1))
            x2 = max(0, min(x2, img_copy.shape[1]-1))
            y2 = max(0, min(y2, img_copy.shape[0]-1))
            
            rectangles.append((x1, y1, x2, y2))
            
            # Calculate average color
            avg_color = calculate_average_color(img_copy, x1, y1, x2, y2)
            rectangle_colors.append(avg_color)
            
            print(f"\nRectangle #{len(rectangles)}: Top-left ({x1}, {y1}), Bottom-right ({x2}, {y2})")
            print(f"   Width: {x2-x1}, Height: {y2-y1}")
            
            if len(img_copy.shape) == 3:
                print(f"   Average Color (BGR): {avg_color}")
                print(f"   Average Color (RGB): [{avg_color[2]}, {avg_color[1]}, {avg_color[0]}]")
                print(f"   Overall Intensity: {np.mean(avg_color):.1f}")
            else:
                print(f"   Average Gray Value: {avg_color}")
            
            update_display()

def show_color_analysis():
    """Display detailed color analysis for all rectangles"""
    if not rectangles:
        print("\n[No rectangles to analyze]")
        return
    
    print("\n" + "="*60)
    print("COLOR ANALYSIS REPORT")
    print("="*60)
    
    for i, ((x1, y1, x2, y2), avg_color) in enumerate(zip(rectangles, rectangle_colors), 1):
        print(f"\nRectangle {i}:")
        print(f"  Position: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Size: {x2-x1} × {y2-y1} pixels")
        print(f"  Total pixels: {(x2-x1) * (y2-y1)}")
        
        if len(img_copy.shape) == 3:
            print(f"  Average BGR: {avg_color}")
            print(f"  Average RGB: [{avg_color[2]}, {avg_color[1]}, {avg_color[0]}]")
            print(f"  Blue channel:  {avg_color[0]}")
            print(f"  Green channel: {avg_color[1]}")
            print(f"  Red channel:   {avg_color[2]}")
            print(f"  Overall intensity: {np.mean(avg_color):.1f}")
            
            # Calculate min/max values in the region
            roi = img_copy[y1:y2, x1:x2]
            print(f"  Min BGR values: {np.min(roi.reshape(-1, 3), axis=0)}")
            print(f"  Max BGR values: {np.max(roi.reshape(-1, 3), axis=0)}")
            print(f"  Std deviation: {np.std(roi.reshape(-1, 3), axis=0).astype(int)}")
        else:
            print(f"  Average gray: {avg_color}")

if __name__ == "__main__":
    # Load image
    img = cv2.imread("frame_000645.jpg")
    if img is None:
        print("Error: Could not load 'frame_000645.jpg'.")
        exit(1)
    
    img_copy = img.copy()
    display_img = img_copy.copy()
    
    # Set window size based on image dimensions
    # For very small images, use the actual image size or a minimum size
    img_height, img_width = img.shape[:2]
    window_width = max(img_width, 400)  # Minimum width of 400 pixels for UI
    window_height = max(img_height, 200)  # Minimum height of 200 pixels for UI
    
    print(f"Image dimensions: {img_width}x{img_height}")
    print(f"Window dimensions: {window_width}x{window_height}")
    
    # Create resizable window to allow the user to adjust if needed
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI", window_width, window_height)
    cv2.setMouseCallback("Select ROI", mouse_callback)
    
    print("\nSimplified ROI Selector Tool")
    print("===========================")
    print("Drawing:")
    print(" • LEFT CLICK and DRAG to draw a rectangle")
    print("\nControls:")
    print(" • Press 'a' to show color analysis report")
    print(" • Press 'u' to undo last rectangle")
    print(" • Press 'c' to clear all selections")
    print(" • Press 's' to save selections")
    print(" • Press ESC to exit")
    print("\nNote: Average color is calculated automatically for each selection\n")
    
    update_display()
    
    while True:
        cv2.imshow("Select ROI", display_img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):  # Show color analysis
            show_color_analysis()
        
        elif key == ord('c'):  # Clear all
            rectangles.clear()
            rectangle_colors.clear()
            update_display()
            print("\n[Cleared all selections]")
        
        elif key == ord('u'):  # Undo last
            if rectangles:
                rectangles.pop()
                rectangle_colors.pop()
                print(f"\n[Undone] Rectangle #{len(rectangles)+1}")
                update_display()
        
        elif key == ord('s'):  # Save
            if rectangles:
                # Save with color information
                with open("roi_selections.txt", "w") as f:
                    f.write("ROI Selections with Color Analysis\n")
                    f.write("="*50 + "\n\n")
                    for i, ((x1, y1, x2, y2), avg_color) in enumerate(zip(rectangles, rectangle_colors), 1):
                        f.write(f"Rectangle {i}:\n")
                        f.write(f"  Coordinates: {x1}, {y1}, {x2}, {y2}\n")
                        f.write(f"  Size: {x2-x1} × {y2-y1}\n")
                        if len(img_copy.shape) == 3:
                            f.write(f"  Average BGR: {avg_color}\n")
                            f.write(f"  Average RGB: [{avg_color[2]}, {avg_color[1]}, {avg_color[0]}]\n")
                        else:
                            f.write(f"  Average Gray: {avg_color}\n")
                        f.write("\n")
                
                print(f"\n[Saved {len(rectangles)} rectangles with color data to 'roi_selections.txt']")
                
                # Save annotated image with color swatches
                img_with_rois = img_copy.copy()
                for i, ((a, b, c, d), color) in enumerate(zip(rectangles, rectangle_colors), 1):
                    cv2.rectangle(img_with_rois, (a, b), (c, d), (0, 255, 0), 2)
                    cv2.putText(img_with_rois, str(i), (a+5, b+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw color swatch
                    swatch_y = b + 30
                    if swatch_y + 20 < d:  # Only draw if there's space
                        if len(img_copy.shape) == 3:
                            cv2.rectangle(img_with_rois, (a+5, swatch_y), 
                                        (a+25, swatch_y+20), color.tolist(), -1)
                            cv2.rectangle(img_with_rois, (a+5, swatch_y), 
                                        (a+25, swatch_y+20), (255, 255, 255), 1)
                
                cv2.imwrite("roi_selections.jpg", img_with_rois)
                print("[Saved annotated image with color swatches to 'roi_selections.jpg']")
        
        elif key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()
    
    if rectangles:
        print(f"\nFinal selections ({len(rectangles)} rectangles):")
        for i, ((x1, y1, x2, y2), avg_color) in enumerate(zip(rectangles, rectangle_colors), 1):
            print(f"  Rectangle {i}: ({x1}, {y1}) to ({x2}, {y2}), Avg Color: {avg_color}")