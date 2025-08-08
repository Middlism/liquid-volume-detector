import numpy as np 

class Segments:
    def __init__(self, threshold=100):
        # 7-segment layout indices:
        #   _0_
        # 1|   |2
        #   _3_
        # 4|   |5
        #   _6_
        # Mapping of segment bit patterns to digit values
        self.seg_digits = {
            '1111011': 0,
            '0010010': 1,
            '1011101': 2,
            '1011011': 3,
            '0111010': 4,
            '1101011': 5,
            '1101111': 6,
            '1010010': 7,
            '1111111': 8,
            '1111011': 9
        }
        # Coordinates of the 21 segments: (x1, y1, x2, y2)
        self.seg_pos = [
            # digit 0 segments
            (593, 678, 601, 684), (588, 684, 592, 692), (601, 684, 607, 694),
            (592, 696, 599, 701), (587, 703, 591, 711), (599, 703, 606, 712),
            (591, 713, 598, 719),
            # digit 1 segments
            (613, 678, 619, 684), (608, 684, 613, 694), (618, 684, 625, 696),
            (611, 695, 617, 702), (607, 702, 612, 713), (617, 703, 624, 713),
            (611, 713, 617, 719),
            # digit 2 segments
            (630, 678, 637, 684), (625, 684, 632, 695), (636, 685, 644, 695),
            (630, 695, 637, 703), (625, 703, 630, 713), (635, 703, 641, 714),
            (628, 714, 636, 718)
        ]
        self.threshold = threshold

    @staticmethod
    def _compute_gray(avg_color):
        b, g, r = avg_color
        return 0.114 * b + 0.587 * g + 0.299 * r

    def detect_weight(self, frame):
        digits = []
        for digit_idx in range(3):  
            segment_lums = []
            print(f"\n=== Digit #{digit_idx} ===")
            for seg_idx in range(7):
                global_idx = digit_idx * 7 + seg_idx
                x1, y1, x2, y2 = self.seg_pos[global_idx]
                segment = frame[y1:y2, x1:x2]
                avg_color = np.mean(segment.reshape(-1, 3), axis=0)
                lum = self._compute_gray(avg_color)
                segment_lums.append(lum)
                print(
                    f"Digit {digit_idx}, Segment {seg_idx}: "
                    f"RGB = {avg_color.round(2).tolist()}, "
                    f"Gray = {lum:.2f}"
                )
            avg_thresh = np.mean(segment_lums)
            print(f"-> Digit {digit_idx} average gray threshold = {avg_thresh:.2f}")
            segment_states = [
                1 if lum < 100 else 0
                for lum in segment_lums
            ]
            pattern = ''.join(str(b) for b in segment_states)
            print(f"-> Digit {digit_idx} bit pattern = {pattern}")

            if pattern == '0000000':
                continue
            if pattern in self.seg_digits:
                digits.append(self.seg_digits[pattern])
            else:
                print(f"Warning: unrecognized pattern for digit {digit_idx}")
                return None

        if digits:
            number = int(''.join(map(str, digits)))
            print(f"\nDetected number: {number}")
            return number
        return None

    def detect_weight_1(self, edges):

            ON_THRESHOLD = 100  # Segments with avg gray < this are considered "on"
            MIN_CONFIDENCE = 0.7  # Minimum confidence to attempt pattern matching
            
            digits = []
            
            for digit_idx in range(3):
                segments_avg = []
                segments_pattern = []
                
                # Calculate average gray value for each segment
                for seg_idx in range(7):
                    x1, x2, y1, y2 = self.segments_region[digit_idx][seg_idx]
                    segment_region = edges[y1:y2, x1:x2]
                    
                    # Calculate average gray value
                    avg_gray = np.mean(segment_region)
                    segments_avg.append(avg_gray)
                    
                    # Determine if segment is on (1) or off (0)
                    is_on = '1' if avg_gray < ON_THRESHOLD else '0'
                    segments_pattern.append(is_on)
                
                # Create pattern string
                pattern = ''.join(segments_pattern)
                
                # Check if pattern matches exactly
                if pattern in self.seg_digits:
                    digits.append(self.seg_digits[pattern])
                else:
                    # Calculate confidence: how clearly segments are on/off
                    confidences = []
                    for avg in segments_avg:
                        # Distance from threshold normalized to 0-1
                        confidence = abs(avg - ON_THRESHOLD) / ON_THRESHOLD
                        confidences.append(min(1.0, confidence))
                    
                    avg_confidence = np.mean(confidences)
                    
                    # Only attempt approximation if segments are confidently detected
                    if avg_confidence >= MIN_CONFIDENCE:
                        # Find closest matching pattern
                        best_match = None
                        min_difference = 8  # Start with impossible value
                        
                        for valid_pattern, digit_value in self.seg_digits.items():
                            # Count different segments
                            difference = sum(p != v for p, v in zip(pattern, valid_pattern))
                            
                            # Only consider if it's closer and reasonably similar (max 2 differences)
                            if difference < min_difference and difference <= 2:
                                min_difference = difference
                                best_match = digit_value
                        
                        if best_match is not None:
                            digits.append(best_match)
                        else:
                            # Can't reliably detect this digit
                            break
                    else:
                        # Segments not confidently detected, skip
                        break
            
            # Convert to integer
            if not digits:
                return None
            
            # Build result (e.g., [1, 2, 3] -> 123)
            result = 0
            for digit in digits:
                result = result * 10 + digit
            
            # Validate result is reasonable (0-999 for 3-digit display)
            return result if 0 <= result <= 999 else None