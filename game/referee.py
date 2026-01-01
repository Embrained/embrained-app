
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MotionReferee:
    """
    Computer Vision Referee for 'Red Light, Green Light'.
    Detects movement during the Red Light phase to label data as 'fail'.
    """
    def __init__(self, history=500, varThreshold=16):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=history, 
            varThreshold=varThreshold, 
            detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def detect_movement(self, frame: np.ndarray) -> float:
        """
        Returns a motion score (0.0 to 1.0).
        """
        # Apply Background Subtraction
        fgmask = self.fgbg.apply(frame)
        
        # Remove Noise (Morphological Opening)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        
        # Calculate Motion Energy
        # normalized count of non-zero pixels (or sum of intensity)
        # Using simple mean intensity of the mask (0-255) -> 0-1
        motion_score = np.sum(fgmask) / 255.0
        
        # Normalize by frame size to get percentage of screen moving
        height, width = fgmask.shape
        total_pixels = height * width
        
        normalized_score = motion_score / total_pixels
        
        return normalized_score

    def reset(self):
        # Reset bg model if needed? MOG2 adapts over time.
        pass
