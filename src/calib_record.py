import numpy as np
from hardware.cameras import Cameras
from hardware.kuka import Kuka
import cv2
import apriltag


'''
NOTE: Calibrate camera setup on manipulation station using AprilTag

tag_size = 5.6cm

Data Format:
-> cam2tag pose (apriltag lib detects in cm)
-> kuka_base2tag pose
'''

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

# --- camera visualization API ---
def plotPoint(image, center, color):
    center = (int(center[0]), int(center[1]))
    image = cv2.line(image,
                     (center[0] - LINE_LENGTH, center[1]),
                     (center[0] + LINE_LENGTH, center[1]),
                     color,
                     3)
    image = cv2.line(image,
                     (center[0], center[1] - LINE_LENGTH),
                     (center[0], center[1] + LINE_LENGTH),
                     color,
                     3)
    return image

def plotText(image, center, color, text):
    center = (int(center[0]) + 4, int(center[1]) - 4)
    return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 3)
    
if __name__ == '__main__':
    kuka = Kuka(scenario_file='../config/calib_med.yaml')
    pass