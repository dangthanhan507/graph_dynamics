import numpy as np
from hardware.cameras import Cameras
import cv2
import apriltag

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

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

'''
NOTE: Calibrate camera setup on manipulation station using AprilTag

tag_size = 5.6cm

Data Format:
-> cam2tag pose
-> kuka_base2tag pose
'''

if __name__ == '__main__':    
    
    detector = apriltag.Detector()
    exposure_time = 10
    cam = Cameras(
        WH=[640, 480],
        capture_fps=15,
        obs_fps=30,
        n_obs_steps=2,
        enable_color=True,
        enable_depth=True,
        process_depth=True)
    cam.start(exposure_time=10)
    obs = cam.get_obs(get_color=True, get_depth=False)
    print(obs.keys())
    print(cam.get_intrinsics().shape)
    
    while 1:
        obs = cam.get_obs(get_color=True, get_depth=False)
        color0 = obs['color_0'][-1]
        color1 = obs['color_1'][-1]
        color2 = obs['color_2'][-1]
        color3 = obs['color_3'][-1]
        
        gray0 = cv2.cvtColor(color0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(color3, cv2.COLOR_BGR2GRAY)
        
        detections = detector.detect(gray0)
        if detections:
            intrinsic0 = cam.get_intrinsics()[0]
            fx = intrinsic0[0, 0]
            fy = intrinsic0[1, 1]
            cx = intrinsic0[0, 2]
            cy = intrinsic0[1, 2]
            camera_params = (fx, fy, cx, cy)
            for detect in detections:
                pose, _, error = detector.detection_pose(detect, camera_params=camera_params, tag_size=5.6)
                translation = pose[:3, 3]
                print(translation, error)
                
                color0 = plotPoint(color0, detect.center, CENTER_COLOR)
                color0 = plotText(color0, detect.center, CENTER_COLOR, detect.tag_id)
                for corner in detect.corners:
                    color0 = plotPoint(color0, corner, CORNER_COLOR)
        detections = detector.detect(gray1)
        if detections:
            for detect in detections:
                color1 = plotPoint(color1, detect.center, CENTER_COLOR)
                color1 = plotText(color1, detect.center, CENTER_COLOR, detect.tag_id)
                for corner in detect.corners:
                    color1 = plotPoint(color1, corner, CORNER_COLOR)
        detections = detector.detect(gray2)
        if detections:
            for detect in detections:
                color2 = plotPoint(color2, detect.center, CENTER_COLOR)
                color2 = plotText(color2, detect.center, CENTER_COLOR, detect.tag_id)
                for corner in detect.corners:
                    color2 = plotPoint(color2, corner, CORNER_COLOR)
        detections = detector.detect(gray3)
        if detections:
            for detect in detections:
                color3 = plotPoint(color3, detect.center, CENTER_COLOR)
                color3 = plotText(color3, detect.center, CENTER_COLOR, detect.tag_id)
                for corner in detect.corners:
                    color3 = plotPoint(color3, corner, CORNER_COLOR)
        top = cv2.hconcat([color0, color1])
        bottom = cv2.hconcat([color2, color3])
        fullrgb = cv2.vconcat([top, bottom])
        cv2.imshow('rgb', fullrgb)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cam.stop()
    pass