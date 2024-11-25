import numpy as np
from hardware.cameras import Cameras
from hardware.kuka import Kuka
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

#given 4x4 pose, visualize frame on camera (RGB)
def plotPose(image, pose, length=10, thickness=10, K = np.eye(3), dist=np.zeros(5)):
    rotm = pose[:3,:3]
    tvec = pose[:3,3]
    
    rvec = cv2.Rodrigues(rotm)[0]
    
    axis = np.float32([[length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    imgpts = imgpts.astype(int)
    center, _ = cv2.projectPoints(np.float32([[0,0,0]]), rvec, tvec, K, dist)
    center = tuple(center[0].ravel())
    center = (int(center[0]), int(center[1]))
    
    image = cv2.line(image, center, tuple(imgpts[0].ravel()), (255,0,0), thickness=thickness)
    image = cv2.line(image, center, tuple(imgpts[1].ravel()), (0,255,0), thickness=thickness)
    image = cv2.line(image, center, tuple(imgpts[2].ravel()), (0,0,255), thickness=thickness)
    return image

def visualize_detections(image, detections, K=np.eye(3), dist=np.zeros(5)):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    camera_params = (fx, fy, cx, cy)
    for detect in detections:
        pose, _, error = detector.detection_pose(detect, camera_params=camera_params, tag_size=5.6)
        
        image = plotPose(image, pose, K=K, dist=dist)
        image = plotPoint(image, detect.center, CENTER_COLOR)
        image = plotText(image, detect.center, CENTER_COLOR, detect.tag_id)
        for corner in detect.corners:
            image = plotPoint(image, corner, CORNER_COLOR)
    
    return image

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
    
    intrinsics = cam.get_intrinsics()
    cam0_K = intrinsics[0]
    cam1_K = intrinsics[1]
    cam2_K = intrinsics[2]
    cam3_K = intrinsics[3]
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
        color0 = visualize_detections(color0, detections, K=cam0_K)
        detections = detector.detect(gray1)
        color1 = visualize_detections(color1, detections, K=cam1_K)
        detections = detector.detect(gray2)
        color2 = visualize_detections(color2, detections, K=cam2_K)
        detections = detector.detect(gray3)
        color3 = visualize_detections(color3, detections, K=cam3_K)
        top = cv2.hconcat([color0, color1])
        bottom = cv2.hconcat([color2, color3])
        fullrgb = cv2.vconcat([top, bottom])
        
        cv2.imshow('rgb', fullrgb)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cam.stop()
    pass