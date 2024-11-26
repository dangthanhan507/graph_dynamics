#NOTE: MOSTLY Yoinked from https://github.com/robo-alex/gs-dynamics
import argparse
import numpy as np
import torch
import open3d as o3d
import cv2
from PIL import Image
from hardware.cameras import Cameras
# from real_world.utils.pcd_utils import visualize_o3d, depth2fgpcd

from segment_anything import SamPredictor, sam_model_registry
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model as dino_build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from hardware.cameras import depth2pcd
#NOTE: purpose of this module is to get point clouds of desired object from all cameras

class Perception3DModule:
    def __init__(self, vis_path = "", workspace_bbox = None, device='cuda:0'):
        self.device = device
        self.vis_path = vis_path
        
        #NOTE: bbox follows min-max format [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
        self.workspace_bbox = workspace_bbox
        
        print("Loading models...")
        # Load Grounding DINO model for detection
        det_model = dino_build_model(
            SLConfig.fromfile(
                '../third-party/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py'
            )
        )
        chkpt = torch.load(
            '../weights/groundingdino_swinb_cogcoor.pth', map_location='cpu'
        )
        det_model.load_state_dict(clean_state_dict(chkpt['model']), strict=False)
        det_model.eval()
        det_model = det_model.to(self.device) # load on proper device
        
        # Load SAM model for segmentation
        sam = sam_model_registry['default'](checkpoint='../weights/sam_vit_h_4b8939.pth')
        sam_model = SamPredictor(sam)
        sam_model.model = sam_model.model.to(self.device)
        
        self.det_model = det_model
        self.sam_model = sam_model
        print("Successfully loaded models")
    def del_model(self):
        del self.det_model
        del self.sam_model
        torch.cuda.empty_cache()
        self.det_model = None
        self.sam_model = None
        
    def detect(self, image, captions, box_thresholds, verbose=False):
        
        #preprocessing to run into DINO detection
        image = Image.fromarray(image)
        captions = [caption.lower().strip() +  ('' if caption.endswith('.') else '.')  for caption in captions]
        n_captions = len(captions)
        
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor, _ = transform(image, None) # (3,H,W)
        image_tensor = image_tensor[None].repeat(n_captions, 1, 1, 1).to(self.device)
        
        # running dino detection + formatting output
        with torch.no_grad():
            outputs = self.det_model(image_tensor, captions=captions)
        logits = outputs['pred_logits'].sigmoid() # (n_caption, nq, 256)
        boxes  = outputs['pred_boxes'] # (n_caption, nq, 4)
        labels = torch.ones((*logits.shape[:2], 1)) * torch.arange(logits.shape[0])[:, None, None]  # (n_captions, nq, 1)
        labels = labels.to(device=self.device, dtype=logits.dtype)
                
        if isinstance(box_thresholds, list): 
            # do a per-caption thresholding
            filt_mask = logits.max(dim=2)[0] > torch.tensor(box_thresholds).to(device=self.device, dtype=logits.dtype)[:, None]
        else: 
            # if it is a float, simple comparison
            filt_mask = logits.max(dim=2)[0] > box_thresholds
            
        logits = logits[filt_mask] # num_filt, 256
        boxes  = boxes[filt_mask]
        labels = labels[filt_mask].reshape(-1).to(dtype=torch.int64)
        
        scores = logits.max(dim=1)[0]
        
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i,2) for i in box.tolist()]
            if verbose:
                print(f"Detected {captions[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
        return boxes, scores, labels
    
    def segment(self, image, boxes, scores, labels, text_prompts):
        #NOTE: boxes, scores, labels come from output of Perception3DModule.detect
        self.sam_model.set_image(image)
        
        # get masks
        masks, _, _ = self.sam_model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=self.sam_model.transform.apply_boxes_torch(boxes, image.shape[:2]),
            multimask_output=False
        )
        masks = masks[:, 0, :, :] #(n_detections, H, W)
        
        text_labels = []
        for category in range(len(text_prompts)):
            text_labels = text_labels + ([text_prompts[category].rstrip('.')] * (labels == category).sum().item())
        
        # remove masks where corresponding boxes have large IoU
        num_masks = masks.shape[0]
        to_remove = []
        for i in range(num_masks):
            for j in range(i+1, num_masks):
                iou = (masks[i] & masks[j]).sum().item() / (masks[i] | masks[j]).sum().item()
                if iou > 0.9:
                    to_remove.append(j if scores[i].item() > scores[j].item() else i)
        to_remove = np.unique(to_remove)
        to_keep = np.setdiff1d(np.arange(num_masks), to_remove)
        to_keep = torch.from_numpy(to_keep).to(device=self.device, dtype=torch.int64)
        
        masks = masks[to_keep]
        text_labels = [text_labels[i] for i in to_keep]
        
        aggr_mask = torch.zeros(masks[0].shape).to(device=self.device, dtype=torch.uint8)
        for obj_i in range(masks.shape[0]):
            aggr_mask[masks[obj_i]] = obj_i + 1
        
        return (masks, aggr_mask, text_labels), (boxes, scores, labels)
    
    def camera_improc_fn(self, image, depth, intrinsic, extrinsic, additional_obj_names = []):
        #NOTE: get relevant point clouds for object through image processing (aka deep learning)
        text_prompts = ['table'] + additional_obj_names
        
        H,W,_ = image.shape
        K = intrinsic
        H_cam2world = np.linalg.inv(extrinsic)
        R_cam2world = H_cam2world[:3,:3]
        t_cam2world = H_cam2world[:3,3]
        
        pts3d,ptsrgb = depth2pcd(depth, K, rgb=image[:,:,::-1])
        im = image.copy()
        
        mask = ((depth > 0) & (depth < 2.0))
        
        # detect and segment
        boxes, scores, labels = self.detect(im, text_prompts, box_thresholds=0.3) #NOTE: boxes are in format [x0,y0,w,h]
        boxes = boxes * torch.Tensor([[W, H, W, H]]).to(device=self.device, dtype=boxes.dtype)
        boxes[:,:2] -= boxes[:,2:] / 2
        boxes[:, 2:] += boxes[:,:2] #NOTE: now boxes are in format [x0,y0,x1,y1]
        
        (masks, _, text_labels), _ = self.segment(im, boxes, scores, labels, text_prompts)
        masks = masks.detach().cpu().numpy()
        
        mask_table = np.zeros(masks[0].shape, dtype=bool)
        not_mask_table = np.zeros(masks[0].shape, dtype=bool)
        mask_objs  = np.zeros(masks[0].shape, dtype=bool)
        for obj_i in range(masks.shape[0]):
            if text_labels[obj_i] == 'table':
                mask_table = mask_table | masks[obj_i]
            else:
                not_mask_table = not_mask_table | masks[obj_i]
                mask_objs = mask_objs | masks[obj_i]
        mask_table = mask_table & (~not_mask_table)
        mask_obj_and_background = (~mask_table)
        # take segmentation mask and ensure it is within obj and background only
        mask = mask & mask_obj_and_background
        
        # mask_objs = np.zeros(masks[0].shape, dtype=bool)
        # for obj_i in range(masks.shape[0]):
        #     if text_labels[obj_i] != 'table':
        #         mask_objs = mask_objs | masks[obj_i]
        # mask = mask & mask_objs
        
        
        mask = mask.flatten()
        
        pts3d = pts3d[mask,:]
        ptsrgb = ptsrgb[mask,:]
        if len(pts3d.shape) == 1:
            pts3d = pts3d.reshape(-1,3)
            ptsrgb = ptsrgb.reshape(-1,3)
        
        pts3d = (R_cam2world @ pts3d.T).T + t_cam2world
        
        if self.workspace_bbox is not None:
            bbox_mask = (pts3d[:,0] > self.workspace_bbox[0][0]) & (pts3d[:,0] < self.workspace_bbox[0][1])
            bbox_mask = bbox_mask & (pts3d[:,1] > self.workspace_bbox[1][0]) & (pts3d[:,1] < self.workspace_bbox[1][1])
            bbox_mask = bbox_mask & (pts3d[:,2] > self.workspace_bbox[2][0]) & (pts3d[:,2] < self.workspace_bbox[2][1])
            pts3d = pts3d[bbox_mask, :]
            ptsrgb = ptsrgb[bbox_mask, :]
        
        return pts3d, ptsrgb
    def get_pcd(self, cameras: Cameras, object_names=['object']):
        intrinsics = cameras.get_intrinsics()
        extrinsics = cameras.get_extrinsics()
        obs = cameras.get_obs(get_depth=True, get_color=True)
        
        colors = [ obs[f'color_{i}'][-1] for i in range(cameras.n_fixed_cameras) ]
        depths = [ obs[f'depth_{i}'][-1] for i in range(cameras.n_fixed_cameras) ]
        
        pts3d = []
        ptsrgb = []
        for i in range(cameras.n_fixed_cameras):
            pts3d_i, ptsrgb_i = self.camera_improc_fn(colors[i], depths[i], intrinsics[i], extrinsics[i], additional_obj_names=object_names)
            pts3d.append(pts3d_i)
            ptsrgb.append(ptsrgb_i)
        pts3d = np.concatenate(pts3d, axis=0)
        ptsrgb = np.concatenate(ptsrgb, axis=0)
        
        return pts3d, ptsrgb