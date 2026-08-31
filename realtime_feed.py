#!/usr/bin/env python3
"""
Real-Time 4D-HMR Video Feed Translation & 3D Visualization
==========================================================
Fixed:
- Correct X-axis coordinate alignment
- Dynamic aspect ratio preservation matching camera sensor
- Mirror mode support for webcam
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import trimesh
import pyrender

from hmr2.configs import CACHE_DIR_4DHUMANS, get_config
from hmr2.models import HMR2, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import cam_crop_to_full, create_raymond_lights

LIGHT_CYAN = (0.35, 0.85, 0.95)
VIBRANT_CORAL = (0.95, 0.45, 0.25)
BG_DARK = (0.10, 0.12, 0.16)

def parse_args():
    parser = argparse.ArgumentParser(description="4D-Humans Real-Time Video Feed Translation")
    parser.add_argument("--source", type=str, default="0", help="Video source: camera index ('0') or video filepath")
    parser.add_argument("--mirror", action="store_true", default=None, help="Horizontally mirror camera feed")
    parser.add_argument("--no_mirror", action="store_true", default=False, help="Disable camera mirroring")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="HMR 2.0 checkpoint path")
    parser.add_argument("--detector", type=str, default="regnety", choices=["vitdet", "regnety"], help="Detector backend")
    parser.add_argument("--det_thresh", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--side_view", action="store_true", default=True, help="Render dual side-by-side 3D perspective")
    parser.add_argument("--rot_angle", type=float, default=45.0, help="Rotation angle for side view in degrees")
    parser.add_argument("--save_video", type=str, default=None, help="Output path to save translated video")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--headless", action="store_true", default=False, help="Run without opening GUI display window")
    return parser.parse_args()

def setup_detector(detector_type="regnety"):
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    if detector_type == "vitdet":
        from detectron2.config import LazyConfig
        import hmr2
        cfg_path = Path(hmr2.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        return DefaultPredictor_Lazy(detectron2_cfg)
    elif detector_type == "regnety":
        from detectron2 import model_zoo
        detectron2_cfg = model_zoo.get_config("new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        return DefaultPredictor_Lazy(detectron2_cfg)

def render_front_mesh(faces, all_verts, all_cam_t, render_res, focal_length, mesh_color=LIGHT_CYAN):
    if len(all_verts) == 0:
        return np.zeros((render_res[1], render_res[0], 4), dtype=np.float32)
    r = pyrender.OffscreenRenderer(viewport_width=render_res[0], viewport_height=render_res[1], point_size=1.0)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0.0], ambient_light=(0.35, 0.35, 0.35))
    for i, (verts, cam_t) in enumerate(zip(all_verts, all_cam_t)):
        vcol = np.array([(*mesh_color, 1.0)] * verts.shape[0])
        mesh = trimesh.Trimesh(verts.copy() + cam_t, faces.copy(), vertex_colors=vcol)
        rot_x = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot_x)
        scene.add(pyrender.Mesh.from_trimesh(mesh), f"m_{i}")
    cam = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=render_res[0] / 2.0, cy=render_res[1] / 2.0, zfar=1e12)
    scene.add_node(pyrender.Node(camera=cam, matrix=np.eye(4)))
    for node in create_raymond_lights():
        scene.add_node(node)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    return color.astype(np.float32) / 255.0

def render_side_perspective(faces, all_verts, all_cam_t, render_res, focal_length, rot_angle=45.0, mesh_color=VIBRANT_CORAL):
    if len(all_verts) == 0:
        return np.full((render_res[1], render_res[0], 3), (25, 30, 40), dtype=np.uint8)
    r = pyrender.OffscreenRenderer(viewport_width=render_res[0], viewport_height=render_res[1], point_size=1.0)
    scene = pyrender.Scene(bg_color=[*BG_DARK, 0.0], ambient_light=(0.35, 0.35, 0.35))
    for i, (verts, cam_t) in enumerate(zip(all_verts, all_cam_t)):
        vcol = np.array([(*mesh_color, 1.0)] * verts.shape[0])
        mesh = trimesh.Trimesh(verts.copy(), faces.copy(), vertex_colors=vcol)
        rot_y = trimesh.transformations.rotation_matrix(np.radians(rot_angle), [0, 1, 0])
        mesh.apply_transform(rot_y)
        mesh.vertices += cam_t
        rot_x = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot_x)
        scene.add(pyrender.Mesh.from_trimesh(mesh), f"m_{i}")
    cam = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=render_res[0] / 2.0, cy=render_res[1] / 2.0, zfar=1e12)
    scene.add_node(pyrender.Node(camera=cam, matrix=np.eye(4)))
    for node in create_raymond_lights():
        scene.add_node(node)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    return (color[:, :, :3] * 255).astype(np.uint8)

def main():
    args = parse_args()
    print("=========================================================")
    print("  4D-Humans (HMR 2.0) Real-Time Video Feed Translation  ")
    print("=========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[+] Compute Hardware: {gpu_name}")
    
    is_cam = args.source.isdigit()
    mirror_feed = True if (is_cam and not args.no_mirror) else (args.mirror is True)
    
    # Load HMR 2.0
    print(f"[+] Loading HMR 2.0 Model from {args.checkpoint}...")
    model, model_cfg = load_hmr2(args.checkpoint)
    model = model.to(device).eval()
    
    detector = setup_detector(args.detector)
    
    source_val = int(args.source) if is_cam else args.source
    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"[-] Error: Could not open {args.source}")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect = width / height
    print(f"[+] Feed Aspect Ratio: {width}x{height} ({aspect:.2f}:1)")
    
    writer = None
    if args.save_video:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_video)) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = width * (2 if args.side_view else 1)
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (out_w, height))
        
    frame_idx = 0
    start_time = time.time()
    fps_display = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if mirror_feed:
                frame = cv2.flip(frame, 1)
                
            frame_start = time.time()
            frame_idx += 1
            h, w = frame.shape[:2]
            
            det_out = detector(frame)
            valid_idx = (det_out["instances"].pred_classes == 0) & (det_out["instances"].scores > args.det_thresh)
            boxes = det_out["instances"].pred_boxes.tensor[valid_idx].cpu().numpy()
            
            all_verts = []
            all_cam_t = []
            if len(boxes) > 0:
                dataset = ViTDetDataset(model_cfg, frame, boxes)
                loader = torch.utils.data.DataLoader(dataset, batch_size=len(boxes), shuffle=False)
                batch = recursive_to(next(iter(loader)), device)
                with torch.no_grad():
                    out = model(batch)
                pred_cam = out["pred_cam"].float()
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
                for n in range(len(boxes)):
                    all_verts.append(out["pred_vertices"][n].detach().cpu().numpy())
                    all_cam_t.append(pred_cam_t_full[n])
            else:
                scaled_focal_length = float(model_cfg.EXTRA.FOCAL_LENGTH)
                
            front_rgba = render_front_mesh(model.smpl.faces, all_verts, all_cam_t, [w, h], scaled_focal_length, LIGHT_CYAN)
            in_f = frame.astype(np.float32)[:, :, ::-1] / 255.0
            in_f = np.concatenate([in_f, np.ones_like(in_f[:, :, :1])], axis=2)
            comp = in_f[:, :, :3] * (1.0 - front_rgba[:, :, 3:]) + front_rgba[:, :, :3] * front_rgba[:, :, 3:]
            front_bgr = np.ascontiguousarray((comp * 255).astype(np.uint8)[:, :, ::-1])
            
            if args.side_view:
                side_bgr = render_side_perspective(model.smpl.faces, all_verts, all_cam_t, [w, h], scaled_focal_length, args.rot_angle, VIBRANT_CORAL)
                side_bgr = np.ascontiguousarray(side_bgr[:, :, ::-1])
                cv2.rectangle(front_bgr, (10, 8), (280, 36), (20, 20, 24), -1)
                cv2.putText(front_bgr, "4D-HMR: Front 3D Mesh", (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 180), 1)
                cv2.rectangle(side_bgr, (10, 8), (280, 36), (20, 20, 24), -1)
                cv2.putText(side_bgr, f"4D-HMR: 3D Side View ({int(args.rot_angle)} deg)", (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 255), 1)
                display_frame = np.concatenate([front_bgr, side_bgr], axis=1)
            else:
                display_frame = front_bgr
                
            frame_dur = time.time() - frame_start
            inst_fps = 1.0 / max(frame_dur, 1e-4)
            fps_display = 0.85 * fps_display + 0.15 * inst_fps if fps_display > 0 else inst_fps
            
            dh, dw = display_frame.shape[:2]
            cv2.rectangle(display_frame, (0, dh - 28), (dw, dh), (15, 15, 18), -1)
            hud = f"FPS: {fps_display:.1f} | Persons: {len(all_verts)} | GPU: {gpu_name[:22]}"
            cv2.putText(display_frame, hud, (15, dh - 9), cv2.FONT_HERSHEY_DUPLEX, 0.45, (220, 230, 245), 1, cv2.LINE_AA)
            
            if writer is not None:
                writer.write(display_frame)
            if not args.headless:
                cv2.imshow("4D-Humans Feed", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
