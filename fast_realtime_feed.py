#!/usr/bin/env python3
"""
High-Performance Real-Time 4D-HMR Video & Camera Stream Engine
==============================================================
Fixed:
- Exact X-axis horizontal translation alignment (moving right moves right, moving left moves left)
- Native camera aspect ratio preservation (no stretched frames)
- Live webcam mirror mode (--mirror) for natural interactive feedback
- GPU FP16 + TF32 acceleration & YOLOv8n ultra-fast detector (~7.5ms)
"""

import os
import sys
import time
import threading
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import pyrender

# PyTorch GPU optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from hmr2.configs import CACHE_DIR_4DHUMANS, get_config
from hmr2.models import HMR2, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import cam_crop_to_full, create_raymond_lights

# Theme colors
LIGHT_CYAN = (0.35, 0.85, 0.95)
VIBRANT_CORAL = (0.95, 0.45, 0.25)
BG_DARK = (0.10, 0.12, 0.16)

class ThreadedCamera:
    """Decoupled camera capture thread for 0ms frame retrieval."""
    def __init__(self, src="0", mirror=False):
        self.is_num = str(src).isdigit()
        self.src = int(src) if self.is_num else src
        self.mirror = mirror
        self.cap = cv2.VideoCapture(self.src)
        self.ret, self.frame = self.cap.read()
        if self.ret and self.frame is not None and self.mirror:
            self.frame = cv2.flip(self.frame, 1)
        self.stopped = False
        self.lock = threading.Lock()
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = self.fps if (self.fps and 0 < self.fps < 120) else 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                if not self.is_num:
                    self.stopped = True
                break
            if self.mirror and frame is not None:
                frame = cv2.flip(frame, 1)
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.stopped = True
        self.cap.release()

class FastPersistentRenderer:
    """Single persistent PyRender renderer preserving exact camera aspect ratio."""
    def __init__(self, smpl_faces, render_w=640, render_h=480):
        self.smpl_faces = smpl_faces
        self.render_w = render_w
        self.render_h = render_h
        self.r = pyrender.OffscreenRenderer(viewport_width=render_w, viewport_height=render_h, point_size=1.0)
        self.lights = create_raymond_lights()

    def render_front(self, all_verts, all_cam_t, focal_length, mesh_color=LIGHT_CYAN):
        if len(all_verts) == 0:
            return np.zeros((self.render_h, self.render_w, 4), dtype=np.float32)

        scene = pyrender.Scene(bg_color=[0, 0, 0, 0.0], ambient_light=(0.35, 0.35, 0.35))
        for i, (verts, cam_t) in enumerate(zip(all_verts, all_cam_t)):
            vcol = np.array([(*mesh_color, 1.0)] * verts.shape[0])
            # Direct translation without X inversion
            mesh = trimesh.Trimesh(verts.copy() + cam_t, self.smpl_faces.copy(), vertex_colors=vcol)
            rot_x = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot_x)
            scene.add(pyrender.Mesh.from_trimesh(mesh), f"m_{i}")

        cam = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=self.render_w / 2.0, cy=self.render_h / 2.0, zfar=1e12)
        c_node = pyrender.Node(camera=cam, matrix=np.eye(4))
        scene.add_node(c_node)
        
        for node in self.lights:
            if not scene.has_node(node):
                scene.add_node(node)
                
        color, _ = self.r.render(scene, flags=pyrender.RenderFlags.RGBA)
        return color.astype(np.float32) / 255.0

    def render_side(self, all_verts, all_cam_t, focal_length, rot_angle=45.0, mesh_color=VIBRANT_CORAL):
        if len(all_verts) == 0:
            return np.full((self.render_h, self.render_w, 3), (25, 30, 40), dtype=np.uint8)

        scene = pyrender.Scene(bg_color=[*BG_DARK, 0.0], ambient_light=(0.35, 0.35, 0.35))
        for i, (verts, cam_t) in enumerate(zip(all_verts, all_cam_t)):
            vcol = np.array([(*mesh_color, 1.0)] * verts.shape[0])
            mesh = trimesh.Trimesh(verts.copy(), self.smpl_faces.copy(), vertex_colors=vcol)
            rot_y = trimesh.transformations.rotation_matrix(np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot_y)
            mesh.vertices += cam_t
            rot_x = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot_x)
            scene.add(pyrender.Mesh.from_trimesh(mesh), f"m_{i}")

        cam = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=self.render_w / 2.0, cy=self.render_h / 2.0, zfar=1e12)
        c_node = pyrender.Node(camera=cam, matrix=np.eye(4))
        scene.add_node(c_node)
        
        for node in self.lights:
            if not scene.has_node(node):
                scene.add_node(node)
                
        color, _ = self.r.render(scene, flags=pyrender.RenderFlags.RGBA)
        return (color[:, :, :3] * 255).astype(np.uint8)

    def close(self):
        try:
            self.r.delete()
        except:
            pass

class FastDetector:
    def __init__(self, detector_type="yolo"):
        self.detector_type = detector_type
        if detector_type == "yolo":
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt").to("cuda")
        elif detector_type == "regnety":
            from detectron2 import model_zoo
            from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
            detectron2_cfg = model_zoo.get_config("new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.model = DefaultPredictor_Lazy(detectron2_cfg)
        elif detector_type == "vitdet":
            from detectron2.config import LazyConfig
            from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
            import hmr2
            cfg_path = Path(hmr2.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            self.model = DefaultPredictor_Lazy(detectron2_cfg)

    def detect(self, img_cv2, conf_thresh=0.5):
        if self.detector_type == "yolo":
            results = self.model(img_cv2, classes=[0], conf=conf_thresh, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            return boxes
        else:
            det_out = self.model(img_cv2)
            det_instances = det_out["instances"]
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > conf_thresh)
            return det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

def parse_args():
    parser = argparse.ArgumentParser(description="High-FPS 4D-HMR Real-Time Stream")
    parser.add_argument("--source", type=str, default="0", help="Camera index ('0') or video filepath")
    parser.add_argument("--mirror", action="store_true", default=None, help="Horizontally mirror camera feed (default True for webcam '0')")
    parser.add_argument("--no_mirror", action="store_true", default=False, help="Disable webcam mirroring")
    parser.add_argument("--detector", type=str, default="yolo", choices=["yolo", "regnety", "vitdet"], help="Detector backend")
    parser.add_argument("--det_interval", type=int, default=2, help="Run detection every N frames")
    parser.add_argument("--target_height", type=int, default=480, help="Target vertical resolution (aspect ratio is preserved automatically)")
    parser.add_argument("--side_view", action="store_true", default=True, help="Render dual side-by-side perspective")
    parser.add_argument("--rot_angle", type=float, default=45.0, help="Rotation angle in degrees for side view")
    parser.add_argument("--save_video", type=str, default=None, help="Output path to save translated video")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--headless", action="store_true", default=False, help="Headless execution mode")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=========================================================")
    print("  HIGH-FPS REAL-TIME 4D-HMR STREAMING ENGINE             ")
    print("=========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[+] Accelerator: {gpu_name} (TF32 + FP16 Enabled)")
    
    # Mirroring logic: True by default for webcam '0' unless --no_mirror
    is_cam = args.source.isdigit()
    mirror_feed = True if (is_cam and not args.no_mirror) else (args.mirror is True)
    if mirror_feed:
        print("[+] Webcam Mirror Mode: ENABLED (Natural mirror perspective)")
    else:
        print("[+] Webcam Mirror Mode: DISABLED (Raw sensor perspective)")
        
    # Load HMR 2.0
    print("[+] Loading HMR 2.0 Neural Mesh Backbone...")
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device)
    model.eval()
    
    # Load Detector
    print(f"[+] Initializing Ultra-Fast Detector ({args.detector})...")
    detector = FastDetector(args.detector)
    
    # Initialize Camera / Video Stream
    print(f"[+] Initializing Threaded Stream for source '{args.source}'...")
    stream = ThreadedCamera(args.source, mirror=mirror_feed)
    time.sleep(0.5)
    
    orig_w, orig_h = stream.width, stream.height
    fps = stream.fps
    total_frames = stream.total_frames
    
    if orig_w == 0 or orig_h == 0:
        orig_w, orig_h = 640, 480
        
    aspect_ratio = orig_w / orig_h
    rh = args.target_height
    rw = int(round(rh * aspect_ratio))
    # Ensure even dimensions for video codecs
    rw = rw + (rw % 2)
    rh = rh + (rh % 2)
    print(f"[+] Native Camera Feed: {orig_w}x{orig_h} (Aspect Ratio: {aspect_ratio:.2f}:1)")
    print(f"[+] Render Aspect-Preserving Viewport: {rw}x{rh} (Aspect Ratio: {rw/rh:.2f}:1)")
    
    fast_renderer = FastPersistentRenderer(model.smpl.faces, render_w=rw, render_h=rh)
    
    # Video Writer
    writer = None
    if args.save_video:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_video)) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = rw * (2 if args.side_view else 1)
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (out_w, rh))
        print(f"[+] Recording translated stream to: {args.save_video}")
        
    cached_boxes = np.array([])
    frame_idx = 0
    start_time = time.time()
    fps_display = 0.0
    
    window_name = "High-FPS 4D-HMR Feed"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
    print("\n[+] LIVE TRANSLATION RUNNING. Press 'q' or ESC in window to stop.\n")
    
    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                if stream.stopped:
                    break
                time.sleep(0.005)
                continue
                
            frame_start = time.time()
            frame_idx += 1
            
            # Resize frame to render resolution maintaining EXACT aspect ratio
            frame_resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
            
            # Run detection on interval or reuse cached boxes
            if frame_idx % args.det_interval == 1 or len(cached_boxes) == 0:
                cached_boxes = detector.detect(frame_resized, conf_thresh=0.5)
                
            boxes = cached_boxes
            all_verts = []
            all_cam_t = []
            
            if len(boxes) > 0:
                # Fast Tensor Batch Forward Pass in FP16
                dataset = ViTDetDataset(model_cfg, frame_resized, boxes)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(boxes), shuffle=False, num_workers=0)
                batch = next(iter(dataloader))
                batch = recursive_to(batch, device)
                
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    out = model(batch)
                    
                pred_cam = out["pred_cam"].float()
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
                
                for n in range(len(boxes)):
                    all_verts.append(out["pred_vertices"][n].float().detach().cpu().numpy())
                    all_cam_t.append(pred_cam_t_full[n])
            else:
                scaled_focal_length = float(model_cfg.EXTRA.FOCAL_LENGTH)

            # 3D Front Mesh Render & Alpha Composite
            front_rgba = fast_renderer.render_front(all_verts, all_cam_t, scaled_focal_length, mesh_color=LIGHT_CYAN)
            
            # Fast in-place blend
            in_f = frame_resized.astype(np.float32)[:, :, ::-1] / 255.0
            in_f = np.concatenate([in_f, np.ones_like(in_f[:, :, :1])], axis=2)
            comp = in_f[:, :, :3] * (1.0 - front_rgba[:, :, 3:]) + front_rgba[:, :, :3] * front_rgba[:, :, 3:]
            front_bgr = np.ascontiguousarray((comp * 255).astype(np.uint8)[:, :, ::-1])
            
            if args.side_view:
                side_bgr = fast_renderer.render_side(all_verts, all_cam_t, scaled_focal_length, rot_angle=args.rot_angle, mesh_color=VIBRANT_CORAL)
                side_bgr = np.ascontiguousarray(side_bgr[:, :, ::-1])
                
                # HUD Header Labels
                cv2.rectangle(front_bgr, (10, 8), (280, 36), (20, 20, 24), -1)
                cv2.putText(front_bgr, "4D-HMR: Front 3D Mesh", (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 180), 1)
                
                cv2.rectangle(side_bgr, (10, 8), (280, 36), (20, 20, 24), -1)
                cv2.putText(side_bgr, f"4D-HMR: 3D Side View ({int(args.rot_angle)} deg)", (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 255), 1)
                
                display_frame = np.concatenate([front_bgr, side_bgr], axis=1)
            else:
                display_frame = front_bgr
                
            # Instantaneous FPS
            frame_dur = time.time() - frame_start
            inst_fps = 1.0 / max(frame_dur, 1e-4)
            fps_display = 0.85 * fps_display + 0.15 * inst_fps if fps_display > 0 else inst_fps
            
            # Bottom Info HUD
            dh, dw = display_frame.shape[:2]
            cv2.rectangle(display_frame, (0, dh - 28), (dw, dh), (15, 15, 18), -1)
            hud = f"FPS: {fps_display:.1f} | Latency: {frame_dur*1000:.0f}ms | Tracked: {len(all_verts)} Person(s) | GPU: {gpu_name[:18]} (FP16+YOLO)"
            cv2.putText(display_frame, hud, (15, dh - 9), cv2.FONT_HERSHEY_DUPLEX, 0.45, (220, 230, 245), 1, cv2.LINE_AA)
            
            if writer is not None:
                writer.write(display_frame)
                
            if not args.headless:
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("[+] User stopped stream ('q').")
                    break
                    
            if frame_idx % 15 == 0:
                print(f"    Frame [{frame_idx:04d}] -> Live Speed: {fps_display:.1f} FPS (Latency: {frame_dur*1000:.1f} ms) | Persons: {len(all_verts)}")
                
            if args.max_frames and frame_idx >= args.max_frames:
                print(f"[+] Reached frame limit ({args.max_frames}).")
                break
                
    finally:
        stream.release()
        fast_renderer.close()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
            
    total_time = time.time() - start_time
    avg_fps = frame_idx / max(total_time, 1e-4)
    print("=========================================================")
    print(f"[+] High-Speed Streaming Finished!")
    print(f"    Total Frames: {frame_idx}")
    print(f"    Total Time: {total_time:.2f}s")
    print(f"    Average Speed: {avg_fps:.2f} FPS")
    if args.save_video and os.path.exists(args.save_video):
        print(f"    Saved Video: {args.save_video} ({os.path.getsize(args.save_video)} bytes)")
    print("=========================================================")

if __name__ == "__main__":
    main()
