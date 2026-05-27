import os
import sys

# Use the local ultralytics fork (contains custom modules like ECA)
_LOCAL_ULTRALYTICS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultralytics')
if _LOCAL_ULTRALYTICS not in sys.path:
    sys.path.insert(0, _LOCAL_ULTRALYTICS)

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from filterpy.kalman import KalmanFilter
import time
import threading
import queue
import torch
import time as _time

# ===========================================================
#  TOAD SYSTEM
# ===========================================================

class TOAD:
    # --- Add counters for YOLO and motion analysis frames ---
    yolo_frame_count = 0  # Global YOLO
    local_frame_count = 0 # Local YOLO
    motion_frame_count = 0
    global_yolo_time = 0.0
    local_yolo_time = 0.0
    motion_time = 0.0
    """
    TOAD: Tracker with Optical-flow Assisted Detection
    but adapted for table-tennis ball detection.
    """

    def _process_global_yolo_result(self, result, frame_shape):
        """Process a YOLO result object as in global YOLO mode, returning (boxes, status, motion_boxes, kalman_pred)."""
        import numpy as np
        h, w = frame_shape[:2]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        if len(boxes) > 0:
            best_conf = np.max(confs)
            best_box = boxes[np.argmax(confs)]
            if best_conf > self.conf_thresh:
                TOAD.yolo_frame_count += 1
                self.roi = best_box
                self._expand_roi(frame_shape, size=300)
                self.mode = "local"
                self.Ng += 1
                self.Nl = 0
                status = "yolo"
                final_box = best_box
                self.template = self._extract_template(np.zeros(frame_shape, dtype=np.uint8), final_box)
                _, _, pred_box = self._kalman_verification(final_box, None, frame_shape)
                self.last_kalman_pred = pred_box
                self.ball_missing_count = 0
                return [final_box], status, getattr(self, "last_motion_boxes", []), self.last_kalman_pred
        # Ball missing logic: skip motion analysis if missing > 5 frames
        self.ball_missing_count += 1
        if self.ball_missing_count > 5:
            pred_box = self._predict_kalman(h, w)
            status = "predict"
            self.mode = "global"
            return [pred_box], status, getattr(self, "last_motion_boxes", []), self.last_kalman_pred
        # Velocity check omitted for batch mode
        return [], "predict", getattr(self, "last_motion_boxes", []), self.last_kalman_pred

    def __init__(self, global_model_path, local_model_path=None,
                 conf_thresh=0.25, motion_thresh=0.05):
        # Global / Local YOLO
        self.global_model = YOLO(global_model_path)
        self.local_model = YOLO(local_model_path or global_model_path)

        self._half = torch.cuda.is_available()  # FP16 only on CUDA
        self.mode = "global"
        self.roi = None
        
        self.prev_roi = None
        self.prev_theta = 0.0
        self.last_motion_boxes = []
        self.last_kalman_pred = None

        # Parameters
        self.conf_thresh = conf_thresh
        self.motion_thresh = motion_thresh
        self.Ng = 0; self.Nl = 0
        self.Ng_thresh = 5  
        self.Nl_thresh = 1
        self.prev_center = None
        self.prev_prev_center = None

        # Previous frames
        self.prev_gray = None
        self.prev_prev_gray = None
        self.template = None

        # Keypoint extractor
        self.orb = cv2.ORB_create(200)

        # Kalman filter
        self.kf = self._init_kalman()

        self.avg_ball_area = 30      # running average of detected ball size
        self.ball_area_history = [] # store last N detected ball sizes
        self.max_history = 30       # use last 30 detections

        # Ball missing counter for skipping motion analysis
        self.ball_missing_count = 0

    # -------------------------------------------------------------
    def _init_kalman(self):
        """
        Traditional Kalman Fileter
        """
        kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0 #Tính bóng di chuyển 1 frame tương lai tại mỗi frame
    
        # State transition
        kf.F = np.eye(8)
        for i in range(4):
            kf.F[i, i + 4] = dt
    
        # Observation model
        kf.H = np.zeros((4, 8))
        kf.H[0, 0] = kf.H[1, 1] = kf.H[2, 2] = kf.H[3, 3] = 1.0
    
        # kf.P *= 100.0
        # kf.R *= 1.5
        # kf.Q *= 10.0

        # Covariances
        kf.P *= 50.0
        kf.R = np.diag([2, 2, 10, 10])   # trust cx,cy more than w,h
        kf.Q = np.eye(8)
        kf.Q[4:, 4:] *= 5.0                    # allow more velocity change
        kf.Q[6, 6] = kf.Q[7, 7] = 10.0         # adaptive w/h dynamics
    
        # ✅ Initialize default state
        kf.x = np.array([[320.0], [240.0], [20.0], [20.0],
                         [0.0],[0.0], [0.0], [0.0]])  # cx, cy, w, h, vx, vy, vw, vh
        return kf
        
    def _box_center(self, box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _is_motion_valid(self, prev_center, prev_prev_center, candidate_box):
        """Check direction, speed, and size consistency to reject noisy boxes."""

        # Box center
        cx, cy = self._box_center(candidate_box)
        px, py = prev_center
        if prev_prev_center is None:
            return True

        ppx, ppy = prev_prev_center

        # 1. Direction consistency
        v_prev = np.array([px - ppx, py - ppy], dtype=float)
        v_now  = np.array([cx - px, cy - py], dtype=float)

        if np.linalg.norm(v_prev) < 1e-3 or np.linalg.norm(v_now) < 1e-3:
            return True

        cos_theta = np.dot(v_prev, v_now) / (np.linalg.norm(v_prev) * np.linalg.norm(v_now))
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)

        if theta > np.deg2rad(60):    # change direction too much
            return False

        # 2. Speed consistency
        speed_prev = np.linalg.norm(v_prev)
        speed_now  = np.linalg.norm(v_now)
        if speed_now > 1.8 * speed_prev:
            return False

        # 3. Size consistency
        x1,y1,x2,y2 = candidate_box
        cw, ch = x2-x1, y2-y1
        pw, ph = self.kf.x[2], self.kf.x[3]  # Kalman w,h
        if abs(cw - pw)/max(pw,1) > 0.5:
            return False
        return True
        
  

    # ----------------------------------------------------------THE DETECTION FUCTION
    def detect(self, frame):
        """Full detection on one frame following the TOAD structure."""
        h, w = frame.shape[:2]

        # --- Local YOLO (focused on ROI) ---
        if self.mode == "local" and self.roi is not None: #if ROI exists
            #crop input image based on ROI
            x1, y1, x2, y2 = map(int, self.roi)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            
            #cropped image already
            if crop.size > 0:
                t0 = _time.perf_counter()
                result = self.local_model(crop, imgsz=320, half=self._half, verbose=False)[0]
                TOAD.local_yolo_time += _time.perf_counter() - t0
                boxes_local = result.boxes.xyxy.cpu().numpy() #return bounding boxes
                confs_local = result.boxes.conf.cpu().numpy() #return confidence score

                if len(boxes_local) > 0: #if the bounding boxes already
                    best_conf = np.max(confs_local) #find the best confidence score
                    best_box = boxes_local[np.argmax(confs_local)] + np.array([x1, y1, x1, y1])
                    if best_conf > self.conf_thresh: #checking with threshold
                        # ✅ confident local YOLO
                        TOAD.local_frame_count += 1
                        self.roi = best_box
                        self._expand_roi(frame.shape, size=300) #expand the ROI to 300x300 around ball center
                        self.Nl = 0
                        status = "yolo"
                        final_box = best_box
                        self.template = self._extract_template(frame, final_box) #update template for tracker
                        #NOTE: template is a patch of ball cropped from current frame
                        _, _, pred_box = self._kalman_verification(final_box, None, frame.shape)
                        self.last_kalman_pred = pred_box
                        self.ball_missing_count = 0  # Reset missing counter if detected
                        return [final_box], status, getattr(self, "last_motion_boxes", []), self.last_kalman_pred
            # If local fails, clear ROI and switch to global
            self.mode = "global"
            self.roi = None

        # --- Global YOLO ---
        t0 = _time.perf_counter()
        result = self.global_model(frame, imgsz=640, half=self._half, verbose=False)[0]
        TOAD.global_yolo_time += _time.perf_counter() - t0
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        if len(boxes) > 0:
            best_conf = np.max(confs)
            best_box = boxes[np.argmax(confs)]
            if best_conf > self.conf_thresh:
                # ✅ YOLO confident — trust detection directly
                TOAD.yolo_frame_count += 1
                self.roi = best_box
                self._expand_roi(frame.shape, size=300)
                self.mode = "local"
                self.Ng += 1
                self.Nl = 0
                status = "yolo"
                final_box = best_box
                self.template = self._extract_template(frame, final_box)
                _, _, pred_box = self._kalman_verification(final_box, None, frame.shape)
                self.last_kalman_pred = pred_box
                self.ball_missing_count = 0  # Reset missing counter if detected
                return [final_box], status, getattr(self, "last_motion_boxes", []), self.last_kalman_pred



        # --- Ball missing logic: skip motion analysis if missing > 5 frames ---
        self.ball_missing_count += 1
        if self.ball_missing_count > 5:
            # Only run global YOLO, skip motion analysis, just predict
            pred_box = self._predict_kalman(h, w)
            status = "predict"
            self.mode = "global"
            return [pred_box], status, getattr(self, "last_motion_boxes", []), self.last_kalman_pred

        # --- Velocity check: skip motion analysis if ball is nearly stationary ---
        skip_motion = False
        velocity_thresh = 2.0  # pixels
        if self.prev_center is not None and self.prev_prev_center is not None:
            dx = self.prev_center[0] - self.prev_prev_center[0]
            dy = self.prev_center[1] - self.prev_prev_center[1]
            velocity = (dx ** 2 + dy ** 2) ** 0.5
            if velocity < velocity_thresh:
                skip_motion = True
        elif self.prev_center is None:
            skip_motion = True

        self.roi = None

        motion_boxes = None
        if not skip_motion:
            t0 = _time.perf_counter()
            motion_boxes = self._motion_information(frame)
            TOAD.motion_time += _time.perf_counter() - t0
        self.last_motion_boxes = motion_boxes or []
        motion_box = None
        if motion_boxes and len(motion_boxes) > 0:
            motion_box = self._template_matching(frame, motion_boxes)

        # ======= APPLY OUTLIER CHECK FOR motion_box ========
        if motion_box is not None and self.prev_center is not None:
            if not self._is_motion_valid(self.prev_center,
                                         self.prev_prev_center,
                                         motion_box):
                nearest_box = None
                nearest_dist = 1e9
                cx_prev, cy_prev = self.prev_center
                for mb in motion_boxes:
                    mb_cx, mb_cy = self._box_center(mb)
                    d = np.hypot(mb_cx - cx_prev, mb_cy - cy_prev)
                    if d < nearest_dist and self._is_motion_valid(self.prev_center,
                                                                  self.prev_prev_center,
                                                                  mb):
                        nearest_dist = d
                        nearest_box = mb
                motion_box = nearest_box  # may be None

        # (2) Kalman refinement and fusion
        if motion_box is not None:
            final_box, kalman_status, pred_box = self._kalman_verification(None, motion_box, frame.shape)
            self.last_kalman_pred = pred_box
            status = "motion+kalman"
            TOAD.motion_frame_count += 1
            return [final_box], status, self.last_motion_boxes, pred_box
        else:
            # Motion failed too → pure Kalman prediction
            pred_box = self._predict_kalman(h, w)
            status = "predict"
            self.mode = "global"
            return [pred_box], status, self.last_motion_boxes, self.last_kalman_pred
                
      

    # ============================================================
    #  FMO Detector (Rozumnyi et al. CVPR 2017, Section 4.1)
    # ============================================================
    @staticmethod
    def fmo_detector_method_4_1(prev, curr, nxt, thresh=30):
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
    
        delta_plus  = cv2.absdiff(curr_gray, prev_gray)
        delta_zero  = cv2.absdiff(next_gray, prev_gray)
        delta_minus = cv2.absdiff(curr_gray, next_gray)
    
        _, db_plus  = cv2.threshold(delta_plus,  thresh, 255, cv2.THRESH_BINARY)
        _, db_zero  = cv2.threshold(delta_zero,  thresh, 255, cv2.THRESH_BINARY)
        _, db_minus = cv2.threshold(delta_minus, thresh, 255, cv2.THRESH_BINARY)
    
        motion_mask = cv2.bitwise_and(db_plus, db_minus)
        motion_mask = cv2.bitwise_and(motion_mask, cv2.bitwise_not(db_zero))
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        return motion_mask

    
    # -------------------------------------------------------------
    # (a) MOTION ANALYSIS   
    def _motion_information(self, frame):    # applying the binary substraction and Optical Flow

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None or self.prev_prev_gray is None:
            self.prev_prev_gray = self.prev_gray
            self.prev_gray = gray
            return None

        # Sparse optical flow using Lucas-Kanade
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
        motion_mask_flow = np.zeros_like(gray)
        if prev_pts is not None:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None)
            for i, (new, old) in enumerate(zip(next_pts[status==1], prev_pts[status==1])):
                a, b = new.ravel()
                c, d = old.ravel()
                # Draw motion as a line if movement is significant
                if np.hypot(a-c, b-d) > 1.0:
                    cv2.line(motion_mask_flow, (int(a), int(b)), (int(c), int(d)), 255, 2)


        
        # =====================================================
        # 🔹 FMO differencing: frames t−2 (a), t−1 (b), t (c)
        # =====================================================
        delta_plus  = cv2.absdiff(self.prev_gray, self.prev_prev_gray)   # b - a
        delta_zero  = cv2.absdiff(gray, self.prev_prev_gray)             # c - a
        delta_minus = cv2.absdiff(self.prev_gray, gray)                  # b - c
    
        _, db_plus  = cv2.threshold(delta_plus,  self.motion_thresh, 255, cv2.THRESH_BINARY)
        _, db_zero  = cv2.threshold(delta_zero,  self.motion_thresh, 255, cv2.THRESH_BINARY)
        _, db_minus = cv2.threshold(delta_minus, self.motion_thresh, 255, cv2.THRESH_BINARY)
    
        motion_mask_fmo = cv2.bitwise_and(db_plus, db_minus)
        motion_mask_fmo = cv2.bitwise_and(motion_mask_fmo, cv2.bitwise_not(db_zero))
    
        # --- fuse optical flow + FMO masks ---
        combined = cv2.addWeighted(motion_mask_fmo, 0.6, motion_mask_flow, 0.4, 0)

         # Morphological smoothing
        kernel = np.ones((3, 3), np.uint8)
        th = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        th = cv2.morphologyEx(th, cv2.MORPH_DILATE, kernel, iterations=1)
    
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.prev_prev_gray = self.prev_gray
        self.prev_gray = gray
    
        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            # --- size filtering ---
            if area < 8 or area > 500:  
                continue
        
            x, y, w, h = cv2.boundingRect(c)
        
            # --- aspect ratio filtering (ball ≈ circle) ---
            ratio = max(w, h) / (min(w, h) + 1e-6)
            if ratio > 1.8:
                continue
        
            # --- reject too-large motion region (hand, arm, racket) ---
            if w > 80 or h > 80:
                continue
        
            # --- ignore contours at borders ---
            if x < 2 or y < 2 or (x+w) > frame.shape[1]-2 or (y+h) > frame.shape[0]-2:
                continue
        
            box = [x, y, x+w, y+h]
        
            # ======================================================
            # 🔥 distance filter using prev_center (MOST IMPORTANT)
            # ======================================================
            if self.prev_center is not None:
                cx_prev, cy_prev = self.prev_center
                cx = (x + x+w) / 2
                cy = (y + y+h) / 2
                dist = np.hypot(cx - cx_prev, cy - cy_prev)
        
                if dist > 80:  # too far jump → reject
                    continue
        
            boxes.append(box)
        return boxes
    
    def _fft_match_template(self, roi, template):
        """Template matching using cv2.matchTemplate (normalized cross-correlation)."""
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if roi_gray.shape[0] < template_gray.shape[0] or roi_gray.shape[1] < template_gray.shape[1]:
            return np.zeros((1, 1), dtype=np.float32)
        result = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        # normalize [-1, 1] → [0, 1]
        result = (result.clip(-1.0, 1.0) + 1.0) * 0.5
        return result
    # -------------------------------------------------------------
    # (b) TEMPLATE MATCHING
    def _template_matching(self, frame, motion_boxes):
        if self.template is None or len(motion_boxes) == 0:
            return None
        best_score = 0
        best_box = None
        h_temp, w_temp = self.template.shape[:2]
    
        for box in motion_boxes:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
    
            # --- Multi-scale FFT-based correlation (replaces cv2.matchTemplate) ---
            scales = [0.7, 1.0, 1.3]
            best_corr = 0
            for s in scales:
                new_w = max(4, int(w_temp * s))
                new_h = max(4, int(h_temp * s))
                if new_w < 4 or new_h < 4:
                    continue
    
                template_resized = cv2.resize(self.template, (new_w, new_h))
                if roi.shape[0] < new_h or roi.shape[1] < new_w:
                    continue
    
                res = self._fft_match_template(roi, template_resized)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                best_corr = max(best_corr, max_val)  # already normalized [0,1]
    
            # --- displacement similarity (Eq. 5–9)
            if self.prev_roi is not None:
                px1, py1, px2, py2 = map(float, self.prev_roi)
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                d_now = np.hypot(cx - pcx, cy - pcy)
                theta_now = np.arctan2(cy - pcy, cx - pcx)
                d_norm = d_now / np.hypot(frame.shape[1], frame.shape[0])
                theta_norm = abs(theta_now - getattr(self, 'prev_theta', 0)) / np.pi
                disp_score = 1 - (d_norm + theta_norm) / 2
            else:
                disp_score = 0.5  # neutral
    
            # --- weighted matching (Eq. 10)
            k2, k3 = 0.6, 0.4
            score = k2 * best_corr + k3 * disp_score
    
            if score > best_score:
                best_score = score
                best_box = box
                self.prev_theta = theta_now if 'theta_now' in locals() else 0
    
        self.prev_roi = best_box
        return best_box

    # -------------------------------------------------------------
    # (c) KALMAN VERIFICATION
    def _kalman_verification(self, yolo_box, motion_box, frame_shape):
        # Save last state (from F_{t-1})
        prev_state = self.kf.x.copy() if self.kf.x is not None else None
         # 1. Prediction from previous frame
        self.kf.predict()
        pred_state = self.kf.x.copy()  # x_{t|t-1}
        px, py, pw, ph = pred_state[:4].flatten()
        # --- Compute adaptive dt scaling based on frame-to-frame speed ---
        if hasattr(self, "prev_center") and hasattr(self, "prev_prev_center") and \
           self.prev_center is not None and self.prev_prev_center is not None:
            dx = self.prev_center[0] - self.prev_prev_center[0]
            dy = self.prev_center[1] - self.prev_prev_center[1]
            measured_speed = np.hypot(dx, dy)
        else:
            measured_speed = 0
        
        speed_scale = np.clip(0.5 + 0.1 * measured_speed, 0.5, 4.0)
        
        # Apply scaled velocity correction
        vx, vy = self.kf.x[4:6].flatten()
        px = px + vx * (speed_scale - 1)
        py = py + vy * (speed_scale - 1)
        
        # Save for next frame
        self.prev_prev_center = getattr(self, "prev_center", None)
        self.prev_center = (px, py)
        
        # Construct predicted box
        pred_box = [px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2]

        # 2. Observation at current frame
        observed = yolo_box if yolo_box is not None else motion_box
        if observed is None:
            return pred_box, "predict", pred_box
    
        # ---------------------------
        # 3. Compute IoU
        # ---------------------------
        iou_val = self._iou(pred_box, observed)
        # ---------------------------
        # 4. Update step if observation consistent
        # ---------------------------

        # --------------------------------------
        # (1) Strong match → standard Kalman update
        # --------------------------------------
        if iou_val > 0.25:
            x1, y1, x2, y2 = observed
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw, bh = (x2 - x1), (y2 - y1)
            z = np.array([cx, cy, bw, bh])
            self.kf.update(z)  # Kalman correction step
        
            # After update, predict next frame using velocity
            final_box = self._predict_next_box()
            return final_box, "match", pred_box
        
        # --------------------------------------
        # (2) Weak or no match → soft update
        # --------------------------------------
        else:
            # Blend old and new observation to avoid freezing
            x1, y1, x2, y2 = observed
            cx_o, cy_o = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw_o, bh_o = (x2 - x1), (y2 - y1)
        
            # Get current state
            cx, cy, w, h, vx, vy, vw, vh = self.kf.x.flatten()
        
            # Soft correction toward observation
            alpha = 0.8  # partial trust in observation
            cx_new = (1 - alpha) * cx + alpha * cx_o
            cy_new = (1 - alpha) * cy + alpha * cy_o
            w_new  = (1 - alpha) * w  + alpha * bw_o
            h_new  = (1 - alpha) * h  + alpha * bh_o
        
            # Update state manually
            self.kf.x[:4] = np.array([[cx_new], [cy_new], [w_new], [h_new]])
            # self.kf.x[:4] = np.array([cx_new, cy_new, w_new, h_new])

            # Predict using motion model
            final_box = self._predict_next_box()
            return final_box, "predict", pred_box



    # -------------------------------------------------------------
    # Utility methods
    def _extract_template(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        # adapt padding to ball size
        bw, bh = x2 - x1, y2 - y1
        pad = int(max(bw, bh) * 0.5)  # 50% of ball size
        # pad = 3
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(frame.shape[1] - 1, x2 + pad), min(frame.shape[0] - 1, y2 + pad)
    
        if x2 <= x1 or y2 <= y1:
            return None  # invalid region
    
        template = frame[y1:y2, x1:x2].copy()
        if template.size == 0:
            return None  # empty crop
    
        area = (x2 - x1) * (y2 - y1)
        if area > 0:
            self.ball_area_history.append(area)
            if len(self.ball_area_history) > self.max_history:
                self.ball_area_history.pop(0)
            self.avg_ball_area = np.mean(self.ball_area_history)

        return template

    def _predict_kalman(self, h, w):
        self.kf.predict()
        px, py, pw, ph = self.kf.x[:4].flatten()
        return [px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2]

    def _predict_box(self):
        cx, cy, w, h, vx, vy, vw, vh = self.kf.x.flatten()
        dt = 1.0
        # Predict position and size using motion
        cx_pred = cx + vx * dt
        cy_pred = cy + vy * dt
        w_pred  = max(1, w + vw * dt)
        h_pred  = max(1, h + vh * dt)
        return [cx_pred - w_pred / 2, cy_pred - h_pred / 2,
                cx_pred + w_pred / 2, cy_pred + h_pred / 2]

    def _iou(self, A,B):
        xA,yA=max(A[0],B[0]),max(A[1],B[1])
        xB,yB=min(A[2],B[2]),min(A[3],B[3])
        inter=max(0,xB-xA)*max(0,yB-yA)
        areaA=(A[2]-A[0])*(A[3]-A[1])
        areaB=(B[2]-B[0])*(B[3]-B[1])
        return inter/float(areaA+areaB-inter+1e-6)

    def _expand_roi(self, frame_shape, size=300):
        """Expand ROI to a fixed-size 300x300 region around the last detection."""
        if self.roi is None:
            return
    
        h, w, _ = frame_shape
        x1, y1, x2, y2 = map(int, self.roi)
    
        # Compute current center of detected object
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
        # Define half-size of ROI box
        half = size // 2
    
        # Clamp to frame boundaries
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w - 1, cx + half)
        y2 = min(h - 1, cy + half)
    
        # Update ROI
        self.roi = [x1, y1, x2, y2]
  
        
    def _is_valid_ball(self, box, frame_shape, max_ratio=0.01):
        h, w, _ = frame_shape
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        box_area = bw * bh
        frame_area = w * h
        return (box_area / frame_area) < max_ratio

    def _fmo_detector(self, prev, curr, nxt, thresh=30):
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
    
        delta_plus  = cv2.absdiff(curr_gray, prev_gray)
        delta_zero  = cv2.absdiff(next_gray, prev_gray)
        delta_minus = cv2.absdiff(curr_gray, next_gray)
    
        _, db_plus  = cv2.threshold(delta_plus,  thresh, 255, cv2.THRESH_BINARY)
        _, db_zero  = cv2.threshold(delta_zero,  thresh, 255, cv2.THRESH_BINARY)
        _, db_minus = cv2.threshold(delta_minus, thresh, 255, cv2.THRESH_BINARY)
    
        motion_mask = cv2.bitwise_and(db_plus, db_minus)
        motion_mask = cv2.bitwise_and(motion_mask, cv2.bitwise_not(db_zero))
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        return motion_mask

    def _predict_next_box(self):
        """Predict one-frame-ahead box using current Kalman velocity (for fast objects)."""
        cx, cy, w, h, vx, vy, vw, vh = self.kf.x.flatten()
        dt = 1.5  # one frame ahead
        cx_pred = cx + vx * dt
        cy_pred = cy + vy * dt
        w_pred  = max(1, w + vw * dt)
        h_pred  = max(1, h + vh * dt)
        return [cx_pred - w_pred / 2, cy_pred - h_pred / 2,
                cx_pred + w_pred / 2, cy_pred + h_pred / 2]

def _safe_int(v):
    """Extract a scalar and cast to int safely."""
    import numpy as np
    if isinstance(v, (list, tuple)):
        v = v[0]
    if isinstance(v, np.ndarray):
        v = v.item() if v.size == 1 else float(v.flatten()[0])
    return int(v)

# ===========================================================
# Example usage
# ===========================================================
if __name__=="__main__":

    video_path=f"/video_path_here"
    output_path=f"/output_path_here" #For output video with drawn boxes
    label_output_dir = f"/output_label_dir_here" #For output text files of detected boxes (one txt per frame, same format as YOLO labels)
    global_model_path=f"/link_checkpoint_of_enhanced_YOLO" 
    local_model_path=f"/link_checkpoint_of_enhanced_YOLO_for_Ball_ROI" 
   
    
    os.makedirs(label_output_dir, exist_ok=True)

    cap=cv2.VideoCapture(video_path)
    print("Opened:", cap.isOpened())
    print("Total frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if video_fps <= 0:
        video_fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 9999
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = TOAD(global_model_path, local_model_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height))
    _write_q = queue.Queue(maxsize=60)
    def _writer_worker():
        while True:
            f = _write_q.get()
            if f is None:
                break
            out.write(f)
    _writer_thread = threading.Thread(target=_writer_worker, daemon=True)
    _writer_thread.start()

    print("🎥 Starting TOAD (paper style)…")
    frame_idx = 0
    pipeline_start = time.perf_counter()

    batch_size = 8  # You can tune this
    batch_frames = []
    batch_indices = []
    batch_shapes = []
    with tqdm(total=total, desc="TOAD", unit="frame") as pbar:
        while True:
            frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                # Process any remaining batch
                if batch_frames:
                    results = detector.global_model(batch_frames, imgsz=640, half=detector._half, verbose=False)
                    for idx, result, shape in zip(batch_indices, results, batch_shapes):
                        boxes, status, motion_boxes, kalman_pred = detector._process_global_yolo_result(result, shape)
                        _write_q.put(frames_buffer[idx])
                        pbar.update(1)
                        frame_idx += 1
                break

            # Store frame for drawing after batch
            if 'frames_buffer' not in locals():
                frames_buffer = {}
            frames_buffer[frame_idx] = frame.copy()

            # If in global mode, accumulate batch
            if detector.mode == "global":
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                batch_shapes.append(frame.shape)
                if len(batch_frames) == batch_size:
                    results = detector.global_model(batch_frames, imgsz=640, half=detector._half, verbose=False)
                    for idx, result, shape in zip(batch_indices, results, batch_shapes):
                        boxes, status, motion_boxes, kalman_pred = detector._process_global_yolo_result(result, shape)
                        _write_q.put(frames_buffer[idx])
                        pbar.update(1)
                        frame_idx += 1
                    batch_frames = []
                    batch_indices = []
                    batch_shapes = []
                continue
            else:
                # If not in global mode, process frame as usual
                boxes, status, motion_boxes, kalman_pred  = detector.detect(frame)
                color = (
                    (0, 255, 0)   if status == "match"   else  # green
                    (0, 255, 255) if status == "yolo"    else  # yellow
                    (255, 0, 255) if status == "fused"   else  # magenta
                    (0, 128, 255) if status == "predict" else  # blue
                    (180, 180, 180)                         # gray (ignore)
                )
                # --- Draw detected/fused object box ---
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0])
                    if status == "yolo":
                        color_box = (0, 255, 255)     # yellow
                        label = "YOLO"
                    elif status == "fused":
                        color_box = (255, 0, 255)     # magenta
                        label = "FUSED"
                    elif status == "motion+kalman":
                        color_box = (0, 128, 255)     # blue
                        label = "MOTION"
                    else:
                        color_box = color
                        label = status.upper()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)
                    cv2.putText(frame, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)
                _write_q.put(frame)
                pbar.update(1)
                frame_idx += 1


                       

            frame_elapsed = time.perf_counter() - frame_start
            elapsed_total = time.perf_counter() - pipeline_start
            running_avg_fps = (frame_idx + 1) / elapsed_total if elapsed_total > 1e-9 else 0.0

            cv2.putText(frame, f"Avg FPS: {running_avg_fps:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    cap.release()
    _write_q.put(None)
    _writer_thread.join()
    out.release()
    total_time = time.perf_counter() - pipeline_start
    print(f"🎬 Processed {frame_idx} frames in {total_time:.2f} seconds")
    avg_processing_fps = (frame_idx / total_time) if total_time > 1e-9 else 0.0
    print(f"✅ Output video saved to: {output_path}")
    print(f"✅ Bounding boxes exported to: {label_output_dir}")
    print(f"🎞️ Source video FPS: {video_fps:.2f}")
    print(f"🚀 Average processing FPS: {avg_processing_fps:.2f}")
    print("\n--- Detection Mode Statistics ---")
    print(f"Frames detected by global YOLO: {TOAD.yolo_frame_count}")
    print(f"Frames detected by local YOLO: {TOAD.local_frame_count}")
    print(f"Frames detected by motion analysis: {TOAD.motion_frame_count}")
    print("\n--- Detection Time (seconds) ---")
    print(f"Total time in global YOLO: {TOAD.global_yolo_time:.2f}s")
    print(f"Total time in local YOLO: {TOAD.local_yolo_time:.2f}s")
    print(f"Total time in motion analysis: {TOAD.motion_time:.2f}s")
