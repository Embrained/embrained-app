# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import cv2
import numpy as np
import pandas as pd
import os

def get_rotated(img, angle):
    """Safely rotate an image bounding box using WarpAffine."""
    center = (img.shape[1]//2, img.shape[0]//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def create_circular_mask(h, w, center=None, radius=None):
    """Generate a clean boolean boolean mask to prevent rotation zero-pad clipping."""
    if center is None: center = (int(w/2), int(h/2))
    if radius is None: radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

class TelemetryExtractor:
    """
    Robust pipeline to extract absolute 2D position, exact physical 360-degree orientation, 
    and geometric bounding box raytracing from single physical webcam images using a 
    compiled global Master Template.
    """
    def __init__(self, datasets_dirs):
        self.datasets = datasets_dirs
        self.master_template = None
        self.arena_bounds = None
        self.median_bg = None
        self.robot_radius = 40  # Gap between centroid and bounding nose tip
        
    def _extract_pure_blobs(self, d_path):
        csv_path = os.path.join(d_path, 'episode_data.csv')
        img_dir = os.path.join(d_path, 'images')
        if not os.path.exists(csv_path) or not os.path.exists(img_dir):
            return []
            
        df = pd.read_csv(csv_path)
        sample_rows = df.sample(n=min(20, len(df)), random_state=42)
        frames = []
        for _, row in sample_rows.iterrows():
            ts = row['image_file'].replace('frame_', '').replace('.jpg', '')
            path = os.path.join(img_dir, f'webcam_frame_{ts}.jpg')
            if os.path.exists(path):
                frames.append(cv2.imread(path, 0))
                
        if not frames:
            return []
            
        local_bg = self.initialize_moving_background(frames).astype(np.uint8)
        self.median_bg = local_bg
        
        blobs = []
        for _, row in df.iterrows():
            ts = row['image_file'].replace('frame_', '').replace('.jpg', '')
            path = os.path.join(img_dir, f'webcam_frame_{ts}.jpg')
            if not os.path.exists(path): continue
            
            img_gray = cv2.imread(path, 0)
            diff = cv2.absdiff(img_gray, local_bg)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5),np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 200:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Generate raw principal axis (which tracks the widest wheel-to-wheel span)
                        mu20 = M["mu20"] / M["m00"]
                        mu02 = M["mu02"] / M["m00"]
                        mu11 = M["mu11"] / M["m00"]
                        primary_angle_deg = np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02))
                        
                        # Orthogonal 90-degree shift maps it securely to the longitudinal Nose-to-Tail axis
                        forward_angle_deg = (primary_angle_deg + 90) % 360
                        
                        # Apply Geometric Mass-Drift 180-degree resolution!
                        # The robot's Center of Mass (cx,cy) consistently drifts towards the tail
                        # relative to its explicit Euclidean Bounding Box centroid. 
                        x_b, y_b, w_b, h_b = cv2.boundingRect(c)
                        box_cx, box_cy = x_b + w_b / 2.0, y_b + h_b / 2.0
                        tail_vec = np.array([cx - box_cx, cy - box_cy])
                        
                        axis_rad = np.radians(forward_angle_deg)
                        axis_vec = np.array([np.cos(axis_rad), np.sin(axis_rad)])
                        
                        # Enforce unidirectional tail alignment across all frames
                        if np.dot(tail_vec, axis_vec) < 0:
                            forward_angle_deg = (forward_angle_deg + 180) % 360
                            
                        angle_deg = forward_angle_deg
                        
                        half = 50
                        h, w = thresh.shape
                        if cy-half >= 0 and cy+half < h and cx-half >= 0 and cx+half < w:
                            bin_crop = thresh[cy-half:cy+half, cx-half:cx+half]
                            blobs.append({
                                'img_dir': img_dir, 'ts': ts, 'cx': cx, 'cy': cy, 
                                'angle': angle_deg, 'bin_crop': bin_crop, 'ir': row.get('ir_reading', 0)
                            })
                        
        print(f"Extracted {len(blobs)} pure geometric tracking signatures from {os.path.basename(d_path)}")
        return blobs

    def calibrate(self):
        """Compile the absolute Master Robot Template and explicit room boundaries across all data."""
        print("Calibrating Telemetry Extractor across datasets...")
        all_blobs = []
        for d in self.datasets:
            all_blobs.extend(self._extract_pure_blobs(d))
            
        if not all_blobs:
            raise ValueError("No valid blobs extracted for calibration.")

        print(f"Aggregating Master Shape from {len(all_blobs)} transitions...")
        master = get_rotated(all_blobs[0]['bin_crop'], all_blobs[0]['angle'])
        template_sum = master.astype(np.float32)
        valid = 1
        
        for b in all_blobs[1:]:
            opt1 = get_rotated(b['bin_crop'], b['angle'])
            opt2 = get_rotated(b['bin_crop'], b['angle'] + 180)
            if np.mean((opt1 - master)**2) < np.mean((opt2 - master)**2):
                template_sum += opt1
            else:
                template_sum += opt2
            valid += 1
            
        self.master_template = (template_sum / valid).astype(np.uint8)
        
        from collections import defaultdict
        bounds_pts = defaultdict(lambda: {'x': [], 'y': []})
        for b in all_blobs:
            bounds_pts[b['img_dir']]['x'].append(b['cx'])
            bounds_pts[b['img_dir']]['y'].append(b['cy'])
            
        self.arena_bounds = {}
        for d_dir, pts in bounds_pts.items():
            self.arena_bounds[d_dir] = {
                'min_x': np.min(pts['x']) - self.robot_radius,
                'max_x': np.max(pts['x']) + self.robot_radius,
                'min_y': np.min(pts['y']) - self.robot_radius,
                'max_y': np.max(pts['y']) + self.robot_radius
            }
        
        for d_dir, b in self.arena_bounds.items():
            ds_name = os.path.basename(os.path.dirname(d_dir))
            print(f"  -> [{ds_name}] Envelope: X[{b['min_x']}, {b['max_x']}]  Y[{b['min_y']}, {b['max_y']}]")
            
        min_x_all = min(b['min_x'] for b in self.arena_bounds.values())
        max_x_all = max(b['max_x'] for b in self.arena_bounds.values())
        min_y_all = min(b['min_y'] for b in self.arena_bounds.values())
        max_y_all = max(b['max_y'] for b in self.arena_bounds.values())
        print(f"Calibration Complete. Global Envelope (Across {len(self.arena_bounds)} datasets): X[{min_x_all}, {max_x_all}]  Y[{min_y_all}, {max_y_all}]")
        return all_blobs

    def process_all(self, blobs=None):
        """Map absolute true heading and compute distance raycasts for all valid configurations."""
        if self.master_template is None or self.arena_bounds is None:
            blobs = self.calibrate()
            
        self._precompute_rotations()
        
        print("Tracing exact Euclidean physics hypotenuses against virtual boundaries...")
        results = []
        for item in blobs:
            masked_crop = item['bin_crop'] * self.mask
            best_mse = float('inf')
            best_theta = 0
            
            for theta, masked_temp in self.rotations:
                error = np.mean((masked_crop.astype(np.float32) - masked_temp.astype(np.float32))**2)
                if error < best_mse:
                    best_mse = error
                    best_theta = theta
            # The pristine master template intrinsically anchors statically to strictly Right (+0 degrees OpenCV).
            # Apply the mathematically stable Legacy Negative Phase Inversion + 180 degrees to align with the true physical IR nose
            yaw_rad = np.radians((-best_theta + 180) % 360)
            dx = np.cos(yaw_rad)
            dy = np.sin(yaw_rad)
            
            cx, cy = item['cx'], item['cy']
            b = self.arena_bounds[item['img_dir']]
            
            tx, ty = 99999, 99999
            if dx > 0: tx = (b['max_x'] - cx) / dx
            elif dx < 0: tx = (b['min_x'] - cx) / dx
            if dy > 0: ty = (b['max_y'] - cy) / dy
            elif dy < 0: ty = (b['min_y'] - cy) / dy
            
            dist_px = max(1.0, min(tx, ty) - self.robot_radius)
            
            results.append({
                'img_dir': item['img_dir'],
                'ts': item['ts'],
                'cx': cx, 'cy': cy,
                'dx': dx, 'dy': dy,
                'dist_px': dist_px,
                'yaw_deg': (np.degrees(np.arctan2(dy, dx)) + 360) % 360,
                'ir': item['ir']
            })
            
        df = pd.DataFrame(results)
        print(f"Extracted native analytical telemetry for {len(df)} frames.")
        return df

    def _precompute_rotations(self):
        """Precompute the 360 rotational matrices to save time during live inference."""
        if hasattr(self, 'rotations') and self.rotations:
            return
            
        self.mask = create_circular_mask(100, 100, radius=40)
        self.rotations = []
        for theta in range(0, 360, 2):
            r_temp = get_rotated(self.master_template, theta)
            self.rotations.append((theta, r_temp * self.mask))

    def process_single_frame(self, img_gray):
        """Processes a single live grayscale frame. Returns 4D latent vector format parameters."""
        if self.master_template is None or self.arena_bounds is None or self.median_bg is None:
            raise RuntimeError("TelemetryExtractor must be calibrated before processing single frames.")
            
        diff = cv2.absdiff(img_gray, self.median_bg)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return None
            
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) <= 200: return None
            
        M = cv2.moments(c)
        if M["m00"] == 0: return None
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        half = 50
        h, w = thresh.shape
        if not (cy-half >= 0 and cy+half < h and cx-half >= 0 and cx+half < w):
            return None
            
        bin_crop = thresh[cy-half:cy+half, cx-half:cx+half]
        self._precompute_rotations()
            
        best_mse = float('inf')
        best_theta = 0
        masked_crop = bin_crop * self.mask
        
        for theta, masked_temp in self.rotations:
            error = np.mean((masked_crop.astype(np.float32) - masked_temp.astype(np.float32))**2)
            if error < best_mse:
                best_mse = error
                best_theta = theta
                
        yaw_rad = np.radians((-best_theta + 180) % 360)
        
        # Enforce exact geometric Forward axis shift corresponding to the physical Center-of-Mass drift
        # Visual raymarch proofs confirm a 0-degree offset natively aligns the array output.
        dx = np.cos(yaw_rad)
        dy = np.sin(yaw_rad)
        
        # Standardize for CQL 4D Input identically to dataset 
        return {
            'cx_norm': cx / 640.0,
            'cy_norm': cy / 480.0,
            'cos_yaw': np.cos(yaw_rad),
            'sin_yaw': np.sin(yaw_rad),
            'raw_cx': cx,
            'raw_cy': cy
        }

    def save_cache(self, path):
        """Save the calibrated components to disk to skip recalculation."""
        if self.master_template is not None:
             np.savez(path, master_template=self.master_template, arena_bounds=self.arena_bounds, median_bg=self.median_bg)

    def load_cache(self, path):
        """Restore calibrated components from disk."""
        if os.path.exists(path):
             data = np.load(path, allow_pickle=True)
             self.master_template = data['master_template']
             self.arena_bounds = data.get('arena_bounds', {}).item() if data.get('arena_bounds', None) is not None else None
             self.median_bg = getattr(data, 'median_bg', None) if 'median_bg' in data else None
             return True
        return False

    def initialize_moving_background(self, frames, verbose=False):
        """Build exactly an Empty Room local-background mathematically off raw frame matrices natively assuming motion."""
        if verbose:
            print("Synthesizing Oracle Background using discrete structural arrays...")
        self.median_bg = np.median(frames, axis=0).astype(np.uint8)
        if verbose:
            print("Live Telemetry Environment successfully bootstrapped!")
        return self.median_bg

if __name__ == "__main__":
    # Example usage parsing all markov datasets dynamically
    base = r"c:\Users\chris\Embrained\software_suite\data"
    dirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and 'markov_' in d]
    
    extractor = TelemetryExtractor(dirs)
    df_telemetry = extractor.process_all()
    
    df_telemetry.to_csv("master_telemetry.csv", index=False)
    print("Saved master_telemetry.csv")
