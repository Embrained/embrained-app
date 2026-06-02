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


import logging
import time
import threading
import cv2
import requests
import numpy as np

try:
    import serial
except ImportError:
    serial = None
from contextlib import closing
import websocket
import socket
from config import (
    CMD_FWD_VAL, CMD_REV_VAL, LEFT_MOTOR_INVERT,
    BASE_SPEED, TURN_SPEED, ACTION_NAMES,
    DATA_DIR, IMG_W, IMG_H, ROBOT_IP, 
    RECORD_W, RECORD_H # [NEW]
)
from modules.robot_controller import RobotController

class NervousSystem:
    def __init__(self, dummy_mode=False, robot_ip=None, stream_port=81, spikerbot_quirks=False, use_webcam=False):
        self.ws = None
        self.spikerbot_quirks = spikerbot_quirks
        self.running = True
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.dummy_mode = dummy_mode
        self.use_webcam = use_webcam
        
        self.latest_webcam_frame = None
        self.webcam_lock = threading.Lock()
        self.webcam_cap = None
        
        self.telemetry = {'dist': '0', 'batt': '0', 'ping': '0'}
        self.telemetry_lock = threading.Lock()
        self.last_log_time = 0 # [DEBUG]
        self.latest_ping_ms = 0.0



        # Determine Robot IP
        self.robot_ip = robot_ip if robot_ip else ROBOT_IP
        
        # Determine Video Source
        # Dynamic URL construction based on IP
        self.stream_port = stream_port
        self.ws_url = f"ws://{self.robot_ip}/ws"
        if self.stream_port == 80:
             self.video_source = f"http://{self.robot_ip}/stream"
        else:
             self.video_source = f"http://{self.robot_ip}:{self.stream_port}/stream"

        if self.dummy_mode:
            self.video_source = None
        
        # Camera State
        self.camera_valid = False
        self.mock_video = True if self.dummy_mode else False
        self.cap = None
        self.serial = None
        
        if not self.dummy_mode:
            # 1. Pre-check / Wake-up Connection (Socket) - Strict like teleop.py
            # SKIP PING for Plexus/SpikerBot (Port 80/81) and Serial Modes
            # Pinging Port 81 and immediately closing it hangs the ESP32 WiFiClient
            if self.stream_port not in [80, 81]:
                logging.info(f"Pinging Robot at {self.robot_ip}:{self.stream_port} ...")
                try:
                    with socket.create_connection((self.robot_ip, self.stream_port), timeout=5):
                        pass
                    logging.info("Robot is reachable.")
                except (socket.timeout, socket.error) as e:
                    # [MODIFIED] Relaxed check for V5 Firmware (Port 81 might not reply to ping)
                    logging.warning(f"Robot host unreachable (Socket Check): {e}. Proceeding anyway...")
            else:
                logging.info(f"Skipping Socket Ping for {self.robot_ip}:{self.stream_port} (Single-Client Stream)")

            # 2. Connect WebSocket
            logging.info(f"Connecting to WS: {self.ws_url} ...")
            
            # Retry Mechanism for WebSocket Connection
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Increased timeout to 10s
                    self.ws = websocket.create_connection(self.ws_url, timeout=10, ping_interval=10, ping_timeout=4)
                    logging.info("WebSocket Connected.")
                    break # Success
                except Exception as e:
                    logging.warning(f"WS Connection Attempt {attempt+1}/{max_retries} Failed: {e}")
                    
                    # Check for Protocol Violation (Robot sending data before handshake)
                    if "codec can't decode" in str(e) or "invalid start byte" in str(e):
                        logging.warning("POTENTIAL PROTOCOL VIOLATION: Robot is sending data before handshake completes.")
                        logging.warning("SOLUTION: Please perform a HARD REBOOT of the robot (Power Cycle).")
                        time.sleep(3.0) # Wait longer to let buffers drain
                    elif attempt < max_retries - 1:
                        time.sleep(1.0) # Wait before retry
                    else:
                        # Final Attempt Failed
                        if self.stream_port == 80:
                            logging.warning(f"WS Connection Failed (Expected for Plexus Video-Only): {e}")
                            self.ws = None
                        else:
                            raise ConnectionError(f"WS Connection Failed after {max_retries} attempts: {e}")

            # 3. Connect Camera (HTTP Stream OR Local Video in NON-DUMMY mode if use_webcam is on)
            logging.info(f"Stream Configuration: {self.video_source}")

            # 4. Start Threads
            self.vid_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.vid_thread.start()
            
            if self.use_webcam:
                self.webcam_thread = threading.Thread(target=self._webcam_loop, daemon=True)
                self.webcam_thread.start()
            
            if self.ws:
                self.drain_thread = threading.Thread(target=self._ws_drain_loop, daemon=True)
                self.drain_thread.start()
        else:
            logging.debug("NervousSystem initialized in DUMMY MODE.")
            
            logging.debug("No Camera Source (Webcam Disabled). Defaulting to Mock Video.")
            self.camera_valid = True # Mock is valid
            self.mock_video = True

            # Start Video Loop anyway (it handles mock generation)
            self.vid_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.vid_thread.start()
        # Robot Controller (Normalization Logic)
        self.robot = RobotController(config_path="robot_config.json")

        # Motor Speeds (Refined: Start at 0.0 when stopped)
        self.base_speed = 0.0
        self.turn_speed = 0.0
        
        # [NEW] Independent V/W state for persistent WASD teleop
        self.manual_v_pwm = 0
        self.manual_w_pwm = 0

        # Background Ping Loop (Started after everything is initialized)
        self.ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self.ping_thread.start()


    def _process_frame(self, frame, flip=True):
        """
        Enforce 4:3 Aspect Ratio (Center Crop) and Resize to Config Dimensions.
        User Requirement: "Crop widescreen to fit 3x4 (height by width)" -> 4:3 Aspect.
        """
        if frame is None:
            return None

        h, w = frame.shape[:2]
        target_aspect = 4.0 / 3.0
        current_aspect = w / h

        # If significantly wider than 4:3 (e.g. 16:9), crop width
        if current_aspect > target_aspect + 0.05:
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            frame = frame[:, start_x:start_x+new_w]
        
        # If significantly taller (unlikely for landscape cams, but possible), crop height
        elif current_aspect < target_aspect - 0.05:
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            frame = frame[start_y:start_y+new_h, :]

        if frame is not None:
             # Scale/Crop logic ...
             pass 

        if flip:
            # Flip vertically only (camera is physically upside-down but was mirroring left-to-right with -1)
            # We use 0 (vertical flip) instead of -1 (both axes) so objects on the right appear on the right
            frame = cv2.flip(frame, 0) 
        
        # Resize to standard dimensions (320x240 for UI/Recording)
        # Using RECORD_W, RECORD_H from config
        frame = cv2.resize(frame, (RECORD_W, RECORD_H))
        return frame

    def _ping_loop(self):
        """Measures request-response latency to the robot's web server periodically."""
        while self.running:
            if getattr(self, 'dummy_mode', False) or not getattr(self, 'robot_ip', None):
                self.latest_ping_ms = 14.0 # Fake ping for UI
                with self.telemetry_lock:
                    self.telemetry['ping'] = "14"
                time.sleep(2.0)
                continue
                
            try:
                start_t = time.perf_counter()
                
                # We request a non-existent endpoint to guarantee a fast 404 response instantly
                url = f"http://{self.robot_ip}/ping_endpoint_for_rtt"
                res = requests.get(url, timeout=1.0)
                
                # Any response means the round trip succeeded
                rtt_ms = (time.perf_counter() - start_t) * 1000
                
                if self.latest_ping_ms == 0:
                    self.latest_ping_ms = rtt_ms
                else:
                    self.latest_ping_ms = (self.latest_ping_ms * 0.7) + (rtt_ms * 0.3)
                    
                with self.telemetry_lock:
                    self.telemetry['ping'] = f"{int(self.latest_ping_ms)}"
            except Exception:
                # If ping fails, we just keep the last or show 999
                with self.telemetry_lock:
                    self.telemetry['ping'] = "999"
                    
            time.sleep(1.0) # Ping 1Hz

    def _webcam_loop(self):
        """Dedicated background thread for external webcam capturing."""
        try:
            logging.info("Starting external webcam background thread...")
            # Windows DSHOW frequently deadlocks on modern builds.
            # Using MSMF (default) to ensure non-blocking failures.
            self.webcam_cap = None
            for cam_idx in [0, 1, 2]:
                logging.info(f"Trying to open webcam device {cam_idx}...")
                
                # Default MSMF backend
                cap = cv2.VideoCapture(cam_idx)
                
                if cap.isOpened():
                    # Quick test read to ensure it actually works and isn't a phantom handle
                    ret, _ = cap.read()
                    if ret:
                        self.webcam_cap = cap
                        logging.info(f"Successfully bound to webcam device {cam_idx}.")
                        break
                    else:
                        logging.warning(f"Device {cam_idx} opened but read() failed. Releasing.")
                        cap.release()
                        
            if self.webcam_cap is None or not self.webcam_cap.isOpened():
                 logging.error("Failed to open ANY external webcam (devices 1, 0, 2).")
                 return
                 
            # Optional: Set webcam properties for low latency if needed
            self.webcam_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            frames_grabbed = 0
            while self.running:
                ret, frame = self.webcam_cap.read()
                if ret:
                    frames_grabbed += 1
                    if frames_grabbed % 100 == 1:
                        logging.debug(f"Webcam thread running smoothly. Grabbed {frames_grabbed} frames.")
                        
                    with self.webcam_lock:
                        self.latest_webcam_frame = frame
                    # Limit frame rate slightly to avoid 100% CPU on fast webcams
                    time.sleep(0.01)
                else:
                    logging.warning("Failed to grab external webcam frame; retrying...")
                    time.sleep(1.0)
        except Exception as e:
            logging.error(f"CRITICAL ERROR in webcam thread: {e}")
        finally:
            if hasattr(self, 'webcam_cap') and self.webcam_cap is not None:
                self.webcam_cap.release()
            logging.info("External webcam thread exited.")

    def _video_loop(self):
        """
        Continuously read frames from HTTP stream or Webcam.
        Implemented robustness from teleop.py to reconnect on drops.
        """

        if self.video_source is None and not self.mock_video:
            return
        
        # Mock generator state
        cx, cy = 80, 60
        dx, dy = 1, 1
        frame_count = 0


        while self.running:
            # Lazy Init Camera in Thread
            if not self.mock_video and (self.cap is None or not self.cap.isOpened()):
                 if self.video_source is not None:
                      logging.info(f"Connecting to Stream (Thread): {self.video_source} ...")
                      
                      # Handle Local Webcam (Integer)
                      if isinstance(self.video_source, int):
                          # Try Windows DirectShow first (often fixes MSMF errors), then default
                          self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
                          if not self.cap.isOpened():
                              self.cap = cv2.VideoCapture(self.video_source)
                              
                          if not self.cap.isOpened():
                              logging.warning(f"Webcam {self.video_source} Failed to Open.")
                              # Fallback to Mock if local cam fails repeatedly in dummy mode
                              self.retry_count = getattr(self, 'retry_count', 0) + 1
                              if self.dummy_mode and self.retry_count > 10:
                                   logging.warning("Webcam failed repeatedly. Switching to MOCK VIDEO fallback.")
                                   self.mock_video = True
                                   self.camera_valid = True
                                   continue
                                   
                              time.sleep(1.0)
                              continue
                          else:
                                logging.info(f"Webcam {self.video_source} Opened.")
                                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # [FIX] Minimize Latency
                                self.retry_count = 0 # Reset on success
 
                      else:
                          # Handle Network Stream (String URL)
                          # [MODIFIED] Prioritize robust manual reader for network streams to avoid CV2/FFmpeg stalls
                          logging.info(f"Connecting to Network Stream: {self.video_source}")
                          self._fetch_mjpeg_frames()
                          if not self.running: break
                          time.sleep(1.0)
                          continue
            
            try:
                if self.mock_video:
                    if getattr(self, '_mock_sim', None) is None:
                        import torch
                        import numpy as np
                        
                        class NeuralVisualizer:
                            def __init__(self, width, height, num_nodes=40):
                                self.width = width
                                self.height = height
                                self.num_nodes = num_nodes
                                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                
                                self.pos = torch.rand((num_nodes, 2), device=self.device)
                                self.pos[:, 0] *= width
                                self.pos[:, 1] *= height
                                
                                self.vel = (torch.rand((num_nodes, 2), device=self.device) - 0.5) * 1.5
                                
                                self.base_sizes = torch.rand(num_nodes, device=self.device) * 2.0 + 1.0
                                self.phases = torch.rand(num_nodes, device=self.device) * 2 * np.pi
                                
                                self.connection_dist = min(width, height) * 0.25

                            def step(self):
                                self.pos += self.vel
                                
                                # Bounce off walls softly
                                out_x = (self.pos[:, 0] < 5) | (self.pos[:, 0] > self.width - 5)
                                self.vel[out_x, 0] *= -1
                                self.pos[:, 0] = torch.clamp(self.pos[:, 0], 5, self.width - 5)
                                
                                out_y = (self.pos[:, 1] < 5) | (self.pos[:, 1] > self.height - 5)
                                self.vel[out_y, 1] *= -1
                                self.pos[:, 1] = torch.clamp(self.pos[:, 1], 5, self.height - 5)
                                
                                t = time.time()
                                sizes = self.base_sizes + torch.sin(self.phases + t * 3.0) * 1.5
                                sizes = torch.clamp(sizes, min=0.5)
                                
                                pos_np = self.pos.cpu().numpy()
                                sizes_np = sizes.cpu().numpy()
                                
                                # Background: Off-white / Clean living room floor
                                frame = np.full((self.height, self.width, 3), 245, dtype=np.uint8)
                                
                                # Draw edges (connections)
                                for i in range(self.num_nodes):
                                    for j in range(i + 1, self.num_nodes):
                                        dist = float(np.linalg.norm(pos_np[i] - pos_np[j]))
                                        if dist < self.connection_dist:
                                            thickness = int(max(1, 2 - (dist / self.connection_dist) * 2))
                                            # Dark Grey structural lines
                                            color = (180, 180, 180)
                                            cv2.line(frame, (int(pos_np[i,0]), int(pos_np[i,1])), 
                                                     (int(pos_np[j,0]), int(pos_np[j,1])), color, thickness, cv2.LINE_AA)
                                
                                # Draw nodes
                                for i in range(self.num_nodes):
                                    x, y = int(pos_np[i, 0]), int(pos_np[i, 1])
                                    s = int(sizes_np[i])
                                    
                                    if i == 0:
                                        # Intelligence / Current location (Blue/Purple pulsating)
                                        color = (255, 50, 150) # Vibrant Purple BGR
                                        cv2.circle(frame, (x, y), s + 3, color, -1, cv2.LINE_AA)
                                        # Outer glow
                                        glow = int((np.sin(t * 5) + 1) * 2) + s + 4
                                        cv2.circle(frame, (x, y), glow, (250, 150, 200), 1, cv2.LINE_AA)
                                    elif i < 4:
                                        # Rewards / Goal locations (Green)
                                        color = (50, 200, 50) # Vibrant Green BGR
                                        cv2.circle(frame, (x, y), s + 2, color, -1, cv2.LINE_AA)
                                    elif i < 7:
                                        # Alerts / Heat (Orange)
                                        color = (0, 140, 255) # Orange BGR
                                        cv2.circle(frame, (x, y), s + 1, color, -1, cv2.LINE_AA)
                                    else:
                                        # Standard functional nodes (Dark Grey / Yellow)
                                        if i % 5 == 0:
                                            color = (0, 220, 255) # Yellow BGR
                                        else:
                                            color = (120, 120, 120) # Neural Topology grey
                                        cv2.circle(frame, (x, y), s, color, -1, cv2.LINE_AA)
                                        
                                return frame
                                
                        self._mock_sim = NeuralVisualizer(RECORD_W, RECORD_H)

                    frame = self._mock_sim.step()
                    
                    with self.frame_lock:
                        self.latest_frame = frame
                    time.sleep(0.033) # ~30fps
                    continue

                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        # Process Real Frame (Crop Widescreen + Resize)
                        frame = self._process_frame(frame)
                        
                        with self.frame_lock:
                            self.latest_frame = frame
                        
                        frame_count += 1
                    else:
                         # Stream dropped, try to reconnect
                        logging.warning(f"Stream dropped (ret=False). Reconnecting to {self.video_source}...")
                        self.cap.release()
                        time.sleep(1.0)
                else:
                    time.sleep(0.1)

            except Exception as e:
                 logging.error(f"Video Loop Error: {e}")
                 # Critical Failure Handling
                 # if we loop too fast with errors, we switch to mock
                 if "Invalid URL" in str(e): # Stop invalid URL fallback loops quickly
                      logging.error("Critical Stream Error: Switching to Mock Video.")
                      self.mock_video = True
                 time.sleep(1.0)

    def _fetch_mjpeg_frames(self):
        """Robust MJPEG reader using requests. Parses raw JPEG markers from multipart stream."""
        logging.info(f"Connecting to Stream (Requests): {self.video_source} ...")
        frame_count = 0
        try:
            with requests.get(self.video_source, stream=True, timeout=(5, 15)) as r:
                if r.status_code != 200:
                    logging.error(f"Stream Status Error: {r.status_code}")
                    return

                bytes_buffer = bytes()
                for chunk in r.iter_content(chunk_size=16384):
                    if not self.running: break
                    bytes_buffer += chunk
                    
                    while True:
                        a = bytes_buffer.find(b'\xff\xd8') # JPEG Start
                        b = bytes_buffer.find(b'\xff\xd9') # JPEG End
                        
                        if a != -1 and b != -1:
                            if a > b:
                                bytes_buffer = bytes_buffer[a:]
                                continue

                            jpg = bytes_buffer[a:b+2]
                            bytes_buffer = bytes_buffer[b+2:]
                            
                            try:
                                nparr = np.frombuffer(jpg, dtype=np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                if frame is not None:
                                    frame = self._process_frame(frame)
                                    with self.frame_lock:
                                        self.latest_frame = frame
                                    frame_count += 1
                            except Exception as decode_err:
                                logging.debug(f"Frame Decode Error: {decode_err}")
                        else:
                            break
                    
                    # Safety: prevent unbounded buffer growth
                    if len(bytes_buffer) > 1024 * 1024:
                         bytes_buffer = bytes_buffer[-1024:] 
                         
        except Exception as e:
            logging.error(f"MJPEG Reader Connection Error: {e}")


    def _ws_drain_loop(self):
        """
        Drain incoming WS messages (telemetry) and handle automatic reconnection.
        Format: v1,v2,dist,batt,...
        """
        msg_count = 0
        while self.running:
            if not self.ws:
                if self.dummy_mode or self.serial:
                    time.sleep(1.0)
                    continue
                
                # Attempt Reconnection
                logging.info(f"WebSocket disconnected. Attempting to reconnect to {self.ws_url}...")
                try:
                    self.ws = websocket.create_connection(self.ws_url, timeout=5)
                    logging.info("WebSocket Reconnected.")
                except Exception as e:
                    logging.warning(f"WS Reconnection Failed: {e}")
                    time.sleep(2.0)
                    continue

            try:
                message = self.ws.recv()
                if message:
                    try:
                        msg_count += 1
                        
                        # Ignore command echoes e.g., "l:100;r:100;"
                        if message.strip().startswith('l:') or message.strip().startswith('s:'):
                            continue
                        
                        parts = message.strip().split(',')
                        if len(parts) >= 4:
                            with self.telemetry_lock:
                                # Ensure we don't crash on bad float strings
                                try:
                                    self.telemetry['dist'] = parts[2].strip()
                                    self.telemetry['batt'] = parts[3].strip()
                                except Exception:
                                    pass
                        else:
                             logging.warning(f"WS RX Format Error: {message}")
                             
                    except ValueError as e:
                        logging.error(f"WS Parse Error: {e} | Raw: {message}")
                else:
                    # Empty message usually means connection closed
                    logging.warning("Received empty message from WS. Closing socket.")
                    self.ws.close()
                    self.ws = None

            except Exception as e:
                # Timeout or error
                if "timed out" in str(e).lower():
                    pass 
                else:
                    logging.warning(f"WS Read Error: {e}")
                    try: self.ws.close()
                    except Exception: pass
                    self.ws = None
                    time.sleep(1.0)

    def get_latest_frame(self):
        """
        Returns the latest decoded image (numpy array).
        Protocol Change: Previously returned bytes. Now returns cv2 image.
        """
        with self.frame_lock:
            return self.latest_frame

    def get_latest_webcam_frame(self):
        """
        Returns the latest external webcam image (numpy array), or None.
        """
        with self.webcam_lock:
            return self.latest_webcam_frame

    def accelerate_logic(self, action_id):
        """
        Proxy method mapped to direct values via engine.py now handles the updates.
        We preserve this to safely ignore legacy external calls.
        """
        pass

    def send_velocity(self, v, w):
        """
        Sends continuous velocity (linear v, angular w).
        v, w should be normalized [-1.0, 1.0].
        Standard diff drive: L = v - w, R = v + w
        """
        # Get mixed motor commands from controller
        l_val, r_val = self.robot.get_motor_commands(v, w, quirks=self.spikerbot_quirks)

        cmd_str = f"l:{int(l_val)};r:{int(r_val)};"
        
        if self.dummy_mode:
            return f"MOCK: {cmd_str}"
            
        # Send the command
        if self.serial:
            try:
                self.serial.write(cmd_str.encode('utf-8'))
            except Exception as e:
                logging.error(f"Serial Velocity Failed: {e}")
        
        if self.ws:
            try:
                self.ws.send(cmd_str)
            except Exception as e:
                logging.error(f"WebSocket Velocity Failed: {e}")
                
        return cmd_str

    def send_pwm(self, left, right):
        """
        Sends raw PWM integers directly to the robot.
        left, right: integers (e.g. -255 to 255)
        """
        cmd_str = f"l:{int(left)};r:{int(right)};"
        
        if self.dummy_mode:
            return f"MOCK: {cmd_str}"
            
        if self.ws:
            try:
                self.ws.send(cmd_str)
            except Exception as e:
                logging.error(f"WebSocket PWM Failed: {e}")
                
        return cmd_str

    def send_command(self, action_id):
        """
        Maps Action ID to Motor Commands.
        Supports both Discrete IDs (int) and direct strings (for VLA passthrough).
        """
        # 1. Direct Passthrough for String Commands
        if isinstance(action_id, str):
            cmd_str = action_id
            if self.dummy_mode:
                return f"MOCK: {cmd_str}"
        else:
            # 2. Map Action ID to Motor Commands
            
            # Use current speeds (set by accelerate_logic)
            # Wrap speeds through robot controller for final safety checks
            S = self.robot.process_action(self.base_speed)
            T = self.robot.process_action(self.turn_speed)
            cmd_str = "l:0.0;r:0.0;"
            
            if self.spikerbot_quirks:
                # --- SPIKERBOT HARDWARE QUIRKS (LEGACY/INVERTED) ---
                if action_id == 1:   cmd_str = f"l:-{S:.2f};r:{S:.2f};"
                elif action_id == 2: cmd_str = f"l:-{T:.2f};r:-{T:.2f};"
                elif action_id == 3: cmd_str = f"l:{T:.2f};r:{T:.2f};"
                elif action_id == 0: cmd_str = "l:0.0;r:0.0;"
                elif action_id == 4: cmd_str = f"l:{S:.2f};r:-{S:.2f};"
            else:
                # --- STANDARD MAPPING (CARTESO / PLEXUS / OTHER) ---
                if action_id in [1, 2, 3, 4]:
                    # Linear and Angular are tracked in comms, mixing happens in engine.py
                    v_raw = self.manual_v_pwm
                    w_raw = self.manual_w_pwm
                    cmd_str = f"l:{v_raw};r:{w_raw};" # Placeholder, actual mixing in engine
                elif action_id == 0:
                    cmd_str = "l:0.0;r:0.0;"
                else:
                    cmd_str = "l:0.0;r:0.0;"

            if self.dummy_mode:
                return f"MOCK: {cmd_str}"

        # 1. Try WebSocket (SpikerBot/Plexus)
        if self.ws is None:
             logging.warning("Command Failed: WebSocket not connected.")
             return False

        try:
            self.ws.send(cmd_str)
            return cmd_str
        except Exception as e:
            logging.error(f"Command Failed: {e}")
            return False

    def set_led(self, rgb_tuple):
        """
        Set all LEDs to the given RGB tuple (0-255).
        Legacy Protocol: "d:{i},{r},{g},{b};" for i in 0..3
        """
        r, g, b = rgb_tuple
        
        if self.dummy_mode:
             return f"d:ALL,{r},{g},{b};"

        if not self.ws:
            return
            
        color_str = f"{r},{g},{b}"
        # Assuming 4 LEDs as per legacy
        for i in range(4):
            cmd = f"d:{i},{color_str};"
            try:
                self.ws.send(cmd)
            except Exception as e:
                logging.error(f"WS LED Error: {e}")
        
        return f"d:ALL,{r},{g},{b};" # Return summary command string

    def send_sound_command(self, freq):
        """
        Sends sound command. Returns the command string.
        Protocol: "s:hz;"
        """
        cmd = f"s:{freq};"
        
        if self.dummy_mode:
            return cmd

        if not self.ws:
            return None
        
        try:
            self.ws.send(cmd)
            return cmd
        except Exception as e:
            logging.error(f"WS Sound Error: {e}")
            return None

    def verify_handshake(self):
        # WebSocket usually doesn't need explicit application handshake if connection succeeded
        return True if self.ws else False

    def close(self):
        self.running = False
        
        # 1. Stop WebSocket
        if self.ws:
            try: self.ws.send("l:0;r:0;") 
            except Exception: pass
            self.ws.close()
            
        # 2. Join Threads to ensure they stop using resources
        if hasattr(self, 'vid_thread') and self.vid_thread.is_alive():
            self.vid_thread.join(timeout=1.0)
            
        if hasattr(self, 'webcam_thread') and self.webcam_thread.is_alive():
            self.webcam_thread.join(timeout=1.0)
            
        if hasattr(self, 'drain_thread') and self.drain_thread.is_alive():
            self.drain_thread.join(timeout=1.0)

        # 3. Clean up
        self.running = False
        if self.webcam_cap and self.webcam_cap.isOpened():
             self.webcam_cap.release()
