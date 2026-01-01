
import logging
import time
import threading
import cv2
import requests
import numpy as np
from contextlib import closing
import websocket
import socket
from config import (
    WS_URL, STREAM_URL, ROBOT_IP,
    CMD_FWD_VAL, CMD_REV_VAL, LEFT_MOTOR_INVERT,
    BASE_SPEED, TURN_SPEED, ACTION_NAMES
)

class NervousSystem:
    def __init__(self, dummy_mode=False, use_webcam=False, robot_ip=None, stream_port=81):
        self.ws = None
        self.cap = None
        self.running = True
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.dummy_mode = dummy_mode
        self.use_webcam = use_webcam
        
        # Telemetry Storage
        self.telemetry = {'dist': '0', 'batt': '0'}
        self.telemetry_lock = threading.Lock()


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
            self.video_source = 0 if self.use_webcam else None
        
        # Camera State
        self.camera_valid = False
        self.mock_video = False
        
        if not self.dummy_mode:
            # 1. Pre-check / Wake-up Connection (Socket) - Strict like teleop.py
            # SKIP PING for Plexus (Port 80) to avoid occupying the single socket of the ESP32
            if self.stream_port != 80:
                logging.info(f"Pinging Robot at {self.robot_ip}:{self.stream_port} ...")
                try:
                    with socket.create_connection((self.robot_ip, self.stream_port), timeout=5):
                        pass
                    logging.info("Robot is reachable.")
                except (socket.timeout, socket.error) as e:
                    # Raise error to trigger engine reconnection logic
                    raise ConnectionError(f"Robot host unreachable (Socket Check): {e}")
            else:
                logging.info(f"Skipping Socket Ping for {self.robot_ip}:{self.stream_port} (Single-Client Mode)")

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

            # 3. Connect Camera (HTTP Stream)
            logging.info(f"Stream Configuration: {self.video_source}")
            # Moved cv2.VideoCapture to _video_loop to prevent blocking startup
            
            # 4. Start Threads
            self.vid_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.vid_thread.start()
            
            if self.ws:
                self.drain_thread = threading.Thread(target=self._ws_drain_loop, daemon=True)
                self.drain_thread.start()
        else:
            logging.info("NervousSystem initialized in DUMMY MODE.")
            
            if self.use_webcam:
                logging.info("Initializing Local Webcam for Dry Run...")
                # Search for valid camera
                self.video_source = self._find_valid_camera()
                
                if self.video_source is not None:
                    logging.info(f"Local Webcam Found at Index {self.video_source}.")
                    self.cap = cv2.VideoCapture(self.video_source)
                    self.camera_valid = True
                else:
                    logging.warning("No verified Webcam found. Defaulting to Index 0 and attempting to connect in loop.")
                    self.video_source = 0
                    self.mock_video = False # Do not force mock, let loop retry
                    self.camera_valid = False
            else:
                logging.info("No Camera Source (Webcam Disabled). Defaulting to Mock Video.")
                self.camera_valid = True # Mock is valid
                self.mock_video = True

            # Start Video Loop anyway (it handles mock generation)
            self.vid_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.vid_thread.start()

    def _find_valid_camera(self):
        """Search for the first available camera index."""
        print("Searching for available cameras...")
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    return i
        return None

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
                              # Fallback to Mock if local cam fails repeatedly handling?
                              # For now, just retry slowly, or better: switch to mock after N failures.
                              time.sleep(1.0)
                              continue
                          else:
                               logging.info(f"Webcam {self.video_source} Opened.")

                      else:
                          # Handle Network Stream (String URL)
                          self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
                          if not self.cap.isOpened():
                               logging.warning("Stream Open Failed. Switching to Requests-based Fallback...")
                               # If CV2 fails, we'll loop in the fallback method
                               self._fetch_mjpeg_frames()
                               # If that returns, it means stream ended or failed
                               if not self.running: break 
                               time.sleep(1.0)
                               continue
                          else:
                               logging.info("Stream Opened Successfully (CV2).")
            
            try:
                if self.mock_video:
                    # Generate Mock Frame (Bouncing Box)
                    import numpy as np
                    frame = np.zeros((120, 160, 3), dtype=np.uint8)
                    # Simple animation
                    cx += dx
                    cy += dy
                    if cx <= 10 or cx >= 150: dx *= -1
                    if cy <= 10 or cy >= 110: dy *= -1
                    cv2.rectangle(frame, (cx-10, cy-10), (cx+10, cy+10), (0, 255, 0), -1)
                    cv2.putText(frame, "MOCK CAMERA", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    with self.frame_lock:
                        self.latest_frame = frame
                    time.sleep(0.033) # ~30fps
                    continue

                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        with self.frame_lock:
                            self.latest_frame = frame
                        
                        frame_count += 1
                        if frame_count % 30 == 0:
                            logging.info(f"Video Loop Alive (CV2): {frame_count} frames captured. Stats: {frame.shape}")
                    else:
                         # Stream dropped, try to reconnect
                        logging.warning(f"Stream dropped (ret=False). Reconnecting to {self.video_source}...")
                        self.cap.release()
                        time.sleep(2.0) # wait longer for system to clean up
                else:
                    # Try to open
                    if self.video_source is not None:
                         # Re-verify if camera still exists before spamming open
                         # Or just backoff longer
                         time.sleep(2.0)
                         self.cap = cv2.VideoCapture(self.video_source)
                    else:
                         time.sleep(1.0)

            except Exception as e:
                 logging.error(f"Video Loop Error: {e}")
                 # Critical Failure Handling
                 # if we loop too fast with errors, we switch to mock
                 if "Invalid URL" in str(e): # Stop invalid URL fallback loops quickly
                      logging.error("Critical Stream Error: Switching to Mock Video.")
                      self.mock_video = True
                 time.sleep(1.0)

    def _fetch_mjpeg_frames(self):
        """Fallback MJPEG reader using requests for when CV2 fails."""
        logging.info("Starting Requests-based MJPEG Stream Reader...")
        frame_count = 0
        try:
            with requests.get(self.video_source, stream=True, timeout=5) as r:
                if r.status_code != 200:
                    logging.error(f"Fallback Stream Failed: Status {r.status_code}")
                    return

                bytes_buffer = bytes()
                for chunk in r.iter_content(chunk_size=1024):
                    if not self.running: break
                    bytes_buffer += chunk
                    
                    # Typical MJPEG boundary search
                    a = bytes_buffer.find(b'\xff\xd8') # JPEG Start
                    b = bytes_buffer.find(b'\xff\xd9') # JPEG End
                    
                    if a != -1 and b != -1:
                        jpg = bytes_buffer[a:b+2]
                        bytes_buffer = bytes_buffer[b+2:]
                        
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self.frame_lock:
                                self.latest_frame = frame
                            
                            frame_count += 1
                            if frame_count % 30 == 0:
                                logging.info(f"Video Loop Alive (Fallback): {frame_count} frames.")
        except Exception as e:
            logging.error(f"Fallback Reader Crashed: {e}")


    def _ws_drain_loop(self):
        """
        Drain incoming WS messages (timeouts/pings) AND parse telemetry.
        Format: v1,v2,dist,batt,...
        """
        msg_count = 0
        while self.running and self.ws:
            try:
                message = self.ws.recv()
                if message:
                    try:
                        msg_count += 1
                        # Debug Log: Print every 10th message
                        if msg_count % 10 == 0:
                            logging.info(f"WS RX ({msg_count}): {message}") 
                        
                        parts = message.strip().split(',')
                        if len(parts) >= 4:
                            with self.telemetry_lock:
                                self.telemetry['dist'] = parts[2].strip()
                                self.telemetry['batt'] = parts[3].strip()
                        else:
                             logging.warning(f"WS RX Format Error: {message}")
                             
                    except ValueError as e:
                        logging.error(f"WS Parse Error: {e} | Raw: {message}")
            except Exception as e:
                # Timeout or error, just loop
                if "timed out" in str(e):
                    # Common timeout, ignore or log sparingly
                    pass 
                else:
                    logging.warning(f"WS Read Error: {e}")
                pass

    def get_latest_frame(self):
        """
        Returns the latest decoded image (numpy array).
        Protocol Change: Previously returned bytes. Now returns cv2 image.
        """
        with self.frame_lock:
            return self.latest_frame

    def send_command(self, action_id):
        """
        Maps Action ID to Motor Commands.
        
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !!! SPIKERBOT PROTOTYPE 2025 HARDWARE PATCH                   !!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Hardware wiring is non-standard. We manually map logic here to 
        match the physical reality observed in teleop.py.
        
        Phys FWD  (0) <- 'left' logic     (l:-S, r:S)
        Phys LEFT (1) <- 'backward' logic (l:-S, r:-S)
        Phys RIGHT(2) <- 'forward' logic  (l:S, r:S)
        Phys STOP (3) <- 'stop' logic     (l:0, r:0)
        Phys BACK (4) <- 'right' logic    (l:S, r:-S)
        """
        S = BASE_SPEED
        T = TURN_SPEED
        
        # Map Action ID to legacy motor command string based on hardware patch docstring
        if action_id == 0:   # FORWARD (Phys) <- Left logic
            cmd_str = f"l:-{S};r:{S};"
        elif action_id == 1: # LEFT (Phys) <- Backward logic
            cmd_str = f"l:-{T};r:-{T};"
        elif action_id == 2: # RIGHT (Phys) <- Forward logic
            cmd_str = f"l:{T};r:{T};"
        elif action_id == 3: # STOP
            cmd_str = "l:0;r:0;"
        elif action_id == 4: # BACKWARD (Phys) <- Right logic
            cmd_str = f"l:{S};r:-{S};"
        else:
            cmd_str = "l:0;r:0;"

        if self.dummy_mode:
            # Mock success in dummy mode
            return f"MOCK: {cmd_str}"

        # Send the command
        # If WS is down but we are in Video-Only mode (Plexus), ignore and pretend success
        # Send the command
        if self.ws is None:
             logging.warning("Command Failed: WebSocket not connected.")
             return False

        try:
            self.ws.send(cmd_str)
            return True
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
            except: pass
            self.ws.close()
            
        # 2. Join Threads to ensure they stop using resources
        if hasattr(self, 'vid_thread') and self.vid_thread.is_alive():
            self.vid_thread.join(timeout=1.0)
            
        if hasattr(self, 'drain_thread') and self.drain_thread.is_alive():
            self.drain_thread.join(timeout=1.0)

        # 3. Release Camera
        if self.cap:
            self.cap.release()
            logging.info("Camera Released.")
