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


import threading
import collections
import asyncio
import logging
import cv2
import time

class VLAController:
    def __init__(self, policy_server=None):
        self.policy_server = policy_server
        self.action_queue = collections.deque()
        self.is_fetching_chunk = False
        self.fetch_lock = threading.Lock()
        self.queue_lock = threading.Lock()
        
    def get_action(self, img):
        """
        Management of Client-Side Action Queue.
        Returns a string command "l:L;r:R;" or int 3 (STOP).
        """
        # 1. Trigger Fetch if needed (Threshold ~25 actions remaining (0.5s))
        trigger_threshold = 25
        with self.queue_lock:
             current_q_len = len(self.action_queue)
        
        if current_q_len < trigger_threshold and not self.is_fetching_chunk:
             self._trigger_fetch(img)
        
        # 2. Return Action
        if current_q_len > 0:
             with self.queue_lock:
                 # Pop from LEFT (FIFO)
                 chunk_action = self.action_queue.popleft() # Tensor or numpy (2,)
                 
                 # Conversion logic (Throttle, Turn) -> (Left, Right)
                 # Assume action is [throttle, turn]
                 # throttle in [-1, 1], turn in [-1, 1]
                 throttle = float(chunk_action[0])
                 turn = float(chunk_action[1])
                 
                 # Differential drive mixing
                 left = throttle + turn
                 right = throttle - turn
                 
                 # Clamp
                 left = max(-1.0, min(1.0, left))
                 right = max(-1.0, min(1.0, right))
                 
                 # "l:%.2f;r:%.2f;"
                 return f"l:{left:.2f};r:{right:.2f};"
        
        return 3 # Stop if starved

    def clear_queue(self):
        with self.queue_lock:
            self.action_queue.clear()

    def _trigger_fetch(self, img):
        # Start background thread
        if self.policy_server and img is not None:
             threading.Thread(target=self._run_fetch, args=(img,), daemon=True).start()

    def _run_fetch(self, img):
         # Lock to prevent double fetch
         if self.is_fetching_chunk: return
         
         with self.fetch_lock:
             if self.is_fetching_chunk: return # Double check
             self.is_fetching_chunk = True
             
         try:
             # Prepare Image
             success, encoded_img = cv2.imencode('.jpg', img)
             if not success: return
             
             image_bytes = encoded_img.tobytes()
             
             # Call Async Server
             # Note: creating a loop every time is heavy but robust for threads in this context
             actions = asyncio.run(self.policy_server.predict(image_bytes))
             
             # Append to queue
             with self.queue_lock:
                 # "Temporal Ensembling or Replacement"
                 # Instruction says "start with replacement: discard remaining actions"
                 self.action_queue.clear()
                 self.action_queue.extend(actions)
                 
         except Exception as e:
             logging.error(f"VLA Fetch Error: {e}")
         finally:
             with self.fetch_lock:
                 self.is_fetching_chunk = False
