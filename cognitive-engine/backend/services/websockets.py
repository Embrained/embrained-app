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


import asyncio
import logging
from fastapi import WebSocket, WebSocketDisconnect
from backend.utils import sanitize_for_json

logger = logging.getLogger("WebSocketController")

class WebSocketController:
    def __init__(self, engine):
        self.engine = engine

    async def handle_connection(self, websocket: WebSocket):
        await websocket.accept()
        logger.debug("Client Connected")
        
        # Run both loops
        try:
            receive_task = asyncio.create_task(self._receive_loop(websocket))
            send_task = asyncio.create_task(self._send_loop(websocket))
            
            # Wait for either to finish (likely disconnect)
            done, pending = await asyncio.wait(
                [receive_task, send_task], 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in pending:
                task.cancel()
                
        except Exception as e:
            logger.error(f"WS Error: {e}")
        finally:
            logger.debug("Client Disconnected")

    async def _receive_loop(self, websocket: WebSocket):
        try:
            while True:
                data = await websocket.receive_json()
                if self.engine:
                    cmd_type = data.get("type")
                    payload = data.get("payload")
                    logger.info(f"WS RECEIVED: {cmd_type} with payload: {payload}")
                    
                    if cmd_type == "SET_MODE":
                        self.engine.set_mode(payload)
                    elif cmd_type == "RECORD_START":
                         self.engine.start_recording()
                    elif cmd_type == "RECORD_STOP":
                         self.engine.stop_recording()
                    else:
                        # Delegate other commands to engine's dispatcher
                        # Note: Engine now has handle_command that delegates to dispatcher
                        self.engine.handle_command(cmd_type, payload)
                        
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WS Rx Error: {e}")

    async def _send_loop(self, websocket: WebSocket):
        try:
            while True:
                if self.engine:
                    # Get snapshot of state using the new property or manager
                    # Engine.state is a property that returns the dict
                    with self.engine.state_lock:
                        data = self.engine.state.copy()
                        # Consume large image payloads so they only transmit once
                        if data.get('training_policy_heatmap'):
                            self.engine.state['training_policy_heatmap'] = None
                        if data.get('training_manifold_plot'):
                            self.engine.state['training_manifold_plot'] = None
                    
                    safe_data = sanitize_for_json(data)
                    await websocket.send_json(safe_data)
                
                # Stream rate (10fps to match engine tick rate)
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            pass
        except RuntimeError as e:
            if "Unexpected ASGI message" in str(e):
                pass # Normal during sudden client disconnects
            else:
                logger.error(f"WS Tx Error (Runtime): {e}")
        except Exception as e:
            logger.error(f"WS Tx Error: {e}")
