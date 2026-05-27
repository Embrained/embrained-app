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
import json
import websockets
import logging

logging.basicConfig(level=logging.INFO)

async def test_speed_update():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Wait for initial state
        msg = await websocket.recv()
        state = json.loads(msg)
        logging.info(f"Initial base_speed: {state.get('base_speed')}")

        # Send speed update
        new_speed = 0.5
        logging.info(f"Sending SET_SPEED: {new_speed}")
        await websocket.send(json.dumps({
            "type": "SET_SPEED",
            "payload": new_speed
        }))

        # Wait for state update (give it a few cycles)
        for _ in range(10):
            msg = await websocket.recv()
            state = json.loads(msg)
            if state.get('base_speed') == new_speed:
                logging.info(f"Success! base_speed updated to: {state.get('base_speed')}")
                return

        logging.error("Failed to update base_speed in time.")

if __name__ == "__main__":
    try:
        asyncio.run(test_speed_update())
    except Exception as e:
        logging.error(f"Test failed: {e}")
