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

import sys
import os

# Prevent 20-30s timeout delays when connected to the offline Plexus robot WiFi
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "offline"

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure we can import backend if needed, though engine is imported via path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# from backend.engine import CognitiveEngine
from backend.routes import router as api_router

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("API")

# Global Engine Instance shared via app.state
# (Will be initialized in lifespan)

def get_configured_engine():
    """
    Parses sys.argv to determine embodiment mode and initializes the CognitiveEngine.
    Returns:
        CognitiveEngine: The initialized engine instance.
    """
    is_plexus = '--plexus' in sys.argv
    is_spikerbot = '--spikerbot' in sys.argv
    is_simulation = '--simulation' in sys.argv
    
    rec_prefix = "markov"
    if "--suffix" in sys.argv:
        idx = sys.argv.index("--suffix")
        if idx + 1 < len(sys.argv):
            rec_prefix = sys.argv[idx + 1]
            
    # Check strict embodiment flags
    flag_count = sum([is_plexus, is_spikerbot, is_simulation])
    if flag_count > 1:
        logger.critical("Startup Aborted: Too many embodiment flags. Choose only one.")
        sys.exit(1)
        
    # Configuration defaults
    target_ip = None
    target_port = 81
    dry_run = False
    use_cam = '--webcam' in sys.argv

    if is_plexus:
        from config import ROBOT_IP
        target_ip = ROBOT_IP
        target_port = 81
        dry_run = False
    elif is_spikerbot:
        target_ip = None 
        target_port = 81
        dry_run = False
    elif is_simulation:
        target_ip = None
        target_port = 81
        dry_run = False
    else:
        # Default: Dummy Mode
        target_ip = None
        target_port = 81
        dry_run = True

    try:
        msg = "Plexus" if is_plexus else ("SpikerBot" if is_spikerbot else "Dummy")
        logger.debug(f"Connecting to embodiment: {msg}")
        
        # [LAZY LOAD]
        if '--continuous' in sys.argv:
            from backend.engine_continuous import ContinuousCognitiveEngine as CognitiveEngine
        else:
            from backend.engine import CognitiveEngine
        
        engine = CognitiveEngine(dry_run=dry_run, robot_ip=target_ip, stream_port=target_port, rec_prefix=rec_prefix, use_webcam=use_cam, is_simulation=is_simulation)
        
        # Set Explicit Embodiment Name
        if is_plexus: engine.state["embodiment"] = "PLEXUS"
        elif is_spikerbot: engine.state["embodiment"] = "SPIKERBOT"
        elif is_simulation: engine.state["embodiment"] = "SIMULATION"
        else: engine.state["embodiment"] = "DUMMY"
        
        return engine

    except Exception as e:
        logger.critical(f"Engine Failed to Start: {e}")
        sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.debug("Welcome to Embrained! Preparing startup sequence...")
    engine = get_configured_engine()
    engine.start()
    
    logger.debug("Core sub-systems initialized.")
    logger.debug("API Services ready.")
    app.state.engine = engine
    yield
    # --- Shutdown ---
    logger.debug("Server Shutting Down...")
    if engine:
        engine.stop()

app = FastAPI(lifespan=lifespan)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In dev, allow all. Could restrict to http://localhost:5173
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routes
app.include_router(api_router)

# Static Files (Frontend)
# We expect 'frontend/dist' to exist after build.
frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "dist")

if os.path.exists(frontend_path):
    # Serve assets directly if they exist
    assets_path = os.path.join(frontend_path, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Prevent the API namespace from being shadowed
        if full_path.startswith("api/") or full_path.startswith("training/") or full_path.startswith("record/") or full_path == "shutdown":
            return {"detail": "Not Found"}
            
        # 1. Attempt to serve an actual file (e.g., vite.svg, family_sketch.png)
        file_path = os.path.join(frontend_path, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
            
        # 2. Fallback to index.html for client-side routing
        index_path = os.path.join(frontend_path, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(
                index_path,
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
            )
            
        return {"detail": "Not Found"}
else:
    @app.get("/")
    def read_root():
        return {"message": "Frontend not found. Please build React app in /frontend"}

if __name__ == "__main__":
    import uvicorn
    # Allow passing args to this script
    parser = argparse.ArgumentParser()
    parser.add_argument("--plexus", action="store_true", help="Connect to Plexus Robot")
    parser.add_argument("--spikerbot", action="store_true", help="Connect to SpikerBot")
    parser.add_argument("--simulation", action="store_true", help="Launch robust headless Raycast Virtual Environment")
    parser.add_argument("--webcam", action="store_true", help="Record external webcam if used with Plexus")
    parser.add_argument("--continuous", action="store_true", help="Launch the Action Chunking (Continuous) engine instead of discrete Markov.")
    parser.add_argument("--suffix", type=str, default="markov", help="Prefix for new recording folders")
    parser.add_argument("--ip", type=str, default="192.168.4.1", help="Robot IP Address")
    
    parser.add_argument("--port", type=int, default=8080, help="API Server Port")
    
    # We parse args just to make sure they are valid, but the app uses sys.argv in check
    # or we can pass logic. But sticking to sys.argv pattern in lifespan is safer if uvicorn
    # reloads.
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port, access_log=False, ws_ping_interval=None, ws_ping_timeout=None)
