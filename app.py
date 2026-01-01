import sys
import os
import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Ensure we can import backend if needed, though engine is imported via path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.engine import CognitiveEngine
from backend.routes import router as api_router

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("API")

# Global Engine Instance shared via app.state
# (Will be initialized in lifespan)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Server Starting...")
    
    is_plexus = '--plexus' in sys.argv
    is_spikerbot = '--spikerbot' in sys.argv
    is_webcam = '--webcam' in sys.argv
    is_simulation = '--simulation' in sys.argv
    
    # Check strict embodiment flags
    flag_count = sum([is_plexus, is_spikerbot, is_webcam, is_simulation])
    if flag_count > 1:
        logger.critical("Startup Aborted: Too many embodiment flags. Choose only one: [--plexus, --spikerbot, --webcam, --simulation]")
        sys.exit(1)
        
    # Configuration based on flag
    target_ip = None
    target_port = 81
    dry_run = False
    use_cam = False
    
    if is_plexus:
        target_ip = "10.0.0.26"
        target_port = 80
        dry_run = False
        use_cam = False # Plexus handles stream
    elif is_spikerbot:
        target_ip = None # Defaults to config.ROBOT_IP (usually Little SpikerBot AP)
        target_port = 81
        dry_run = False
        use_cam = False
    elif is_webcam:
        target_ip = None
        target_port = 81
        dry_run = True # Webcam mode implies no robot connection, just local cam
        use_cam = True
    elif is_simulation:
        target_ip = None
        target_port = 81
        dry_run = False
        use_cam = False # Simulator handles stream
    else:
        # Default: Dummy Mode (No flags)
        logger.info("No embodiment flag provided. Starting in DUMMY MODE (Bouncing Ball).")
        target_ip = None
        target_port = 81
        dry_run = True
        use_cam = False

    engine = None
    try:
        logger.info(f"Connecting to embodiment: {'Plexus' if is_plexus else 'SpikerBot' if is_spikerbot else 'Webcam' if is_webcam else 'Simulation'}")
        
        # Determine strict mode params
        engine = CognitiveEngine(dry_run=dry_run, use_webcam=use_cam, simulation=is_simulation, robot_ip=target_ip, stream_port=target_port)
        engine.start()
        
        # Share engine with routes
        app.state.engine = engine

    except Exception as e:
        logger.critical(f"Engine Failed to Start: {e}")
        # In strict mode, if connection fails, we just fail. We don't fallback to dry_run implicitly unless requested.
        sys.exit(1)
    
    yield
    
    # --- Shutdown ---
    logger.info("Server Shutting Down...")
    if engine:
        engine.stop()

app = FastAPI(lifespan=lifespan)

# Include Routes
app.include_router(api_router)

# Static Files (Frontend)
# We expect 'frontend/dist' to exist after build.
frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "dist")

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
else:
    @app.get("/")
    def read_root():
        return {"message": "Frontend not found. Please build React app in /frontend"}

if __name__ == "__main__":
    import uvicorn
    # Allow passing args to this script
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", action="store_true", help="Use local webcam instead of robot stream")
    parser.add_argument("--simulation", action="store_true", help="Run in PyBullet simulation mode")
    parser.add_argument("--plexus", action="store_true", help="Connect to Plexus Robot")
    parser.add_argument("--spikerbot", action="store_true", help="Connect to SpikerBot")
    parser.add_argument("--ip", type=str, default="10.0.0.11", help="Robot IP Address")
    
    # We parse args just to make sure they are valid, but the app uses sys.argv in check
    # or we can pass logic. But sticking to sys.argv pattern in lifespan is safer if uvicorn
    # reloads.
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
