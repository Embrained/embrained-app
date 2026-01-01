
import os
import signal
import threading
import time
import asyncio
import logging
from typing import List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from pydantic import BaseModel
from backend.training import TrainingPipeline
from backend.latents import LatentGenerator
from backend.pca_service import PCAService

# Create Router
router = APIRouter()
logger = logging.getLogger("API")

# --- DATA MODELS ---

class CountRequest(BaseModel):
    name: str
    path: str = None

class ProcessRequest(BaseModel):
    datasets: List[str]
    root_path: str = None

class LatentsRequest(BaseModel):
    filename: str
    root_path: str = None

class TitleRequest(BaseModel):
    dataset: str
    root_path: str = None

class TrainRequest(BaseModel):
    root_path: str = None
    num_epochs: int = 5

# --- HELPER ---
def get_engine(request: Request):
    return request.app.state.engine

# --- ROUTES ---

@router.post("/shutdown")
async def shutdown(request: Request):
    logger.info("Shutdown Request Received.")
    engine = get_engine(request)
    
    # Schedule Process Termination
    def kill_process():
        if engine:
            logger.info("Stopping Engine...")
            engine.stop()
        
        time.sleep(1) # Give time for the HTTP response to be sent
        logger.info("Killing Process...")
        os.kill(os.getpid(), signal.SIGTERM) 
        
    threading.Thread(target=kill_process).start()
    
    return {"status": "Shutting down..."}

@router.post("/record/start")
async def start_recording(request: Request):
    engine = get_engine(request)
    if engine:
        engine.start_recording()
    return {"status": "started"}

@router.post("/record/stop")
async def stop_recording(request: Request):
    engine = get_engine(request)
    if engine:
        engine.stop_recording()
    return {"status": "stopped"}

@router.get("/datasets")
async def list_datasets(path: str = None, fast: bool = False):
    """List all dataset directories in the specified path or ./data default."""
    if path and os.path.isdir(path):
        data_dir = path
    else:
        # Assuming run from root where data/ is likely
        # Better to make this relative to the file location if needed, but app.py is usually root
        data_dir = os.path.join(os.getcwd(), "data") 

    if not os.path.exists(data_dir):
        return {"datasets": [], "root": data_dir}
    
    datasets = []
    try:
        for d in os.listdir(data_dir):
            full_path = os.path.join(data_dir, d)
            if os.path.isdir(full_path) and d != "logs":
                count = -1
                if not fast:
                    try:
                        # Check images dir first (old format)
                        img_path = os.path.join(full_path, "images")
                        if os.path.exists(img_path) and os.path.isdir(img_path):
                            # Simple count of files
                            count = len([f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))])
                        else:
                            # Flat structure (New/Legacy)
                            img_path = full_path
                            count = len([f for f in os.listdir(img_path) 
                                         if os.path.isfile(os.path.join(img_path, f)) and f.lower().endswith('.jpg')])
                    except:
                        count = 0
                
                datasets.append({"name": d, "count": count})
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")

    # Sort by name
    datasets.sort(key=lambda x: x["name"])
    return {"datasets": datasets, "root": data_dir}

@router.post("/api/dataset_count")
async def get_dataset_count(req: CountRequest):
    """Get image count for a specific dataset."""
    data_dir = req.path if req.path else os.path.join(os.getcwd(), "data")
    full_path = os.path.join(data_dir, req.name)
    
    count = 0
    if os.path.exists(full_path) and os.path.isdir(full_path):
        try:
             # Check images dir first (old format)
            img_path = os.path.join(full_path, "images")
            if os.path.exists(img_path) and os.path.isdir(img_path):
                count = len([f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))])
            else:
                img_path = full_path
                count = len([f for f in os.listdir(img_path) 
                             if os.path.isfile(os.path.join(img_path, f)) and f.lower().endswith('.jpg')])
        except Exception as e:
            logger.error(f"Error counting {req.name}: {e}")
            
    return {"name": req.name, "count": count}

@router.post("/api/browse")
async def browse_folder():
    """Open a server-side folder browser dialog."""
    def _open_dialog():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw() # Hide main window
            root.attributes('-topmost', True) # Bring to front
            folder_selected = filedialog.askdirectory()
            root.destroy()
            return folder_selected
        except Exception as e:
            logger.error(f"Browse failed: {e}")
            return ""

    path = await asyncio.to_thread(_open_dialog)
    return {"path": path}

@router.post("/training/process")
async def process_datasets(req: ProcessRequest):
    """Trigger dataset processing pipeline."""
    data_dir = req.root_path if req.root_path else os.path.join(os.getcwd(), "data")
    pipeline = TrainingPipeline(data_dir)
    result = await asyncio.to_thread(pipeline.process_datasets, req.datasets)
    return result

@router.post("/training/visualize_dataset")
async def visualize_dataset(req: TitleRequest):
    """Generate PCA plot for a dataset."""
    data_dir = req.root_path if req.root_path else os.path.join(os.getcwd(), "data")
    dataset_path = os.path.join(data_dir, req.dataset)
    
    if not os.path.exists(dataset_path):
        return {"status": "error", "message": "Dataset not found"}
        
    def progress_callback(msg):
        logger.info(f"PCA Progress: {msg}")
    
    def _run():
        service = PCAService()
        return service.generate_plot(req.dataset, dataset_path, progress_cb=progress_callback)
        
    try:
        img_base64 = await asyncio.to_thread(_run)
        return {"status": "success", "image": img_base64}
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/training/get_latents")
async def get_latents(req: LatentsRequest):
    """Generate latents for the selected trajectory file."""
    data_dir = req.root_path if req.root_path else os.path.join(os.getcwd(), "data")
    
    # Run in thread to avoid blocking
    def _run():
        gen = LatentGenerator(data_dir)
        return gen.generate_latents_from_file(req.filename)
        
    try:
        output_file = await asyncio.to_thread(_run)
        return {"status": "success", "file": output_file}
    except Exception as e:
        logger.error(f"Latent generation failed: {e}")
        return {"status": "error", "message": str(e)}

training_stop_event = threading.Event()

@router.post("/training/stop")
async def stop_training():
    """Signal training to stop."""
    training_stop_event.set()
    return {"status": "stopping"}

@router.post("/training/train_cql")
async def train_cql(request: Request, req: TrainRequest = None):
    """Trigger the full CQL training pipeline."""
    # req might be null if no body sent, handle default
    if req is None: req = TrainRequest()
    
    data_dir = os.path.join(os.getcwd(), "data")
    if req and req.root_path:
        data_dir = req.root_path
        
    # Reset stop event
    training_stop_event.clear()

    engine = get_engine(request)

    def progress_cb(epoch, loss):
        # Update Engine State for WS Broadcast
        if engine:
            with engine.state_lock:
                engine.state['training_epoch'] = epoch
                engine.state['training_loss'] = float(loss) # JSON safety

    def _run():
        pipeline = TrainingPipeline(data_dir)
        return pipeline.run_cql_pipeline(
            num_epochs=req.num_epochs, 
            stop_event=training_stop_event,
            progress_callback=progress_cb
        )
        
    try:
        result = await asyncio.to_thread(_run)
        return result
    except Exception as e:
        logger.error(f"CQL Training failed: {e}")
        return {"status": "error", "message": str(e)}

@router.get("/training/files")
async def list_training_files(path: str = None):
    """List available trajectory files in data directory."""
    data_dir = path if path else os.path.join(os.getcwd(), "data")
    files = []
    
    target_files = ["all_transitions.json", "episodes.json", "cql_model.pth"]
    
    if os.path.exists(data_dir):
        for f in target_files:
            file_p = os.path.join(data_dir, f)
            if os.path.exists(file_p):
                # Get size/mod time? Just name is probably enough for now.
                size = os.path.getsize(file_p) / (1024 * 1024) # MB
                files.append({
                    "name": f,
                    "size_mb": round(size, 2),
                    "path": f # Dont expose full path, just name usually
                })
                
    return {"files": files}

@router.get("/api/models")
async def list_models():
    """List available trained models (.pth) in data and models folders."""
    models = []
    
    # Check data dir
    data_dir = os.path.join(os.getcwd(), "data")
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith(".pth"):
                models.append(f)
                
    # Check models dir
    models_dir = os.path.join(os.getcwd(), "models")
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
             # Avoid duplicates if mapped same (though usually distinct)
             if f.endswith(".pth") and f not in models:
                 models.append(f)
                 
    return {"models": models}

@router.get("/api/manifold_points")
async def get_manifold_points(request: Request):
    """Returns the background manifold points for visualization."""
    engine = get_engine(request)
    if engine and engine.manifold and engine.manifold.is_ready:
        return {"points": engine.manifold.manifold_points}
    return {"points": []}

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client Connected")
    
    # Access engine via app state
    engine = websocket.app.state.engine
    
    async def receive_loop():
        try:
            while True:
                data = await websocket.receive_json()
                if engine:
                    cmd_type = data.get("type")
                    payload = data.get("payload")
                    
                    if cmd_type == "SET_MODE":
                        engine.set_mode(payload)
                    elif cmd_type in ["MOVE", "LED", "SOUND", "SET_CONTROLLER"]:
                        engine.handle_command(cmd_type, payload)
                    elif cmd_type == "RECORD_START":
                         engine.start_recording()
                    elif cmd_type == "RECORD_STOP":
                         engine.stop_recording()
                        
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WS Rx Error: {e}")

    async def send_loop():
        try:
            while True:
                if engine:
                    # Get snapshot of state
                    with engine.state_lock:
                        data = engine.state.copy()
                    
                    # Optimization: Only send if changed? 
                    # For video, it always changes.
                    await websocket.send_json(data)
                
                # Stream rate (30fps)
                await asyncio.sleep(0.033)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WS Tx Error: {e}")

    # Run both loops
    try:
        receive_task = asyncio.create_task(receive_loop())
        send_task = asyncio.create_task(send_loop())
        
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
        logger.info("Client Disconnected")
