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


import os
import signal
import threading
import time
import asyncio
import logging
from typing import List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from pydantic import BaseModel
import matplotlib
matplotlib.use('Agg')
# Configuration
from config import DATA_DIR, MODELS_DIR, ROBOT_IP

# Services
from backend.services.datasets import DatasetService
from backend.services.websockets import WebSocketController

# Create Router
router = APIRouter()
logger = logging.getLogger("API")

# --- SERVICES ---
# Instantiated on first request or globally? globally is better if stateless.
# But DatasetService needs data root which might change? 
# The current app uses global DATA_DIR except for overrides. 
# We'll instantiate default here.
dataset_service = DatasetService() 

# --- DATA MODELS ---

class CountRequest(BaseModel):
    name: str
    path: Optional[str] = None

class ProcessRequest(BaseModel):
    datasets: List[str]
    root_path: Optional[str] = None

class LatentsRequest(BaseModel):
    filename: str
    root_path: Optional[str] = None

class TitleRequest(BaseModel):
    dataset: str
    root_path: Optional[str] = None

class TrainRequest(BaseModel):
    root_path: Optional[str] = None
    num_epochs: int = 5
    batch_size: int = 64 
    learning_rate: float = 0.0001 
    model_size: str = "small" 
    model_filename: Optional[str] = None # [NEW] Fixes missing attribute error
    vae_model: Optional[str] = None 
    dataset_percent: int = 10 # [NEW]
    tag: str = "red_ball" # [NEW]
    cql_goal_type: str = "her" # [NEW] Wall-seeking toggle
    cql_alpha: float = 0.1 # [NEW]
    selected_datasets: Optional[List[str]] = None # [NEW]
    train_from_scratch: bool = False # [NEW] End-to-end DrQ flag
    vae_beta: float = 0.5 # [NEW]
    image_size: int = 64 # [NEW] LatentSLAM Resolution
    num_layers: int = 4 # [NEW] LatentSLAM Depth
    latent_dim: int = 128 # [NEW] LatentSLAM Representation capacity
    forward_approach: str = "mse" # [NEW] topological model selection
    transition_loss_weight: float = 1.0 # [NEW] LatentSLAM Transition weight
    contrastive_weight: float = 0.0 # [NEW] InfoNCE Weight
    architecture: str = "continuous" # [NEW] VQ-VAE Toggle

class VerifyRequest(BaseModel):
    force: bool = False
    only_cache: bool = False
    model_filename: Optional[str] = None
    dataset: Optional[str] = None 
    root_path: Optional[str] = None 

class VerifyForwardRequest(BaseModel):
    model_filename: str
    vae_filename: str
    approach: str
    root_path: Optional[str] = None 

class ImagesRequest(BaseModel):
    path: str 

# --- HELPER ---
def get_engine(request: Request):
    return request.app.state.engine

# --- ROUTES ---

@router.post("/shutdown")
async def shutdown(request: Request):
    logger.info("Shutdown Request Received.")
    engine = get_engine(request)
    
    def kill_process():
        if engine:
            logger.info("Stopping Engine...")
            engine.stop()
        time.sleep(1) 
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
    return dataset_service.list_datasets(path, fast)

@router.post("/api/dataset_count")
async def get_dataset_count(req: CountRequest):
    count = dataset_service.get_dataset_count(req.name, req.path)
    return {"name": req.name, "count": count}

@router.post("/api/browse")
async def browse_folder():
    """Open a server-side folder browser dialog."""
    def _open_dialog():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw() 
            root.attributes('-topmost', True) 
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
    from backend.training import TrainingPipeline
    data_dir = req.root_path if req.root_path else DATA_DIR
    pipeline = TrainingPipeline(data_dir)
    result = await asyncio.to_thread(pipeline.process_datasets, req.datasets)
    return result

@router.post("/training/visualize_dataset")
async def visualize_dataset(req: TitleRequest):
    data_dir = req.root_path if req.root_path else DATA_DIR
    dataset_path = os.path.join(data_dir, req.dataset)
    
    if not os.path.exists(dataset_path):
        return {"status": "error", "message": "Dataset not found"}
        
    def progress_callback(msg):
        logger.info(f"PCA Progress: {msg}")
    
    def _run():
        from backend.pca_service import PCAService
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
    data_dir = req.root_path if req.root_path else os.path.join(os.getcwd(), "data")
    
    def _run():
        from backend.latents import LatentGenerator
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
    training_stop_event.set()
    return {"status": "stopping"}

@router.post("/training/train_forward")
async def train_forward(request: Request, req: TrainRequest = None):
    if req is None: req = TrainRequest()
    
    data_dir = os.path.join(os.getcwd(), "data")
    if req and req.root_path:
        data_dir = req.root_path
        
    training_stop_event.clear()
    engine = get_engine(request)

    def progress_cb(epoch, loss, kld=0.0, manifold_plot=None, policy_heatmap=None, kl=0.0, recon=0.0):
        if engine:
            if epoch is not None: engine.state_manager.update('training_epoch', epoch)
            if loss is not None: engine.state_manager.update('training_loss', float(loss)) 
            if manifold_plot: engine.state_manager.update('training_manifold_plot', manifold_plot)
            if policy_heatmap: engine.state_manager.update('training_policy_heatmap', policy_heatmap)
            if recon is not None and recon > 0: engine.state_manager.update('training_recon', float(recon))
            if kl is not None and kl > 0: engine.state_manager.update('training_kl', float(kl))

    def _run():
        from backend.training import TrainingPipeline
        pipeline = TrainingPipeline(data_dir)
        
        if req.forward_approach == "latentslam":
            def latentslam_progress_cb(epoch, loss, kl=0.0, recon=0.0, manifold_plot=None):
                progress_cb(epoch, loss, kl=kl, recon=recon, policy_heatmap=manifold_plot)
                
            return pipeline.run_latentslam_pipeline(
                num_epochs=req.num_epochs, 
                stop_event=training_stop_event,
                progress_callback=latentslam_progress_cb,
                batch_size=req.batch_size,
                learning_rate=req.learning_rate,
                model_size=req.model_size,
                selected_datasets=req.selected_datasets,
                model_filename=req.model_filename,
                image_size=req.image_size,
                num_layers=req.num_layers,
                latent_dim=req.latent_dim
            )
        else:
            return pipeline.run_topological_pipeline(
                num_epochs=req.num_epochs, 
                stop_event=training_stop_event,
                progress_callback=progress_cb,
                batch_size=req.batch_size,
                learning_rate=req.learning_rate,
                approach=req.forward_approach,
                selected_datasets=req.selected_datasets,
                model_filename=req.model_filename
            )
        
    try:
        result = await asyncio.to_thread(_run)
        return result
    except Exception as e:
        logger.error(f"Topological Forward Training failed: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/training/train_vae")
async def train_vae(request: Request, req: TrainRequest = None):
    if req is None: req = TrainRequest()
    
    data_dir = os.path.join(os.getcwd(), "data")
    if req and req.root_path:
        data_dir = req.root_path
    
    transitions_path = os.path.join(data_dir, "all_transitions.json")
    if not os.path.exists(transitions_path):
        return {"status": "error", "message": "Missing all_transitions.json. Please process a dataset first."}
        
    training_stop_event.clear()
    engine = get_engine(request)

    def progress_cb(epoch, loss, kld=0.0, manifold_plot=None, **kwargs):
        if engine:
            if epoch is not None: engine.state_manager.update('training_epoch', epoch)
            if loss is not None: engine.state_manager.update('training_loss', float(loss))
            if kld is not None: 
                engine.state_manager.update('training_kld', float(kld))
                engine.state_manager.update('training_kl', float(kld)) # Alias for UI compatibility
            if 'recon' in kwargs and kwargs['recon'] is not None and kwargs['recon'] > 0:
                engine.state_manager.update('training_recon', float(kwargs['recon']))
            if manifold_plot: engine.state_manager.update('training_manifold_plot', manifold_plot)

    def _run():
        from backend.training import TrainingPipeline
        pipeline = TrainingPipeline(data_dir)
        return pipeline.run_vae_pipeline(
            num_epochs=req.num_epochs, 
            stop_event=training_stop_event,
            progress_callback=progress_cb,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            beta=req.vae_beta,
            model_size=req.model_size,
            selected_datasets=req.selected_datasets,
            model_filename=req.model_filename,
            architecture=req.architecture,
            latent_dim=req.latent_dim
        )
        
    try:
        result = await asyncio.to_thread(_run)
        return result
    except Exception as e:
        logger.error(f"VAE Training failed: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/training/train_latentslam")
async def train_latentslam(request: Request, req: TrainRequest = None):
    if req is None: req = TrainRequest()
    
    data_dir = os.path.join(os.getcwd(), "data")
    if req and req.root_path:
        data_dir = req.root_path
        
    training_stop_event.clear()
    engine = get_engine(request)

    def progress_cb(epoch, loss, kl=0.0, recon=0.0, manifold_plot=None):
        if engine:
            engine.state_manager.update('training_epoch', epoch)
            engine.state_manager.update('training_loss', float(loss))
            engine.state_manager.update('training_kl', float(kl))
            engine.state_manager.update('training_kld', float(kl)) # Alias
            if recon > 0: engine.state_manager.update('training_recon', float(recon))
            if manifold_plot:
                engine.state_manager.update('training_policy_heatmap', manifold_plot)

    def _run():
        from backend.training import TrainingPipeline
        pipeline = TrainingPipeline(data_dir)
        return pipeline.run_latentslam_pipeline(
            num_epochs=req.num_epochs, 
            stop_event=training_stop_event,
            progress_callback=progress_cb,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            beta=req.vae_beta,
            transition_loss_weight=req.transition_loss_weight,
            contrastive_weight=req.contrastive_weight,
            architecture=req.architecture,
            model_size=req.model_size,
            selected_datasets=req.selected_datasets,
            model_filename=req.model_filename,
            image_size=req.image_size,
            num_layers=req.num_layers,
            latent_dim=req.latent_dim
        )
        
    try:
        result = await asyncio.to_thread(_run)
        return result
    except Exception as e:
        logger.error(f"LatentSLAM Training failed: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/training/train_dreamer")
async def train_dreamer(request: Request, req: TrainRequest = None):
    if req is None: req = TrainRequest()
    
    data_dir = os.path.join(os.getcwd(), "data")
    if req and req.root_path:
        data_dir = req.root_path
        
    training_stop_event.clear()
    engine = get_engine(request)

    def progress_cb(status_dict):
        if engine:
            with engine.state_lock:
                engine.state['training_epoch'] = status_dict.get('epoch', 0)
                engine.state['training_loss'] = float(status_dict.get('loss', 0))
                engine.state['training_val_loss'] = float(status_dict.get('val_loss', 0))

    def _run():
        from backend.training import TrainingPipeline
        pipeline = TrainingPipeline(data_dir)
        return pipeline.run_dreamer_pipeline(
            num_epochs=req.num_epochs, 
            stop_event=training_stop_event,
            progress_callback=progress_cb,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            tag=req.tag,
            selected_datasets=req.selected_datasets,
            model_filename=req.model_filename
        )
        
    try:
        result = await asyncio.to_thread(_run)
        return result
    except Exception as e:
        logger.error(f"DreamerV3 Training failed: {e}")
        return {"status": "error", "message": str(e)}

@router.get("/training/visualize_policy")
async def visualize_policy(path: str = None, model: str = None):
    data_dir = path if path else os.path.join(os.getcwd(), "data")
    
    if model is not None and model == "null": 
        model = None
        
    if not os.path.exists(data_dir):
         return {"status": "error", "message": f"Data directory not found: {data_dir}"}
    
    def _run():
        import base64
        import glob
        import os
        if model:
            base = os.path.splitext(os.path.basename(model))[0]
            # Try Parity Plot first
            parity_path = os.path.join(data_dir, '..', 'models', f"{base}_parity.png")
            if os.path.exists(parity_path):
                with open(parity_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                    return {"status": "success", "image": encoded}
            
            # Fallback for old confusion matrix
            cm_path = os.path.join(data_dir, f"{base}_confusion.png")
            if os.path.exists(cm_path):
                with open(cm_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                    return {"status": "success", "image": encoded}
                    
        try:
            from backend.training import TrainingPipeline
            pipeline = TrainingPipeline(data_dir)
            if hasattr(pipeline, 'visualize_policy_weights'):
                return pipeline.visualize_policy_weights(model_filename=model)
        except Exception:
            pass
            
        return {"status": "error", "message": "Evaluation plot not found."}
        
    try:
        result = await asyncio.to_thread(_run)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/training/verify_manifold")
async def verify_manifold(request: Request, req: VerifyRequest = None):
    if req is None: req = VerifyRequest()
    
    engine = get_engine(request)
    if not engine or not engine.manifold:
        return {"status": "error", "message": "Engine/Manifold service not available"}

    try:
        def _run():
            # Setup paths
            data_dir = req.root_path if req.root_path else os.path.join(os.getcwd(), "data")
            if req.dataset:
                 data_dir = os.path.join(data_dir, req.dataset)
            
            # Use verify_manifold utility
            from analysis_archive.verify_manifold import run_sanity_check
            img_b64 = run_sanity_check(
                data_path=data_dir, 
                model_path_override=req.model_filename,
                force_recompute=req.force,
                only_return_cache=req.only_cache
            )
            
            if img_b64:
                meta = None
                meta_path = os.path.join(data_dir, req.model_filename.replace(".pth", "_meta.json")) if req.model_filename else None
                if meta_path and os.path.exists(meta_path):
                    import json
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                    except Exception: pass
                
                return {"status": "success", "image": img_b64, "metadata": meta}
            
            return {"status": "skipped", "message": "No cache found or generation failed."}

        result = await asyncio.to_thread(_run)
        return result
    except Exception as e:
        logger.error(f"Manifold verification failed: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/training/verify_forward_model")
async def verify_forward_model(request: Request, req: VerifyForwardRequest):
    data_dir = req.root_path if req.root_path else os.path.join(os.getcwd(), "data")
    
    def _run():
        import base64
        import os
        from analysis_archive.verify_forward_model import generate_forward_dashboard
        
        base = os.path.splitext(os.path.basename(req.model_filename))[0]
        parity_path = os.path.join(data_dir, f"{base}_parity.png")
        if not os.path.exists(parity_path):
             parity_path = os.path.join(data_dir, '..', 'models', f"{base}_parity.png")
        
        fwd_path = os.path.join(data_dir, req.model_filename)
        vae_path = os.path.join(data_dir, req.vae_filename) if req.vae_filename else ""
        if not os.path.exists(vae_path) and hasattr(req, 'vae_filename'):
            import glob
            vaes = glob.glob(os.path.join(data_dir, "*_vae_*.pth"))
            if not vaes: vaes = glob.glob(os.path.join(data_dir, "*.pth")) # fallback
            if vaes: vae_path = max([v for v in vaes if 'forward' not in v and 'cql' not in v], key=os.path.getmtime)
            
        img_b64 = generate_forward_dashboard(
            fwd_path, vae_path, parity_path, data_dir, base, req.approach
        )
        if img_b64:
             meta = None
             meta_path = fwd_path.replace(".pth", "_meta.json")
             if os.path.exists(meta_path):
                 import json
                 try:
                     with open(meta_path, "r") as f:
                         meta = json.load(f)
                 except Exception: pass
                 
             return {"status": "success", "image": img_b64, "metadata": meta}
        return {"status": "error", "message": "Failed to generate forward dashboard"}

    try:
        result = await asyncio.to_thread(_run)
        return result
    except Exception as e:
        logger.error(f"Forward Model verification failed: {e}")
        return {"status": "error", "message": str(e)}

@router.get("/training/files")
async def list_training_files(path: str = None):
    data_dir = path if path else DATA_DIR
    files = []
    
    target_files = ["all_transitions.json", "manifold.png", "manifold.pkl"]
    for f in target_files:
        fp = os.path.join(data_dir, f)
        if os.path.exists(fp):
             size = os.path.getsize(fp) / (1024 * 1024)
             mod_time = os.path.getmtime(fp)
             modified_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
             files.append({
                 "name": f,
                 "size_mb": round(size, 2),
                 "modified": modified_str,
                 "path": f
             })

    # Glob all .pth files in data_dir
    import glob
    pth_files = glob.glob(os.path.join(data_dir, "*.pth"))
    for fp in pth_files:
         fname = os.path.basename(fp)
         size = os.path.getsize(fp) / (1024 * 1024)
         mod_time = os.path.getmtime(fp)
         modified_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
         files.append({
             "name": fname,
             "size_mb": round(size, 2),
             "modified": modified_str,
             "path": fname
         })

    return {"files": files}

@router.get("/api/current_goal_image")
async def get_current_goal_image(request: Request):
    engine = get_engine(request)
    if engine and hasattr(engine, 'planner') and engine.planner and getattr(engine.planner, 'goal_image_path', None):
        if os.path.exists(engine.planner.goal_image_path):
            from fastapi.responses import FileResponse
            return FileResponse(engine.planner.goal_image_path)
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="No goal image available")

@router.get("/api/experience_map")
async def get_experience_map(request: Request):
    engine = get_engine(request)
    if not engine or not hasattr(engine, 'latent_slam_service') or not engine.latent_slam_service:
        return {"nodes": [], "edges": [], "active_node": -1}
    
    return engine.latent_slam_service.get_graph_payload()

@router.get("/api/models")
async def list_models(path: str = None):
    return dataset_service.list_models(path)

@router.post("/api/list_images")
async def list_images_in_folder(req: ImagesRequest):
    images = dataset_service.list_images(req.path)
    return {"images": images}

@router.get("/api/manifold_points")
async def get_manifold_points(request: Request):
    engine = get_engine(request) 
    if not engine or not engine.manifold or not engine.manifold.is_ready:
        return {"points": [], "bounds": None}
    
    return {
        "points": engine.manifold.manifold_points if engine.manifold.manifold_points else [], 
        "bounds": getattr(engine.manifold, 'bounds', None)
    }

@router.get("/api/experience_map")
async def get_experience_map(request: Request):
    engine = get_engine(request)
    if not engine or not hasattr(engine, 'latent_slam_service') or engine.latent_slam_service is None:
        return {"nodes": [], "edges": [], "active_node": -1}
    
    return engine.latent_slam_service.get_graph_payload()

from fastapi.responses import FileResponse, Response

@router.get("/api/manifold_image")
async def get_manifold_image(request: Request):
    engine = get_engine(request) 
    if not engine or not engine.manifold or not engine.manifold.cache_path:
        return Response(status_code=404)
        
    png_path = engine.manifold.cache_path.replace('.pkl', '.png')
    if os.path.exists(png_path):
        return FileResponse(png_path)
    return Response(status_code=404)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    engine = websocket.app.state.engine
    controller = WebSocketController(engine)
    await controller.handle_connection(websocket)

@router.post("/training/extract_telemetry")
async def extract_telemetry(req: ProcessRequest):
    data_dir = req.root_path if req.root_path else os.path.join(os.getcwd(), "data")
    
    def _run():
        import base64
        import io
        from matplotlib import pyplot as plt
        from scripts.extract_telemetry import TelemetryExtractor
        from backend.training import TrainingPipeline
        
        pipeline = TrainingPipeline(data_dir)
        expanded_names = pipeline._expand_datasets(req.datasets)
        dirs = [os.path.join(data_dir, d) for d in expanded_names]
        
        print(f"[API EXTRACT_TELEMETRY] Evaluated Dirs to scan: {dirs}")
        
        if not dirs:
            return {"status": "error", "message": "No valid physical dataset directories found for prefix."}
            
        extractor = TelemetryExtractor(dirs)
        df_telemetry = extractor.process_all()
        
        if df_telemetry.empty:
            return {"status": "error", "message": "No telemetry found"}
            
        out_path = os.path.join(data_dir, "master_telemetry.csv")
        df_telemetry.to_csv(out_path, index=False)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Allocentric Telemetry Map (XY + Yaw)", fontsize=14, pad=15)
        
        cx = df_telemetry['cx'].values
        cy = df_telemetry['cy'].values
        dx = df_telemetry['dx'].values
        dy = df_telemetry['dy'].values
        
        ax.scatter(cx, cy, c='blue', alpha=0.3, s=10)
        ax.quiver(cx, cy, dx, dy, color='red', scale=40, width=0.003, alpha=0.5)
        
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xlabel("X Position (Pixels)")
        ax.set_ylabel("Y Position (Pixels)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if getattr(extractor, 'arena_bounds', None):
            import matplotlib.patches as patches
            for dataset_path, b in extractor.arena_bounds.items():
                rect = patches.Rectangle((b['min_x'], b['min_y']), b['max_x']-b['min_x'], b['max_y']-b['min_y'], 
                                         linewidth=2, edgecolor='green', facecolor='none', linestyle=':', alpha=0.3)
                ax.add_patch(rect)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return {"status": "success", "image": img_base64, "path": out_path}
        
    try:
        result = await asyncio.to_thread(_run)
        return result
    except Exception as e:
        logger.error(f"Telemetry Extraction failed: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/training/reset_workspace")
async def reset_workspace():
    from config import MODELS_DIR, DATA_DIR
    deleted_files = []
    
    def try_remove(fpath):
        try:
            if os.path.exists(fpath):
                os.remove(fpath)
                deleted_files.append(fpath)
                logger.info(f"Deleted: {fpath}")
        except Exception as e:
            logger.error(f"Failed to delete {fpath}: {e}")

    targets = [
        os.path.join(MODELS_DIR, "goals.npy"),
        os.path.join(DATA_DIR, "manifold_summary.png")
    ]
    
    import glob
    patterns = [
        "*cql_policy.pth",
        "*tiny_vae_final.pth",
        "*vae_encoder.pth",
        "*spatial_encoder.pth"
    ]
    
    for base_dir in [MODELS_DIR, DATA_DIR]:
        if not os.path.exists(base_dir): continue
        for pat in patterns:
            files = glob.glob(os.path.join(base_dir, "**", pat), recursive=True)
            for f in files:
                try_remove(f)
                
    for t in targets:
        try_remove(t)
        
    return {"status": "success", "deleted": deleted_files}

@router.post("/api/test_board")
async def test_board(request: Request):
    """Temporary endpoint for testing hardware components remotely."""
    data = await request.json()
    action = data.get("action")
    
    if action not in ["buzzer", "tally"]:
        return {"status": "error", "message": "Invalid action"}
        
    def _run_test():
        import websocket
        try:
            ws = websocket.create_connection(f"ws://{ROBOT_IP}/ws", timeout=2)
            if action == "buzzer":
                ws.send("s:1000;")
                time.sleep(0.5)
                ws.send("s:0;")
            elif action == "tally":
                ws.send("t:1;")
                time.sleep(0.1)
            ws.close()
            return True, None
        except Exception as e:
            return False, str(e)
            
    success, error_msg = await asyncio.to_thread(_run_test)
    if success:
         return {"status": "success"}
    else:
         logger.error(f"Failed to test board ({action}): {error_msg}")
         return {"status": "error", "message": error_msg}
