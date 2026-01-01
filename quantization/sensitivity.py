
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import logging
import copy

logger = logging.getLogger(__name__)

class ActionSensitivityProfiler:
    """
    Measures the sensitivity of each network channel/layer to quantization noise
    by observing the divergence in the final action distribution.
    """
    
    def __init__(self, metric: str = "kl_div"):
        self.metric = metric
        
    def measure_sensitivity(self, model: nn.Module, calibration_data: List[Dict]) -> Dict[str, float]:
        """
        Iterates through model layers, applies fake quantization, and measures
        impact on action logits.
        """
        logger.info("Starting Action Sensitivity Profiling...")
        
        # 1. Baseline Run (FP16)
        baseline_logits = []
        with torch.inference_mode():
            for sample in calibration_data:
                # Assuming sample['input'] is a tensor or dict compatible with model
                if isinstance(sample['input'], dict):
                     logits = model(**sample['input']).logits
                else:
                     logits = model(sample['input']).logits
                
                baseline_logits.append(logits.cpu()) # Store on CPU to save VRAM
        
        sensitivity_map = {}
        
        # 2. Sensitivity Analysis Loop
        # Target Linear layers (common for Transformers)
        target_layers = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
        
        for layer_name, layer in target_layers.items():
            original_weight = layer.weight.data.clone()
            
            # Apply Fake Quantization (INT4 Simulation)
            # Symmetric quantization around 0
            w = layer.weight.data
            max_val = w.abs().max()
            scale = max_val / 7.0 
            # Quantize and Dequantize
            w_quant = (w / scale).round().clamp(-7, 7)
            w_fake = w_quant * scale
            
            # Inject fake weights
            layer.weight.data = w_fake
            
            divergence_sum = 0.0
            
            try:
                with torch.inference_mode():
                    for i, sample in enumerate(calibration_data):
                        if isinstance(sample['input'], dict):
                             perturbed_logits = model(**sample['input']).logits
                        else:
                             perturbed_logits = model(sample['input']).logits
                        
                        perturbed_logits = perturbed_logits.cpu()
                        baseline_logit = baseline_logits[i]

                        # Compute Divergence
                        if self.metric == "kl_div":
                            # KL(P || Q) - Baseline is target P, Perturbed is Q
                            p = F.softmax(baseline_logit, dim=-1)
                            log_q = F.log_softmax(perturbed_logits, dim=-1)
                            
                            div = F.kl_div(log_q, p, reduction='batchmean')
                            divergence_sum += div.item()
                        elif self.metric == "mse":
                            div = F.mse_loss(perturbed_logits, baseline_logit)
                            divergence_sum += div.item()
            except Exception as e:
                logger.error(f"Error profiling layer {layer_name}: {e}")
                
            finally:
                # Restore original weights
                layer.weight.data = original_weight
                
            avg_sensitivity = divergence_sum / len(calibration_data)
            sensitivity_map[layer_name] = avg_sensitivity
            logger.info(f"Layer {layer_name}: Sensitivity = {avg_sensitivity:.6e}")
            
        return sensitivity_map
