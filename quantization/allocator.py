
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class GreedyBitAllocator:
    """
    Allocates bit-widths to layers based on their sensitivity scores to meet a 
    global memory/bit-budget.
    """
    
    def __init__(self, default_bitwidth: int = 4):
        self.default_bitwidth = default_bitwidth
        
    def allocate(self, sensitivity_map: Dict[str, float], target_avg_bits: float) -> Dict[str, int]:
        """
        Allocates bits (2, 4, 8) to layers.
        Strategy:
        1. Start all at lowest precision (2-bit).
        2. Iteratively upgrade the most sensitive layers until budget runs out or max precision reached.
        """
        logger.info(f"Allocating bits with target execution average: {target_avg_bits}")
        
        layers = list(sensitivity_map.keys())
        n_layers = len(layers)
        
        # Sort layers by sensitivity (descending)
        sorted_layers = sorted(sensitivity_map.items(), key=lambda x: x[1], reverse=True)
        
        # Allowable bit widths
        # We assume 2, 4, 8 bits support
        available_bits = [2, 4, 8]
        
        # Initial Allocation: All 2 bits (or 4 if conservative)
        # Let's start with minimum
        config = {k: 2 for k in layers}
        
        current_total_bits = 2 * n_layers
        
        # Iterative Upgrade
        # We loop through sorted layers multiple times? 
        # Or just one pass? 
        # "Greedy": Give 8 bits to top X, 4 bits to middle Y, 2 bits to bottom Z.
        
        # More robust approach:
        # Calculate budget in total bits
        total_bit_budget = target_avg_bits * n_layers
        
        # Upgrade loop
        changed = True
        while changed and current_total_bits < total_bit_budget:
            changed = False
            for layer_name, score in sorted_layers:
                current_bits = config[layer_name]
                if current_bits < 8:
                    # Can we afford upgrade?
                    next_step = 4 if current_bits == 2 else 8
                    cost_increase = next_step - current_bits
                    
                    if current_total_bits + cost_increase <= total_bit_budget:
                        config[layer_name] = next_step
                        current_total_bits += cost_increase
                        changed = True
                        if current_total_bits >= total_bit_budget:
                            break
            
        avg = current_total_bits / n_layers
        logger.info(f"Allocation Complete. Average Bits: {avg:.2f}")
        return config
