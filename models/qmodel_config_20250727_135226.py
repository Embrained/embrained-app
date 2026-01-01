from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.models.encoders import VectorEncoderFactory

# This file is auto-generated to ensure consistent model architecture for inference.
# Generated at: 20250727_135226
LATENT_DIM = 32
HIDDEN_UNITS = [512, 512, 512]
ACTION_SIZE = 5

def get_q_network_for_inference(device_str: str):
    # [IMPLEMENTED] Added Layer Normalization to match training configuration
    encoder_factory = VectorEncoderFactory(hidden_units=HIDDEN_UNITS, use_layer_norm=True)
    q_net_config = DiscreteCQLConfig(encoder_factory=encoder_factory)
    q_net = q_net_config.create(device=device_str)
    observation_shape = (LATENT_DIM * 2,)
    q_net.create_impl(observation_shape, ACTION_SIZE)
    return q_net