import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.train_cql import run_cql_train

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
VAE_PATH = os.path.join(DATA_ROOT, 'vqvae_512c_32d_20260427_153402.pth')

print("Starting Fast CQL Evaluation...")
run_cql_train(
    data_root=DATA_ROOT,
    num_epochs=3,  # Just 3 epochs to see the classification report
    vae_model_filename=os.path.basename(VAE_PATH),
    batch_size=128,
    learning_rate=1e-4, 
    alpha=0.2, 
    model_size='large',
    dataset_percent=100,
    goal_type='discrete_exact',
    model_filename='test_discrete_cql.pth',
    train_from_scratch=True
)
