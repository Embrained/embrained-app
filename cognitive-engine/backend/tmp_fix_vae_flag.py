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

import re

file_path = "c:/Users/chris/Embrained/software_suite/backend/train_cql.py"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Update train signature
pattern_sig = re.compile(
r"def train\(data_root, num_epochs=20, stop_event=None, progress_callback=None, vae_model_filename=None, batch_size=128, learning_rate=1e-5, model_size='large', dataset_percent=10, \s*\n?goal_type='her', selected_datasets=None, model_filename=None\):"
)
sig_replacement = r"def train(data_root, num_epochs=20, stop_event=None, progress_callback=None, vae_model_filename=None, batch_size=128, learning_rate=1e-5, model_size='large', dataset_percent=10, goal_type='her', selected_datasets=None, model_filename=None, train_from_scratch=False):"
text = pattern_sig.sub(sig_replacement, text)

# Update VAE auto-discovery fallback
pattern_vae = re.compile(
r"          if not vae_path:\n              # Fallback to auto-discovery if explicit name failed or wasn't provided\n              parent_name = os\.path\.basename\(os\.path\.normpath\(data_root\)\)\n              if not parent_name or parent_name == 'data' or parent_name == '\.':\n                  prefix = \"\"\n              else:\n                  prefix = f\"\{parent_name\}_\"\n\n              possible_new = os\.path\.join\(MODELS_DIR, f\"\{parent_name\}-vae\.pth\"\)\n              if os\.path\.exists\(possible_new\):\n                   vae_path = possible_new\n              else:\n                   raise FileNotFoundError\(f\"Could not find VAE model: \{possible_new\}\. Please train the VAE for this dataset first\.\"\)"
)
vae_replacement = """          if not vae_path and not train_from_scratch:
              # Fallback to auto-discovery if explicit name failed or wasn't provided
              parent_name = os.path.basename(os.path.normpath(data_root))
              if not parent_name or parent_name == 'data' or parent_name == '.':
                  prefix = ""
              else:
                  prefix = f"{parent_name}_"

              possible_new = os.path.join(MODELS_DIR, f"{parent_name}-vae.pth")
              if os.path.exists(possible_new):
                   vae_path = possible_new
              else:
                   raise FileNotFoundError(f"Could not find VAE model: {possible_new}. Please train the VAE for this dataset first. Or use --train_from_scratch")"""
text = pattern_vae.sub(vae_replacement, text)

# Update argparse
pattern_arg = re.compile(
r"    parser\.add_argument\(\"--model_size\", type=str, default='large', help=\"Model Size\"\)\n    args = parser\.parse_args\(\)"
)
arg_replacement = """    parser.add_argument("--model_size", type=str, default='large', help="Model Size")
    parser.add_argument("--train_from_scratch", action="store_true", help="Initialize CNN from scratch, skipping VAE checks")
    args = parser.parse_args()"""
text = pattern_arg.sub(arg_replacement, text)

# Update train call at bottom
pattern_call = re.compile(
r"        dataset_percent=args\.dataset_pct,\n        model_size=args\.model_size\n    \)"
)
call_replacement = """        dataset_percent=args.dataset_pct,
        model_size=args.model_size,
        train_from_scratch=args.train_from_scratch
    )"""
text = pattern_call.sub(call_replacement, text)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)
print("Patch successfully added the --train_from_scratch flag.")
