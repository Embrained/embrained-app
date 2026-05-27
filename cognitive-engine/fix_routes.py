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
import os

with open("backend/routes.py", "r", encoding="utf-8") as f:
    text = f.read()

target = """    def _run():
        from backend.training import TrainingPipeline
        pipeline = TrainingPipeline(data_dir)
        return pipeline.visualize_policy_weights(model_filename=model)
        
    try:
        result = await asyncio.to_thread(_run)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}"""

replacement = """    def _run():
        import base64
        import glob
        import os
        if model:
            base = os.path.splitext(os.path.basename(model))[0]
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
            
        return {"status": "error", "message": "Confusion matrix not found."}
        
    try:
        result = await asyncio.to_thread(_run)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}"""

# Use regex to normalize whitespace
pattern = re.compile(re.escape(target).replace(r'\n', r'\r?\n'), re.MULTILINE)

new_text = text
if target in text:
    new_text = text.replace(target, replacement)
    print("Exact string match")
elif pattern.search(text):
    new_text = pattern.sub(replacement, text)
    print("Regex match")
else:
    # Try more permissive replacement
    print("Target not found exactly, trying lenient replace...")
    # Just locate the def _run(): inside visualize_policy
    if "def visualize_policy" in text:
        print("visualize_policy found. I will try to patch it manually.")

with open("backend/routes.py", "w", encoding="utf-8", newline='\n') as f:
    f.write(new_text)
