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
import sys
import glob
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Determine paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
GOALS_NPY_PATH = os.path.join(MODELS_DIR, "goals.npy")

# Find the VAE associated goals dynamically
import glob
goal_dirs = glob.glob(os.path.join(PROJECT_ROOT, "data", "*_goals"))
if goal_dirs:
    # Use most recently modified goals dir
    goal_dirs.sort(key=os.path.getmtime, reverse=True)
    GOALS_DIR = goal_dirs[0]
else:
    GOALS_DIR = os.path.join(PROJECT_ROOT, "data", "goals")

class GoalFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Goal Filter GUI")
        
        # Data
        self.image_paths = []
        self.latents = None
        self.selected_indices = set()
        self.image_labels = []
        
        self.load_data()
        self.build_ui()

    def load_data(self):
        # 1. Load Images (Sorted alphabetically, just like encode_goals.py)
        if not os.path.exists(GOALS_DIR):
            messagebox.showerror("Error", f"Goals directory not found:\n{GOALS_DIR}")
            sys.exit(1)
            
        self.image_paths = sorted(glob.glob(os.path.join(GOALS_DIR, "*.jpg")) + 
                                  glob.glob(os.path.join(GOALS_DIR, "*.png")))
        
        if not self.image_paths:
            messagebox.showinfo("Info", f"No goal images found in {GOALS_DIR}.")
            sys.exit(0)

        # 2. Load Latents
        if not os.path.exists(GOALS_NPY_PATH):
            messagebox.showerror("Error", f"goals.npy not found at:\n{GOALS_NPY_PATH}")
            sys.exit(1)
            
        try:
            self.latents = np.load(GOALS_NPY_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load goals.npy:\n{e}")
            sys.exit(1)
            
        if len(self.image_paths) != len(self.latents):
            messagebox.showerror(
                "Mismatch", 
                f"Number of images ({len(self.image_paths)}) does not match "
                f"number of latents in goals.npy ({len(self.latents)}).\n\n"
                "Please regenerate goals.npy using encode_goals.py first."
            )
            sys.exit(1)

        # Select all by default
        self.selected_indices = set(range(len(self.image_paths)))

    def build_ui(self):
        # Top Frame for Buttons
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top_frame, text="Select All", command=self.select_all).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Deselect All", command=self.deselect_all).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="OK (Save & Delete Unselected)", bg='#4CAF50', fg='white', 
                  font=('Arial', 10, 'bold'), command=self.save_and_exit).pack(side=tk.RIGHT, padx=10)
        
        lbl_info = tk.Label(top_frame, text=f"Loaded {len(self.image_paths)} goals.")
        lbl_info.pack(side=tk.LEFT, padx=20)

        # Canvas & Scrollbar for Image Grid
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_frame)
        scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        grid_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=grid_frame, anchor="nw")
        
        grid_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Render Images
        cols = 5
        for i, path in enumerate(self.image_paths):
            try:
                img = Image.open(path)
                img.thumbnail((160, 120)) # Resize for display
                photo = ImageTk.PhotoImage(img)
                
                # Create a frame for the border effect
                frame = tk.Frame(grid_frame, bd=4, cursor="hand2")
                frame.grid(row=i // cols, column=i % cols, padx=5, pady=5)
                
                lbl = tk.Label(frame, image=photo)
                lbl.image = photo # Keep reference
                lbl.pack()
                
                # Ensure the filename is visible too
                name_lbl = tk.Label(frame, text=os.path.basename(path), font=("Arial", 8))
                name_lbl.pack()

                self.image_labels.append((frame, (lbl, name_lbl)))
                
                # Bind clicks
                def make_click_handler(idx):
                    return lambda event: self.toggle_selection(idx)
                    
                lbl.bind("<Button-1>", make_click_handler(i))
                name_lbl.bind("<Button-1>", make_click_handler(i))
                frame.bind("<Button-1>", make_click_handler(i))
                
            except Exception as e:
                print(f"Error loading {path}: {e}")
                
        self.update_visuals()

    def toggle_selection(self, idx):
        if idx in self.selected_indices:
            self.selected_indices.remove(idx)
        else:
            self.selected_indices.add(idx)
        self.update_visuals()

    def select_all(self):
        self.selected_indices = set(range(len(self.image_paths)))
        self.update_visuals()

    def deselect_all(self):
        self.selected_indices.clear()
        self.update_visuals()
        
    def update_visuals(self):
        for i, (frame, _) in enumerate(self.image_labels):
            if i in self.selected_indices:
                frame.configure(bg='green')
            else:
                frame.configure(bg='red')

    def save_and_exit(self):
        if not self.selected_indices:
            if not messagebox.askyesno("Warning", "You have 0 goals selected.\nThis will delete ALL goals!\nAre you sure?"):
                return

        # 1. Filter Latents
        selected_indices_sorted = sorted(list(self.selected_indices))
        filtered_latents = self.latents[selected_indices_sorted] if self.latents is not None else []
        
        # 2. Delete unselected images
        deleted_count = 0
        for i, path in enumerate(self.image_paths):
            if i not in self.selected_indices:
                try:
                    os.remove(path)
                    deleted_count += 1
                except OSError as e:
                    print(f"Failed to delete {path}: {e}")
                    
        # 3. Save new goals.npy
        try:
            if len(filtered_latents) > 0:
                np.save(GOALS_NPY_PATH, filtered_latents)
            else:
                if os.path.exists(GOALS_NPY_PATH):
                    os.remove(GOALS_NPY_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save {GOALS_NPY_PATH}: {e}")
            return
            
        messagebox.showinfo("Success", f"Saved {len(filtered_latents)} latents to goals.npy.\nDeleted {deleted_count} unselected images.")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    
    # Try to center window
    window_width = 900
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width / 2)
    center_y = int(screen_height/2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    app = GoalFilterGUI(root)
    root.mainloop()
