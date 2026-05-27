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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def generate_ir_scatter_from_telemetry():
    df = pd.read_csv("master_telemetry.csv")
    print(f"Loaded master telemetry with {len(df)} frames.")
    
    # Clean anomalies and clip mathematical collision asymptotes (dist < 40) where Euclidean denominator -> 0
    df_clean = df[(df['ir'] > 0) & (df['ir'] < 4000) & (df['dist_px'] > 40)]
    print(f"Valid IR frames: {len(df_clean)}")
    
    df_clean['inv_dist'] = 1.0 / df_clean['dist_px']
    corr_inv = df_clean['ir'].corr(df_clean['inv_dist'])
    print(f"Absolute Corrected True-Heading Correlation computed: {corr_inv:.4f}")
    
    try:
        def ir_model(x, a, b, c):
            return a / (np.maximum(x, 1.0) + np.abs(c)+1) + b
        popt, _ = curve_fit(ir_model, df_clean['dist_px'], df_clean['ir'], p0=[1000, 0, 10], maxfev=100000)
    except:
        print("Curve fit failed. Plotting pure scatter.")
        popt = None
        
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    if popt is not None:
        x_range = np.linspace(df_clean['dist_px'].min(), df_clean['dist_px'].max(), 100)
        plt.plot(x_range, ir_model(x_range, *popt), 'r-', linewidth=4, label='Inverse Sensor Model Fit')
        
    plt.scatter(df_clean['dist_px'], df_clean['ir'], alpha=0.1, s=4, c='b')
    plt.xlabel('Exact Straight-Line Distance to Geometric Boundary (pixels)')
    plt.ylabel('Physical Analog IR Sensor Reading')
    plt.title(f'Analytical Dist vs Analog IR Match\nDist/IR Correlation: {corr_inv:.3f}')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if popt is not None:
        df_clean['ir_est'] = ir_model(df_clean['dist_px'], *popt)
        df_sample = df_clean.sample(n=min(1000, len(df_clean)))
        plt.scatter(df_sample['ir'], df_sample['ir_est'], alpha=0.5, c='g', s=15)
        min_val = min(df_sample['ir'].min(), df_sample['ir_est'].min())
        max_val = max(df_sample['ir'].max(), df_sample['ir_est'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.xlabel('Recorded Physical Analog IR')
        plt.ylabel('Virtual Wall Distance Analytical Estimator')
        plt.title('Sub-Sample Estimation Performance')
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig('ir_correlation_true_scatter.png', dpi=150)
    print("Saved ir_correlation_true_scatter.png")

if __name__ == "__main__":
    generate_ir_scatter_from_telemetry()
