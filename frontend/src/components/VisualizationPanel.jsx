import React, { useEffect, useRef, useState } from 'react';

const VisualizationPanel = ({ keypoints, manifoldCoord }) => {
    const [manifoldPoints, setManifoldPoints] = useState([]);

    // Canvas Refs
    const manifoldCanvasRef = useRef(null);
    const keypointsCanvasRef = useRef(null);

    // Fetch Manifold Background Points
    useEffect(() => {
        fetch('/api/manifold_points')
            .then(res => res.json())
            .then(data => {
                if (data.points) {
                    setManifoldPoints(data.points);
                }
            })
            .catch(err => console.error("Failed to fetch manifold points:", err));
    }, []);

    // Draw Manifold
    useEffect(() => {
        const canvas = manifoldCanvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        // Clear
        ctx.clearRect(0, 0, w, h);

        // Background
        ctx.fillStyle = '#f3f4f6'; // Gray-100
        ctx.fillRect(0, 0, w, h);

        // Helper to map coordinates
        // We need to know range. Let's assume -1 to 1 or -2 to 2? 
        // PCA is arbitrary. 
        // Let's compute bounds from points if available, else standard.
        // For simplicity, let's auto-scale based on points.

        let minX = -1, maxX = 1, minY = -1, maxY = 1;
        if (manifoldPoints.length > 0) {
            const xs = manifoldPoints.map(p => p[0]);
            const ys = manifoldPoints.map(p => p[1]);
            minX = Math.min(...xs);
            maxX = Math.max(...xs);
            minY = Math.min(...ys);
            maxY = Math.max(...ys);

            // Add padding
            const padX = (maxX - minX) * 0.1;
            const padY = (maxY - minY) * 0.1;
            minX -= padX; maxX += padX;
            minY -= padY; maxY += padY;
        }

        const mapX = (x) => ((x - minX) / (maxX - minX)) * w;
        const mapY = (y) => h - ((y - minY) / (maxY - minY)) * h; // Flip Y

        // Draw Background Points
        ctx.fillStyle = '#9ca3af'; // Gray-400
        for (let p of manifoldPoints) {
            ctx.beginPath();
            ctx.arc(mapX(p[0]), mapY(p[1]), 2, 0, 2 * Math.PI);
            ctx.fill();
        }

        // Draw Current State
        if (manifoldCoord) {
            ctx.fillStyle = '#ef4444'; // Red-500
            ctx.beginPath();
            ctx.arc(mapX(manifoldCoord[0]), mapY(manifoldCoord[1]), 6, 0, 2 * Math.PI);
            ctx.fill();

            // Ring
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

    }, [manifoldPoints, manifoldCoord]);

    // Draw Keypoints
    useEffect(() => {
        const canvas = keypointsCanvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#1f2937'; // Gray-800
        ctx.fillRect(0, 0, w, h);

        // Grid lines
        ctx.strokeStyle = '#374151';
        ctx.lineWidth = 1;

        // Center lines
        ctx.beginPath();
        ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
        ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
        ctx.stroke();

        if (!keypoints || keypoints.length === 0) return;

        // Keypoints are interlaced [x1, y1, x2, y2...] or flat?
        // backend latents.py generates flattened (Batch, 64).
        // spatial_model.py: coords = stack([x,y], dim=2).view(B, -1) -> x1, y1, x2, y2...

        ctx.fillStyle = '#22c55e'; // Green-500

        for (let i = 0; i < keypoints.length; i += 2) {
            const x = keypoints[i];
            const y = keypoints[i + 1];

            // Map -1..1 to Screen
            const sx = ((x + 1) / 2) * w;
            const sy = ((y + 1) / 2) * h;

            ctx.beginPath();
            ctx.arc(sx, sy, 3, 0, 2 * Math.PI);
            ctx.fill();
        }
    }, [keypoints]);

    return (
        <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
            <h3 className="text-sm font-bold text-gray-400 uppercase mb-3">Visualization</h3>

            <div className="grid grid-cols-2 gap-4">
                {/* Keypoints */}
                <div className="flex flex-col gap-1">
                    <span className="text-xs text-gray-400 text-center">Spatial Config (32)</span>
                    <canvas
                        ref={keypointsCanvasRef}
                        width={200}
                        height={150}
                        className="w-full bg-gray-800 rounded border border-gray-600 shadow-inner"
                    />
                </div>

                {/* Manifold */}
                <div className="flex flex-col gap-1">
                    <span className="text-xs text-gray-400 text-center">Latent Manifold (PCA)</span>
                    <canvas
                        ref={manifoldCanvasRef}
                        width={200}
                        height={150}
                        className="w-full bg-gray-100 rounded border border-gray-300 shadow-inner"
                    />
                </div>
            </div>

            <div className="mt-2 text-xs text-gray-400 text-center">
                {manifoldCoord ? `PCA: [${manifoldCoord[0].toFixed(2)}, ${manifoldCoord[1].toFixed(2)}]` : "Select a trained model to see manifold"}
            </div>
        </div>
    );
};

export default VisualizationPanel;
