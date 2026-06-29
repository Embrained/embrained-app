/*
 * Embrained - Neural Navigation Software Suite
 * Copyright (C) 2026 Embrained
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

import React, { useEffect, useRef, useState } from 'react';

/* Client-side Manifold Rendering */
const VisualizationPanel = ({ keypoints, manifoldCoord, goalCoords, goalIdx, matchCoord, activeCve, bounds: initialBounds, latentDist, latentThreshold, visualizationMode = 'default', onPointClick }) => {
    const canvasRef = useRef(null);
    const mappingRef = useRef(null);
    const [points, setPoints] = useState([]);
    const [dims, setDims] = useState({ w: 0, h: 0 });

    const stateRef = useRef({
        points: [],
        manifoldCoord: null,
        goalCoords: [],
        goalIdx: 0,
        matchCoord: null,
        activeCve: "N/A",
        dims: { w: 0, h: 0 },
        initialBounds: null,
        latentDist: 0,
        latentThreshold: 0,
        visualizationMode: 'default'
    });

    useEffect(() => {
        stateRef.current = { points, manifoldCoord, goalCoords, goalIdx, matchCoord, activeCve, dims, initialBounds, latentDist, latentThreshold, visualizationMode };
    }, [points, manifoldCoord, goalCoords, goalIdx, matchCoord, activeCve, dims, initialBounds, latentDist, latentThreshold, visualizationMode]);

    // 1. Fetch Manifold Points with Polling
    useEffect(() => {
        if (!activeCve || activeCve === "N/A") {
            setPoints([]);
            return;
        }

        setPoints([]); // Clear points when parameters change

        let isMounted = true;
        const fetchPoints = () => {
            fetch('/api/manifold_points')
                .then(r => r.json())
                .then(data => {
                    if (isMounted && data.points) {
                        setPoints(data.points);
                    }
                })
                .catch(err => console.error("Failed to fetch manifold points:", err));
        };

        // Fetch immediately on mount or activeCve/thresholdSpace change
        fetchPoints();

        // Also set up a slow poll just in case points are generated after initial load
        const interval = setInterval(() => {
            setPoints(prev => {
                if (prev.length === 0) {
                    fetchPoints();
                }
                return prev;
            });
        }, 2000);

        return () => {
            isMounted = false;
            clearInterval(interval);
        };
    }, [activeCve]);

    // 2. Resize Observer
    useEffect(() => {
        const resizeObserver = new ResizeObserver((entries) => {
            const canvas = canvasRef.current;
            if (!canvas) return;

            // Adjust internal resolution to match display size
            const { width, height } = entries[0].contentRect;
            canvas.width = width;
            canvas.height = height;
            setDims({ w: width, h: height });
        });

        if (canvasRef.current) resizeObserver.observe(canvasRef.current);
        return () => resizeObserver.disconnect();
    }, []);

    // 3. Main Drawing Loop (Continuous Animation)
    useEffect(() => {
        let animationFrameId;

        const render = () => {
            const canvas = canvasRef.current;
            if (!canvas) {
                animationFrameId = requestAnimationFrame(render);
                return;
            }
            const ctx = canvas.getContext('2d');
            const state = stateRef.current;
            const { w, h } = state.dims;

            if (w === 0 || h === 0) {
                animationFrameId = requestAnimationFrame(render);
                return;
            }

            // --- Helper: Dynamic Bounds ---
            let allX = [];
            let allY = [];

            if (state.points.length > 0) {
                state.points.forEach(p => { allX.push(p[0]); allY.push(p[1]); });
            }

            if (state.manifoldCoord && Array.isArray(state.manifoldCoord)) {
                allX.push(state.manifoldCoord[0]);
                allY.push(state.manifoldCoord[1]);
            }

            if (state.goalCoords && state.goalCoords.length > 0) {
                state.goalCoords.forEach(g => {
                    if (g) { allX.push(g[0]); allY.push(g[1]); }
                });
            }

            let minX = -1, maxX = 1, minY = -1, maxY = 1;

            if (allX.length > 1) {
                minX = Math.min(...allX);
                maxX = Math.max(...allX);
                minY = Math.min(...allY);
                maxY = Math.max(...allY);

                const padX = (maxX - minX) * 0.15;
                const padY = (maxY - minY) * 0.15;
                minX -= padX; maxX += padX;
                minY -= padY; maxY += padY;
            } else if (state.initialBounds) {
                [minX, maxX, minY, maxY] = state.initialBounds;
            }

            const xRange = Math.max(maxX - minX, 0.0001);
            const yRange = Math.max(maxY - minY, 0.0001);

            mappingRef.current = { minX, minY, xRange, yRange, w, h };

            const mapX = (x) => ((x - minX) / xRange) * w;
            const mapY = (y) => h - ((y - minY) / yRange) * h;

            // --- Drawing ---

            // A. Clear
            ctx.clearRect(0, 0, w, h);

            // B. Crisp Light Background
            const grd = ctx.createLinearGradient(0, 0, 0, h);
            grd.addColorStop(0, '#ffffff'); // white
            grd.addColorStop(1, '#f8fafc'); // slate-50
            ctx.fillStyle = grd;
            ctx.fillRect(0, 0, w, h);

            // C. Sharp Black/Grey Coordinate Grid
            ctx.strokeStyle = 'rgba(15, 23, 42, 0.08)'; // Very faint slate-900
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let x = 0; x <= w; x += 40) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
            for (let y = 0; y <= h; y += 40) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
            ctx.stroke();

            const time = performance.now();

            // D. Manifold Cloud
            if (state.points.length > 0) {
                const defaultColor = 'rgba(100, 116, 139, 0.5)'; // Slate 500
                const actionColors = {
                    0: 'rgba(100, 116, 139, 0.4)', // Slate (Stop/Unknown)
                    1: 'rgba(34, 197, 94, 0.8)',   // Green (Fwd)
                    2: 'rgba(239, 68, 68, 0.8)',   // Red (Rev)
                    3: 'rgba(59, 130, 246, 0.8)',  // Blue (Left)
                    4: 'rgba(168, 85, 247, 0.8)',  // Purple (Right)
                };

                state.points.forEach(p => {
                    const px = mapX(p[0]);
                    const py = mapY(p[1]);
                    
                    let fillStyle = defaultColor;

                    if (state.visualizationMode === 'luminance' && p.length >= 3) {
                        const lum = p[2] || 0.5;
                        if (lum < 0.35) {
                            // Dark wall targets glow bright emerald 
                            fillStyle = `rgba(16, 185, 129, ${Math.min(1.0, 1.2 - lum*2)})`;
                        } else {
                            // Bright walls fade into background logic
                            fillStyle = `rgba(148, 163, 184, ${Math.max(0.1, 0.8 - lum)})`; 
                        }
                    } else if (state.visualizationMode === 'actions' && p.length >= 4) {
                        const action = p[3] || 0;
                        fillStyle = actionColors[action] || defaultColor;
                    }

                    ctx.fillStyle = fillStyle;
                    ctx.beginPath();
                    ctx.arc(px, py, (state.visualizationMode !== 'default' && p[2] < 0.35) ? 2.5 : 1.5, 0, 2 * Math.PI);
                    ctx.fill();
                });
            } else {
                ctx.fillStyle = '#64748b'; // slate-500
                ctx.font = '12px monospace';
                ctx.textAlign = 'center';
                if (state.activeCve && state.activeCve !== "N/A") {
                    ctx.fillText("WAITING FOR DATA (Need 1000+ samples)...", w / 2, h / 2);
                } else {
                    ctx.fillText("NO MANIFOLD ACTIVE", w / 2, h / 2);
                }
            }

            // E. Current Goal & Threshold (Green)
            if (state.goalCoords && Array.isArray(state.goalCoords) && typeof state.goalIdx === 'number' && state.goalCoords[state.goalIdx]) {
                const g = state.goalCoords[state.goalIdx];
                const gx = mapX(g[0]);
                const gy = mapY(g[1]);

                // Distance Threshold Outline
                if (state.latentThreshold > 0 && state.manifoldCoord && Array.isArray(state.manifoldCoord)) {
                    const mx = mapX(state.manifoldCoord[0]);
                    const my = mapY(state.manifoldCoord[1]);
                    const pixelDist = Math.sqrt((mx - gx) ** 2 + (my - gy) ** 2);

                    let thresholdRadiusPx = 50; // default visible radius
                    if (state.latentDist > 0.001) {
                        thresholdRadiusPx = (pixelDist / state.latentDist) * state.latentThreshold;
                    }

                    // Cap so it doesn't break visibility
                    thresholdRadiusPx = Math.min(Math.max(thresholdRadiusPx, 15), Math.max(w, h));

                    ctx.beginPath();
                    ctx.arc(gx, gy, thresholdRadiusPx, 0, 2 * Math.PI);
                    // Use dark slate instead of red/green to avoid color collision in light theme
                    ctx.strokeStyle = state.latentDist <= state.latentThreshold ? 'rgba(34, 197, 94, 0.5)' : 'rgba(51, 65, 85, 0.4)';
                    ctx.setLineDash([4, 4]);
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                    ctx.setLineDash([]);
                }

                // Goal Glow & Marker
                const pulse = 2 + Math.sin(time / 200) * 2;

                ctx.beginPath();
                ctx.arc(gx, gy, 6 + pulse, 0, 2 * Math.PI);
                ctx.fillStyle = 'rgba(34, 197, 94, 0.3)'; // Emerald
                ctx.fill();

                ctx.beginPath();
                ctx.arc(gx, gy, 5, 0, 2 * Math.PI);
                ctx.fillStyle = '#22c55e';
                ctx.fill();

                ctx.lineWidth = 1.5;
                ctx.strokeStyle = '#fff';
                ctx.stroke();
            }

            // F. Current Latent (Purple)
            if (state.manifoldCoord && Array.isArray(state.manifoldCoord) && state.manifoldCoord.length === 2) {
                const mx = mapX(state.manifoldCoord[0]);
                const my = mapY(state.manifoldCoord[1]);

                // Connection line to goal if present
                if (state.goalCoords && Array.isArray(state.goalCoords) && typeof state.goalIdx === 'number' && state.goalCoords[state.goalIdx]) {
                    const g = state.goalCoords[state.goalIdx];
                    const gx = mapX(g[0]);
                    const gy = mapY(g[1]);

                    ctx.beginPath();
                    ctx.moveTo(mx, my);
                    ctx.lineTo(gx, gy);
                    ctx.strokeStyle = state.latentDist <= state.latentThreshold ? 'rgba(34, 197, 94, 0.5)' : 'rgba(100, 116, 139, 0.4)'; // slate-500
                    ctx.setLineDash([2, 4]);
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                    ctx.setLineDash([]);
                }

                const pulse = Math.sin(time / 150) * 3;

                // Pulse Ring
                ctx.beginPath();
                ctx.arc(mx, my, 8 + Math.max(0, pulse), 0, 2 * Math.PI);
                ctx.strokeStyle = 'rgba(139, 92, 246, 0.5)';
                ctx.lineWidth = 2;
                ctx.stroke();

                // Main Dot
                ctx.beginPath();
                ctx.arc(mx, my, 5, 0, 2 * Math.PI);
                ctx.fillStyle = '#8b5cf6'; // Purple-500
                ctx.fill();
                ctx.lineWidth = 1.5;
                ctx.strokeStyle = '#fff';
                ctx.stroke();
            }

            // G. Match (Orange)
            if (state.matchCoord && Array.isArray(state.matchCoord)) {
                const rx = mapX(state.matchCoord[0]);
                const ry = mapY(state.matchCoord[1]);

                ctx.beginPath();
                ctx.arc(rx, ry, 5, 0, 2 * Math.PI);
                ctx.fillStyle = '#f97316';
                ctx.fill();
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#fff';
                ctx.stroke();
            }

            animationFrameId = requestAnimationFrame(render);
        };

        render();

        return () => {
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    const handleCanvasClick = (e) => {
        if (!onPointClick || points.length === 0 || !mappingRef.current) return;

        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();

        // Correctly scale click coordinates if CSS size differs from render resolution
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        const clickX = (e.clientX - rect.left) * scaleX;
        const clickY = (e.clientY - rect.top) * scaleY;

        const { minX, minY, xRange, yRange, w, h } = mappingRef.current;

        const mapX = (x) => ((x - minX) / xRange) * w;
        const mapY = (y) => h - ((y - minY) / yRange) * h;

        let closestIdx = -1;
        let minPixelDist = Infinity;

        points.forEach((p, idx) => {
            const px = mapX(p[0]);
            const py = mapY(p[1]);
            const dist = Math.sqrt((px - clickX) ** 2 + (py - clickY) ** 2);
            if (dist < minPixelDist) {
                minPixelDist = dist;
                closestIdx = idx;
            }
        });

        console.log(`Canvas Clicked at (${clickX.toFixed(1)}, ${clickY.toFixed(1)}), Closest Px=${minPixelDist.toFixed(1)}, Idx=${closestIdx}`);

        // Allow selecting points loosely
        if (minPixelDist < 40 && closestIdx !== -1) {
            onPointClick(closestIdx);
        }
    };

    return (
        <div className="w-full h-full relative overflow-hidden bg-slate-900 border border-slate-700 rounded-lg">
            {/* [NEW] Distance / Threshold Overlay */}
            {activeCve && activeCve !== "N/A" && typeof latentDist === 'number' && typeof latentThreshold === 'number' && (
                <div className="absolute top-2 right-2 bg-slate-900/80 backdrop-blur-[2px] border border-slate-700 shadow-sm rounded px-2 py-1 flex items-center gap-3 z-10 pointer-events-none">
                    <span className="text-[10px] font-mono text-slate-400 flex items-center gap-1">
                        DIST:
                        <span className={latentDist <= latentThreshold ? "text-emerald-400 font-bold" : "text-slate-100 font-bold"}>
                            {latentDist.toFixed(2)}
                        </span>
                    </span>
                    <span className="text-slate-600">|</span>
                    <span className="text-[10px] font-mono text-slate-400 flex items-center gap-1">
                        THR: <span className="text-slate-100 font-bold">{latentThreshold.toFixed(2)}</span>
                    </span>
                </div>
            )}
            <canvas
                ref={canvasRef}
                onClick={handleCanvasClick}
                className={`block w-full h-full touch-none ${onPointClick ? 'cursor-pointer' : ''}`}
            />
        </div>
    );
};

export default VisualizationPanel;
