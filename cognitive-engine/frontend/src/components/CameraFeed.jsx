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

import React from 'react';

const CameraFeed = ({ image, goalImage, isRecording, recordingFrames }) => {
    return (
        <div className="relative w-full h-full bg-black flex items-center justify-center overflow-hidden rounded-xl border border-white/10 shadow-2xl">
            {/* Overlay Grid */}
            <div className="absolute inset-0 z-10 pointer-events-none opacity-20"
                style={{
                    backgroundImage: 'linear-gradient(rgba(59, 130, 246, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(59, 130, 246, 0.1) 1px, transparent 1px)',
                    backgroundSize: '40px 40px'
                }}>
            </div>

            {/* Status Overlays */}
            <div className="absolute top-4 left-4 z-20 flex gap-2">

                {isRecording && (
                    <div className="flex gap-2">
                        <div className="px-2 py-1 bg-red-900/80 backdrop-blur text-[10px] font-mono text-red-400 border border-red-500/50 rounded animate-pulse">
                            ● REC
                        </div>
                        <div className="px-2 py-1 bg-black/60 backdrop-blur text-[10px] font-mono text-white border border-white/20 rounded">
                            {recordingFrames || 0} f
                        </div>
                    </div>
                )}
            </div>

            {/* Crosshair Overlay */}
            <div className="absolute inset-0 z-10 pointer-events-none flex items-center justify-center opacity-30">
                <div className="w-8 h-8 border-l border-t border-white/50 absolute top-10 left-10"></div>
                <div className="w-8 h-8 border-r border-t border-white/50 absolute top-10 right-10"></div>
                <div className="w-8 h-8 border-l border-b border-white/50 absolute bottom-10 left-10"></div>
                <div className="w-8 h-8 border-r border-b border-white/50 absolute bottom-10 right-10"></div>
                <div className="w-4 h-4 text-blue-500">+</div>
            </div>

            {/* Goal Image Inset */}
            {goalImage && (
                <div className="absolute top-4 right-4 z-30 w-[128px] h-[128px] rounded-lg border-2 border-emerald-500/50 shadow-2xl overflow-hidden bg-black/50 backdrop-blur">
                    <div className="absolute top-1 right-1 px-1.5 py-0.5 bg-emerald-900/80 backdrop-blur text-[8px] font-mono text-emerald-200 border border-emerald-500/50 rounded z-40 shadow-sm pointer-events-none uppercase tracking-wider">
                        TARGET GOAL
                    </div>
                    <img
                        src={(goalImage.startsWith('http') || goalImage.startsWith('/api/')) ? goalImage : `data:image/jpeg;base64,${goalImage}`}
                        className="w-full h-full object-contain"
                        alt="Goal Feed"
                    />
                </div>
            )}

            {image ? (
                <img src={`data:image/jpeg;base64,${image}`} className="w-full h-full object-contain" alt="Robot Feed" />
            ) : (
                <div className="flex flex-col items-center justify-center text-gray-600 font-mono text-xs">
                    <div className="animate-pulse mb-2">SIGNAL LOST</div>
                    <div>WAITING FOR VIDEO STREAM...</div>
                </div>
            )}
        </div>
    );
};

export default CameraFeed;

