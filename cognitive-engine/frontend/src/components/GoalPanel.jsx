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

const GoalPanel = ({
    goalImage,
    goalIdx,
    distance,
    threshold,
    onThresholdChange,
    stopPenalty,

    onPenaltyChange,
    onEdit
}) => {
    return (
        <div className="h-full bg-white border border-gray-200 rounded relative overflow-hidden shadow-sm flex flex-row">
            {/* LEFT: Goal Image (65%) */}
            <div className="w-[65%] h-full relative bg-gray-50 flex items-center justify-center border-r border-gray-100">
                {goalImage ? (
                    <img src={`data:image/jpeg;base64,${goalImage}`} className="w-full h-full object-contain mix-blend-multiply opacity-95" alt="Goal" />
                ) : (
                    <div className="text-[10px] text-gray-400 font-mono">WAITING...</div>
                )}

                {/* Overlay Info */}
                <div className="absolute top-2 left-2 flex flex-col items-start gap-1">
                    <span className="px-2 py-0.5 bg-white/90 backdrop-blur text-[10px] font-bold text-gray-700 rounded border border-gray-200 shadow-sm">
                        GOAL {goalIdx + 1}
                    </span>
                    {distance !== undefined && (
                        <span className="px-2 py-0.5 bg-white/90 backdrop-blur text-[10px] font-mono font-bold text-blue-600 rounded border border-gray-200 shadow-sm">
                            d: {distance.toFixed(3)}
                        </span>
                    )}
                </div>
            </div>

            {/* RIGHT: Sliders (35%) */}
            <div className="w-[35%] h-full flex flex-col p-2 gap-3 justify-center min-w-[100px]">

                {/* Slider 1: Threshold */}
                <div className="flex flex-col gap-1 w-full max-w-[120px]">
                    <div className="flex justify-between items-end">
                        <span className="text-[10px] font-bold text-gray-400 uppercase leading-none">Stop Dist</span>
                        <span className="text-[10px] font-mono font-bold text-blue-600 leading-none">
                            {threshold?.toFixed(1) || "0.8"}
                        </span>
                    </div>
                    <input
                        type="range"
                        min="2.0"
                        max="6.0"
                        step="0.5"
                        value={threshold !== undefined ? threshold : 3.0}
                        onChange={(e) => onThresholdChange && onThresholdChange(parseFloat(e.target.value))}
                        className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                </div>

            </div>
        </div>
    );
};

export default GoalPanel;
