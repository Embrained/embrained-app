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
import { ArrowUp, ArrowDown, RotateCcw, RotateCw, Square, Disc } from 'lucide-react';

const ActuatorStatus = ({ data }) => {
    return (
        <div className="flex flex-col gap-2 w-full">
            {/* Action Label (Horizontal) */}
            <div className="bg-gray-50/80 p-2 rounded border border-gray-200 flex items-center justify-between shadow-sm">
                <span className="text-[10px] text-gray-400 uppercase tracking-wider font-bold">Action</span>
                <span className="text-xs font-bold font-mono text-slate-700">{data.action}</span>
            </div>

            {/* Motor Values (Side-by-Side) */}
            <div className="flex gap-2">
                <div className="flex-1 bg-white p-2 rounded border border-indigo-100 flex flex-col items-center justify-center shadow-sm">
                    <span className="text-[9px] text-indigo-400 uppercase font-bold mb-1">Left Motor</span>
                    <div className={`text-xl font-mono font-bold ${(data.motor_l || 0) > 0 ? 'text-green-600' : (data.motor_l || 0) < 0 ? 'text-red-600' : 'text-slate-400'}`}>
                        {Math.round(data.motor_l || 0)}
                    </div>
                </div>
                <div className="flex-1 bg-white p-2 rounded border border-indigo-100 flex flex-col items-center justify-center shadow-sm">
                    <span className="text-[9px] text-indigo-400 uppercase font-bold mb-1">Right Motor</span>
                    <div className={`text-xl font-mono font-bold ${(data.motor_r || 0) > 0 ? 'text-green-600' : (data.motor_r || 0) < 0 ? 'text-red-600' : 'text-slate-400'}`}>
                        {Math.round(data.motor_r || 0)}
                    </div>
                </div>
            </div>

            {/* LED Status (Horizontal) */}
            <div className="bg-gray-50/80 p-2 rounded border border-gray-200 flex items-center justify-between shadow-sm">
                <span className="text-[10px] text-gray-400 uppercase tracking-wider font-bold">LED</span>
                <div className="flex items-center gap-2">
                    <Disc
                        size={12}
                        className={data.led_color?.toLowerCase() !== 'n/a' ? 'animate-pulse-led' : ''}
                        style={{ color: data.led_color?.toLowerCase() === 'n/a' ? '#9ca3af' : data.led_color?.toLowerCase() }}
                    />
                    <span className="text-[10px] font-bold font-mono">{data.led_color}</span>
                </div>
            </div>
        </div>
    );
};

export default ActuatorStatus;
