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
import { Activity, Wifi, WifiOff, Disc, Square, Home } from 'lucide-react';
import clsx from 'clsx';

const Header = ({ mode, connected, isRecording, onHome, embodiment }) => {

    const handleStop = async () => {
        try {
            await fetch('/shutdown', { method: 'POST' });
        } catch (e) {
            console.error("Shutdown failed", e);
        }
    };

    const handleRecord = async () => {
        const endpoint = isRecording ? '/record/stop' : '/record/start';
        try {
            await fetch(endpoint, { method: 'POST' });
        } catch (e) {
            console.error("Record toggle failed", e);
        }
    };

    return (
        <div className="fixed top-0 left-0 w-full h-14 flex items-center justify-between px-6 border-b border-purple-100 bg-white/80 z-[100] backdrop-blur shadow-sm">
            <div className="flex items-center gap-3">
                {onHome && (
                    <button onClick={onHome} className="flex items-center gap-2 px-3 py-1 hover:bg-slate-100 rounded text-slate-500 mr-2 transition-colors border border-transparent hover:border-slate-300" title="Exit to Home">
                        <Home size={16} />
                        <span className="text-xs font-bold tracking-wider">EXIT</span>
                    </button>
                )}
                <Activity className="text-blue-600" />
                <span className="font-bold tracking-widest text-lg text-slate-900">EMBRAINED <span className="text-blue-600">DASHBOARD</span></span>

                {embodiment && (
                    <div className="ml-4 px-2 py-0.5 rounded border border-purple-200 bg-purple-50 text-[10px] font-mono text-purple-600 tracking-wider">
                        {embodiment} LINK
                    </div>
                )}
            </div>

            <div className="flex items-center gap-4 text-xs font-bold">
                <button
                    onClick={handleRecord}
                    className={clsx(
                        "flex items-center gap-2 px-3 py-1 rounded transition-colors uppercase tracking-wider border",
                        isRecording
                            ? "bg-red-50 text-red-600 border-red-200 hover:bg-red-100"
                            : "bg-slate-100 text-slate-500 border-slate-200 hover:bg-slate-200 hover:text-slate-700"
                    )}
                >
                    {isRecording ? <Square size={14} fill="currentColor" /> : <Disc size={14} fill="currentColor" />}
                    {isRecording ? "STOP" : "REC"}
                </button>
                <button
                    onClick={handleStop}
                    className="bg-red-50 hover:bg-red-100 text-red-600 border border-red-200 px-3 py-1 rounded transition-colors uppercase tracking-wider"
                >
                    STOP SYSTEM
                </button>
                <div className="w-[1px] h-4 bg-slate-300 mx-2"></div>
                <span className={clsx("px-2 py-1 rounded border", mode === 'DRIVE' ? "bg-red-50 text-red-600 border-red-200" : "bg-blue-50 text-blue-600 border-blue-200")}>
                    MODE: {mode}
                </span>
                <div className="flex items-center gap-2">
                    {connected ? <Wifi size={16} className="text-green-500" /> : <WifiOff size={16} className="text-red-500" />}
                    <span className="text-slate-400">{connected ? "ONLINE" : "DISCONNECTED"}</span>
                </div>
            </div>
        </div>
    );
};

export default Header;
