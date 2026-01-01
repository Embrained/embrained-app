import React from 'react';
import { Activity, Wifi, WifiOff, Disc, Square } from 'lucide-react';
import clsx from 'clsx';

const Header = ({ mode, connected, isRecording }) => {

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
        <div className="fixed top-0 left-0 w-full h-12 flex items-center justify-between px-6 border-b border-gray-200 bg-white/90 z-10 backdrop-blur shadow-sm">
            <div className="flex items-center gap-3">
                <Activity className="text-blue-600" />
                <span className="font-bold tracking-widest text-lg text-gray-900">EMBRAINED <span className="text-blue-500">DASHBOARD</span></span>
            </div>
            <div className="flex items-center gap-4 text-xs font-bold">
                <button
                    onClick={handleRecord}
                    className={clsx(
                        "flex items-center gap-2 px-3 py-1 rounded transition-colors uppercase tracking-wider border",
                        isRecording
                            ? "bg-red-50 text-red-600 border-red-200 hover:bg-red-100"
                            : "bg-gray-100 text-gray-600 border-gray-200 hover:bg-gray-200 hover:text-gray-900"
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
                <div className="w-[1px] h-4 bg-gray-300 mx-2"></div>
                <span className={clsx("px-2 py-1 rounded", mode === 'DRIVE' ? "bg-red-50 text-red-600 border border-red-200" : "bg-blue-50 text-blue-600 border border-blue-200")}>
                    MODE: {mode}
                </span>
                <div className="flex items-center gap-2">
                    {connected ? <Wifi size={16} className="text-green-500" /> : <WifiOff size={16} className="text-red-500" />}
                    <span className="text-gray-600">{connected ? "ONLINE" : "DISCONNECTED"}</span>
                </div>
            </div>
        </div>
    );
};

export default Header;
