import React from 'react';
import { ArrowUp, ArrowDown, RotateCcw, RotateCw, Square, Disc } from 'lucide-react';

const ActuatorStatus = ({ data }) => {
    const getActionIcon = (action) => {
        switch (action) {
            case 'FORWARD': return <ArrowUp size={48} className="text-green-500" />;
            case 'LEFT': return <RotateCcw size={48} className="text-blue-500" />;
            case 'RIGHT': return <RotateCw size={48} className="text-blue-500" />;
            case 'BACKWARD': return <ArrowDown size={48} className="text-red-500" />;
            default: return <Square size={48} className="text-red-500" />;
        }
    };

    return (
        <div className="flex gap-1 h-32">
            <div className="flex-1 bg-gray-50 p-4 rounded border border-gray-200 flex flex-col items-center justify-center gap-2 shadow-sm">
                <span className="text-xs text-gray-500 uppercase tracking-wider">Motor Actuator</span>
                <div className="flex items-center gap-4">
                    {getActionIcon(data.action)}
                    <span className="text-xl font-bold">{data.action}</span>
                </div>
            </div>
            <div className="flex-1 bg-gray-50 p-4 rounded border border-gray-200 flex flex-col items-center justify-center gap-2 shadow-sm">
                <span className="text-xs text-gray-500 uppercase tracking-wider">Status LED</span>
                <div className="flex items-center gap-4">
                    <Disc size={48} style={{ color: data.led_color.toLowerCase() === 'n/a' ? '#9ca3af' : data.led_color.toLowerCase() }} />
                    <span className="text-lg font-bold">{data.led_color}</span>
                </div>
            </div>
        </div>
    );
};

export default ActuatorStatus;
