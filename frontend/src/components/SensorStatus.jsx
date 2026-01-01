import React from 'react';
import { Ruler, Battery } from 'lucide-react';

const SensorStatus = ({ data }) => {
    return (
        <div className="flex gap-1 h-32">
            {/* Distance Sensor */}
            <div className="flex-1 bg-gray-50 p-4 rounded border border-gray-200 flex flex-col items-center justify-center gap-2 shadow-sm">
                <span className="text-xs text-gray-500 uppercase tracking-wider flex items-center gap-1">
                    <Ruler size={14} /> IR Distance
                </span>
                <div className="flex items-end gap-1">
                    <span className="text-4xl font-mono font-bold text-gray-800">
                        {data.sensor_dist || '0'}
                    </span>
                    <span className="text-sm text-gray-500 font-bold mb-1">cm</span>
                </div>
            </div>

            {/* Battery Sensor */}
            <div className="flex-1 bg-gray-50 p-4 rounded border border-gray-200 flex flex-col items-center justify-center gap-2 shadow-sm">
                <span className="text-xs text-gray-500 uppercase tracking-wider flex items-center gap-1">
                    <Battery size={14} /> Battery
                </span>
                <div className="flex items-end gap-1">
                    <span className="text-4xl font-mono font-bold text-gray-800">
                        {data.sensor_batt || '0'}
                    </span>
                    <span className="text-sm text-gray-500 font-bold mb-1">mV</span>
                </div>
            </div>
        </div>
    );
};

export default SensorStatus;
