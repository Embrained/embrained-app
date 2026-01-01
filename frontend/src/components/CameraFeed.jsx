import React from 'react';

const CameraFeed = ({ image }) => {
    return (
        <div className="flex-1 h-full bg-gray-50 relative border border-gray-200 rounded shadow-sm flex items-center justify-center overflow-hidden">
            <div className="absolute top-2 left-2 px-2 py-1 bg-white/80 backdrop-blur text-xs font-bold text-gray-700 rounded border border-gray-200 shadow-sm z-10">
                PRIMARY OPTICAL FEED
            </div>
            {image ? (
                <img src={`data:image/jpeg;base64,${image}`} className="w-[75%] h-[75%] object-contain" alt="Robot Feed" />
            ) : (
                <div className="flex items-center justify-center h-full text-gray-400 font-medium">NO SIGNAL</div>
            )}
        </div>
    );
};

export default CameraFeed;
