import React from 'react';

const GoalPanel = ({ goalImage, goalIdx }) => {
    return (
        <div className="h-1/4 bg-white border border-gray-200 rounded relative overflow-hidden shadow-sm flex items-center justify-center">
            <div className="absolute top-2 left-2 px-2 py-1 bg-white/80 backdrop-blur text-xs font-bold text-gray-700 rounded border border-gray-200 shadow-sm z-10">
                TARGET LATENT: GOAL {goalIdx + 1}
            </div>
            {goalImage ? (
                <img src={`data:image/jpeg;base64,${goalImage}`} className="w-full h-full object-cover opacity-80" alt="Goal" />
            ) : (
                <div className="flex items-center justify-center h-full text-xs text-gray-500">AWAITING GOAL</div>
            )}
        </div>
    );
};

export default GoalPanel;
