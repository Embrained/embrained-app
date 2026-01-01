import React from 'react';

const ModeButton = ({ title, description, onClick, disabled }) => (
    <button
        onClick={onClick}
        disabled={disabled}
        className={`p-6 rounded-lg border-2 text-left transition-all ${disabled
            ? 'border-gray-300 bg-gray-100 text-gray-400 cursor-not-allowed'
            : 'border-blue-500 bg-white hover:bg-blue-500 hover:text-white shadow-md hover:shadow-lg'
            }`}
    >
        <div className="text-xl font-bold mb-2">{title}</div>
        <div className={`text-sm ${disabled ? 'text-gray-400' : 'text-gray-600 hover:text-white'}`}>
            {description}
        </div>
    </button>
);

const Home = ({ onSelectMode }) => {
    return (
        <div className="flex flex-col items-center justify-center h-screen bg-gray-100 font-mono">
            <div className="mb-12 text-center">
                <h1 className="text-5xl font-bold text-blue-500 mb-4 tracking-tighter">EMBRAINED</h1>
                <p className="text-gray-500 text-lg">Disaggregated Artificial Intelligence System</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl w-full px-8">
                <ModeButton
                    title="LIVE"
                    description="Teleoperation and Data Collection via Manual Control"
                    onClick={() => onSelectMode('LIVE')}
                />

                <ModeButton
                    title="TRAINING"
                    description="Fine-tune models on collected datasets"
                    onClick={() => onSelectMode('TRAINING')}
                />
            </div>

            <div className="mt-12 flex gap-8">
                <button className="text-gray-400 hover:text-blue-500 text-sm underline">
                    Settings
                </button>
                <button
                    onClick={async () => {
                        try {
                            await fetch('/shutdown', { method: 'POST' });
                        } catch (e) {
                            console.error("Shutdown failed", e);
                        }
                    }}
                    className="text-red-900 hover:text-red-500 text-sm font-bold tracking-widest border border-red-900/30 px-4 py-1 rounded hover:border-red-500 transition-colors"
                >
                    Stop System
                </button>
            </div>
        </div>
    );
};

export default Home;
