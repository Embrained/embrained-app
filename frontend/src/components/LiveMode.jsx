import React, { useState } from 'react';
import VisualizationPanel from './VisualizationPanel';
import SensorStatus from './SensorStatus';
import ActuatorStatus from './ActuatorStatus';
import Header from './Header';
import CameraFeed from './CameraFeed';
import { useKeyboardControls } from '../hooks/useKeyboardControls';

const LiveMode = ({ data, history, sendMessage, onExit }) => {
    useKeyboardControls(sendMessage);
    const [availableControllers, setAvailableControllers] = useState(['Explorer1', 'Explorer2', 'Rotator1']);
    const [models, setModels] = useState([]);

    React.useEffect(() => {
        // Fetch trained models
        fetch('/api/models')
            .then(res => res.json())
            .then(data => {
                if (data.models) {
                    setModels(data.models);
                }
            })
            .catch(err => console.error("Failed to fetch models:", err));
    }, []);

    return (
        <div className="flex flex-col min-h-screen w-full bg-gray-100 relative overflow-y-auto">
            <button
                onClick={onExit}
                className="fixed top-4 left-4 z-50 bg-blue-500 text-white px-3 py-1 rounded shadow hover:bg-blue-400"
            >
                ← Home
            </button>

            <Header mode="LIVE" connected={true} isRecording={data.is_recording} />

            {/* Main Scrollable Content Area */}
            <div className="flex flex-col items-center w-full pt-20 pb-10">
                <div className="flex flex-col w-[90%] max-w-4xl gap-6">

                    {/* 1. Primary Sensor & View */}
                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-2">
                        <CameraFeed image={data.image} />
                    </div>

                    {/* 2. System Health & Telemetry */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="flex justify-between p-3 bg-white rounded-lg border border-gray-200 shadow-sm">
                            <span className="text-gray-500 font-bold text-sm">FPS</span>
                            <span className="text-green-600 font-mono font-bold">{data.fps ? data.fps.toFixed(1) : '0.0'}</span>
                        </div>
                        <div className="flex justify-between p-3 bg-white rounded-lg border border-gray-200 shadow-sm">
                            <span className="text-gray-500 font-bold text-sm">LATENCY</span>
                            <span className="text-green-600 font-mono font-bold">14ms</span>
                        </div>
                        <div className="flex justify-between p-3 bg-white rounded-lg border border-gray-200 shadow-sm">
                            <span className="text-gray-500 font-bold text-sm">DIST</span>
                            <div className="flex items-center gap-1">
                                <span className="text-blue-600 font-mono font-bold text-lg">{data.sensor_dist || '0'}</span>
                                <span className="text-gray-400 text-xs">cm</span>
                            </div>
                        </div>
                        <div className="flex justify-between p-3 bg-white rounded-lg border border-gray-200 shadow-sm">
                            <span className="text-gray-500 font-bold text-sm">BATT</span>
                            <div className="flex items-center gap-1">
                                <span className="text-orange-600 font-mono font-bold text-lg">{data.sensor_batt || '0'}</span>
                                <span className="text-gray-400 text-xs">raw</span>
                            </div>
                        </div>
                    </div>

                    {/* 3. Latent Visualization */}
                    <div className="w-full bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                        <VisualizationPanel
                            keypoints={data.current_latent}
                            manifoldCoord={data.manifold_coord}
                        />
                    </div>

                    {/* 4. Controls & Actuators */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Autonomous Controllers Panel */}
                        <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-200">
                            <h3 className="text-sm font-bold text-gray-400 uppercase mb-3">Autonomous Controllers</h3>

                            <div className="flex flex-col gap-4">
                                {/* Explorers */}
                                <div>
                                    <h4 className="text-xs font-bold text-gray-300 mb-2">Exploration</h4>
                                    <div className="flex flex-wrap gap-2">
                                        {availableControllers.map((name) => (
                                            <button
                                                key={name}
                                                onClick={() => sendMessage('SET_CONTROLLER', data.controller === name ? null : name)}
                                                className={`px-3 py-2 rounded text-sm font-bold transition-colors ${data.controller === name
                                                    ? 'bg-blue-500 text-white shadow-inner'
                                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                                    }`}
                                            >
                                                {name}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Models */}
                                {models.length > 0 && (
                                    <div>
                                        <h4 className="text-xs font-bold text-gray-300 mb-2">Trained Models</h4>
                                        <div className="flex flex-wrap gap-2">
                                            {models.map((name) => (
                                                <button
                                                    key={name}
                                                    onClick={() => sendMessage('SET_CONTROLLER', data.controller === name ? null : name)}
                                                    className={`px-3 py-2 rounded text-sm font-bold transition-colors ${data.controller === name
                                                        ? 'bg-purple-500 text-white shadow-inner'
                                                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                                        }`}
                                                >
                                                    {name}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {data.controller && (
                                <div className="mt-4 p-2 bg-blue-50 rounded text-center border border-blue-100">
                                    <span className="text-xs font-mono text-blue-600 font-bold animate-pulse">
                                        ● ACTIVE ALGORITHM: {data.controller}
                                    </span>
                                </div>
                            )}
                        </div>

                        {/* Actuator Status */}
                        <ActuatorStatus data={data} />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LiveMode;
