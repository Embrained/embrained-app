import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const TrainingMode = ({ data, onExit }) => {
    const [datasets, setDatasets] = React.useState([]);
    const [selectedDatasets, setSelectedDatasets] = React.useState([]);
    const [isProcessing, setIsProcessing] = React.useState(false);
    const [processResult, setProcessResult] = React.useState(null);

    const [trainingFiles, setTrainingFiles] = React.useState([]);
    const [selectedTrajectory, setSelectedTrajectory] = React.useState(null);

    const [dataRoot, setDataRoot] = React.useState("");

    // Training Config
    const [numEpochs, setNumEpochs] = React.useState(5);
    const [lossHistory, setLossHistory] = React.useState([]);

    // Plotting State
    const [showPlotModal, setShowPlotModal] = React.useState(false);
    const [plotLoading, setPlotLoading] = React.useState(false);
    const [plotImage, setPlotImage] = React.useState(null);
    const [plotError, setPlotError] = React.useState(null);

    // Live Feed Listener for Training Progress
    useEffect(() => {
        if (!isProcessing) return; // Don't update status if not explicitly processing

        if (data && data.training_epoch !== undefined) {
            const newEpoch = data.training_epoch;
            const newLoss = data.training_loss;

            // Handle Initialization Phase (Epoch 0)
            if (newEpoch === 0) {
                const percent = Math.round(newLoss * 100);
                setProcessResult(`Preparing Dataset: ${percent}%`);
                return; // Do not add to loss history
            } else {
                setProcessResult(`Training Epoch ${newEpoch}/${numEpochs}... Loss: ${newLoss.toFixed(4)}`);
            }

            setLossHistory(prev => {
                // Avoid duplicates if same epoch (websocket might send same state multiple times)
                // If restarting (epoch < last), reset?
                if (prev.length > 0 && newEpoch < prev[prev.length - 1].epoch) {
                    return [{ epoch: newEpoch, loss: newLoss }];
                }

                // If duplicate
                if (prev.length > 0 && prev[prev.length - 1].epoch === newEpoch) return prev;

                return [...prev, { epoch: newEpoch, loss: newLoss }];
            });
        }
    }, [data?.training_epoch, data?.training_loss, numEpochs, isProcessing]);

    const handleStopTraining = async () => {
        try {
            await fetch('/training/stop', { method: 'POST' });
            setProcessResult("Stopping...");
        } catch (e) {
            console.error("Stop failed", e);
        }
    };

    const handlePlot = async (datasetName) => {
        setShowPlotModal(true);
        setPlotLoading(true);
        setPlotImage(null);
        setPlotError(null);

        try {
            const res = await fetch('/training/visualize_dataset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset: datasetName, root_path: dataRoot })
            });
            const data = await res.json();

            if (data.status === 'success') {
                setPlotImage(data.image);
            } else {
                setPlotError(data.message);
            }
        } catch (e) {
            setPlotError(e.message);
        } finally {
            setPlotLoading(false);
        }
    };

    const handleBrowse = async () => {
        try {
            const res = await fetch('/api/browse', { method: 'POST' });
            const data = await res.json();
            if (data.path) {
                setDataRoot(data.path);
            }
        } catch (e) {
            console.error("Browse failed", e);
        }
    };

    React.useEffect(() => {
        let active = true;

        const fetchCounts = async (initialDatasets) => {
            for (const ds of initialDatasets) {
                if (!active) break;
                if (ds.count !== -1) continue;

                try {
                    const res = await fetch('/api/dataset_count', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name: ds.name, path: dataRoot })
                    });
                    const data = await res.json();

                    if (active && data.count !== undefined) {
                        setDatasets(prev => prev.map(d =>
                            d.name === data.name ? { ...d, count: data.count } : d
                        ));
                    }
                } catch (e) {
                    console.error(`Failed to count ${ds.name}`, e);
                }
            }
        };

        const fetchDatasets = async () => {
            try {
                const query = dataRoot ? `?path=${encodeURIComponent(dataRoot)}&fast=true` : '?fast=true';
                const res = await fetch(`/datasets${query}`);
                const data = await res.json();

                if (active) {
                    setDatasets(data.datasets || []);
                    if (data.root) setDataRoot(data.root);
                    fetchCounts(data.datasets || []);
                }
            } catch (e) {
                console.error("Failed to fetch datasets", e);
            }
        };
        fetchDatasets();

        const fetchTrainingFiles = async () => {
            try {
                const query = dataRoot ? `?path=${encodeURIComponent(dataRoot)}` : '';
                const res = await fetch(`/training/files${query}`);
                const data = await res.json();
                if (active) setTrainingFiles(data.files || []);
            } catch (e) {
                console.error("Failed to fetch training files", e);
            }
        };
        fetchTrainingFiles();

        return () => { active = false; };
    }, [dataRoot]);

    const toggleDataset = (name) => {
        setSelectedDatasets(prev =>
            prev.includes(name)
                ? prev.filter(d => d !== name)
                : [...prev, name]
        );
    };

    const handleSelectAll = () => {
        if (selectedDatasets.length === datasets.length) {
            setSelectedDatasets([]);
        } else {
            setSelectedDatasets(datasets.map(d => d.name));
        }
    };

    const handlePrepareTrajectories = async () => {
        if (selectedDatasets.length === 0) return;

        setIsProcessing(true);
        setProcessResult(null);

        try {
            const res = await fetch('/training/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ datasets: selectedDatasets, root_path: dataRoot })
            });
            const data = await res.json();

            if (data.status === 'success') {
                setProcessResult(`Success: ${data.episodes_count} episodes created.`);
            } else {
                setProcessResult(`Error: ${data.message}`);
            }
        } catch (e) {
            setProcessResult(`Error: ${e.message}`);
        } finally {
            setIsProcessing(false);
            const query = dataRoot ? `?path=${encodeURIComponent(dataRoot)}` : '';
            const fRes = await fetch(`/training/files${query}`);
            const fData = await fRes.json();
            setTrainingFiles(fData.files || []);
        }
    };

    const handleStartTraining = async () => {
        setIsProcessing(true);
        setProcessResult("Initializing CQL Pipeline...");
        setLossHistory([]); // Reset chart

        try {
            setProcessResult("Training Started...");

            const res = await fetch('/training/train_cql', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    root_path: dataRoot,
                    num_epochs: parseInt(numEpochs)
                })
            });

            const data = await res.json();

            if (data.status === 'success') {
                setProcessResult(`Success: Model trained and saved to ${data.policy_path}`);
                // Refresh files immediately
                const query = dataRoot ? `?path=${encodeURIComponent(dataRoot)}` : '';
                await fetch(`/training/files${query}`).then(r => r.json()).then(d => setTrainingFiles(d.files || []));
            } else {
                setProcessResult(`Error: ${data.message}`);
            }
        } catch (e) {
            setProcessResult(`Error: ${e.message}`);
        } finally {
            setIsProcessing(false);
            const query = dataRoot ? `?path=${encodeURIComponent(dataRoot)}` : '';
            const fRes = await fetch(`/training/files${query}`);
            const fData = await fRes.json();
            setTrainingFiles(fData.files || []);
        }
    };

    const trajectories = trainingFiles.filter(f => f.name.endsWith('.json'));
    const models = trainingFiles.filter(f => f.name.endsWith('.pth'));

    return (
        <div className="flex flex-col h-screen w-full bg-gray-100 p-8 relative overflow-y-auto">
            <button
                onClick={onExit}
                className="absolute top-4 left-4 z-50 bg-gray-200 text-gray-600 px-3 py-1 rounded shadow hover:bg-white"
            >
                ← Home
            </button>

            <button
                onClick={async () => {
                    try {
                        await fetch('/shutdown', { method: 'POST' });
                    } catch (e) {
                        console.error("Shutdown failed", e);
                    }
                }}
                className="absolute top-4 right-4 z-50 bg-red-100 text-red-900 hover:bg-red-200 px-3 py-1 rounded shadow text-sm font-bold border border-red-200"
            >
                Stop System
            </button>

            <div className="max-w-5xl mx-auto w-full pt-12">
                <h1 className="text-3xl font-bold text-blue-500 mb-8">Model Training</h1>

                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">

                    {/* SECTION 1: DATASETS */}
                    <div className="flex justify-between items-center mb-4 gap-4">
                        <div className="flex-grow">
                            <h2 className="text-xl font-bold text-gray-700">1. Select Datasets</h2>
                            <code className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded block mt-1 break-all">
                                {dataRoot || "./data"}
                            </code>
                        </div>
                        <div className="flex items-center gap-3">
                            <button
                                onClick={handleBrowse}
                                className="bg-gray-200 hover:bg-gray-300 text-gray-700 font-medium px-3 py-1 rounded text-sm whitespace-nowrap"
                            >
                                📂 Browse
                            </button>
                            {datasets.length > 0 && (
                                <button
                                    onClick={handleSelectAll}
                                    className="text-sm text-blue-600 hover:text-blue-800 font-medium whitespace-nowrap"
                                >
                                    {selectedDatasets.length === datasets.length ? "Deselect All" : "Select All"}
                                </button>
                            )}
                        </div>
                    </div>

                    <div className="border rounded bg-gray-50 p-4 mb-4 text-gray-500 text-sm max-h-60 overflow-y-auto">
                        {datasets.length === 0 ? (
                            <div>No datasets found. Run Live Mode to collect data.</div>
                        ) : (
                            <div className="space-y-2">
                                {datasets.map(ds => (
                                    <div key={ds.name} className="flex justify-between items-center p-2 hover:bg-gray-100 rounded">
                                        <label className="flex items-center gap-3 cursor-pointer flex-grow">
                                            <input
                                                type="checkbox"
                                                checked={selectedDatasets.includes(ds.name)}
                                                onChange={() => toggleDataset(ds.name)}
                                                className="w-4 h-4 text-blue-600"
                                            />
                                            <span className="text-gray-700 font-medium">{ds.name}</span>
                                            <span className="text-gray-400 text-xs">
                                                {ds.count === -1 ? "(Indexing...)" : `(${ds.count} images)`}
                                            </span>
                                        </label>
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handlePlot(ds.name);
                                            }}
                                            className="ml-2 bg-indigo-100 text-indigo-700 hover:bg-indigo-200 px-2 py-1 rounded text-xs font-bold border border-indigo-200"
                                            title="Generate Latent PCA Plot"
                                        >
                                            📉 Plot
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    <button
                        onClick={handlePrepareTrajectories}
                        disabled={selectedDatasets.length === 0 || isProcessing}
                        className={`w-full py-3 text-white font-bold rounded-lg shadow transition-colors mb-8 ${selectedDatasets.length === 0 || isProcessing
                            ? 'bg-gray-400 cursor-not-allowed'
                            : 'bg-blue-500 hover:bg-blue-400'
                            }`}
                    >
                        {isProcessing ? 'Processing...' : `Prepare Trajectories (${selectedDatasets.length} selected)`}
                    </button>


                    {/* SECTION 2: TRAINING CONFIG & CHART */}
                    <h2 className="text-xl font-bold text-gray-700 mb-4">2. Train Model</h2>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
                        {/* Training Controls */}
                        <div className="flex flex-col gap-4">
                            <div>
                                <label className="block text-sm font-bold text-gray-600 mb-1">Number of Epochs</label>
                                <input
                                    type="number"
                                    value={numEpochs}
                                    onChange={(e) => setNumEpochs(e.target.value)}
                                    className="w-full border rounded p-2"
                                    min="1"
                                    max="1000"
                                />
                            </div>

                            <div className="flex-grow"></div>

                            {isProcessing ? (
                                <button
                                    onClick={handleStopTraining}
                                    className="w-full py-4 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg shadow animate-pulse"
                                >
                                    STOP TRAINING
                                </button>
                            ) : (
                                <button
                                    onClick={handleStartTraining}
                                    disabled={trajectories.length === 0}
                                    className={`w-full py-4 text-white font-bold rounded-lg shadow transition-colors ${trajectories.length === 0
                                        ? 'bg-gray-400 cursor-not-allowed'
                                        : 'bg-purple-600 hover:bg-purple-500'
                                        }`}
                                >
                                    Start Training
                                </button>
                            )}
                        </div>

                        {/* Loss Chart */}
                        <div className="md:col-span-2 bg-white border rounded-lg p-4 h-64">
                            <h3 className="text-sm font-bold text-gray-400 uppercase mb-2">Loss Curve</h3>
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={lossHistory}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis
                                        dataKey="epoch"
                                        type="number"
                                        domain={[0, numEpochs]}
                                        allowDataOverflow={true}
                                    />
                                    <YAxis />
                                    <Tooltip />
                                    <Line type="monotone" dataKey="loss" stroke="#8884d8" dot={false} strokeWidth={2} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {processResult && (
                        <div className={`p-4 mb-4 rounded ${processResult.startsWith('Success') ? 'bg-green-50 text-green-700' : 'bg-blue-50 text-blue-700'}`}>
                            {processResult}
                        </div>
                    )}


                    {/* SECTION 3: ARTIFACTS */}
                    <div className='grid grid-cols-2 gap-8 mb-8'>
                        <div>
                            <h3 className="text-md font-bold text-gray-600 mb-2">Trajectory Files</h3>
                            <div className="border rounded bg-gray-50 p-2 text-gray-500 text-sm max-h-32 overflow-y-auto">
                                {trajectories.map(f => (
                                    <div key={f.name} className="flex justify-between px-2">
                                        <span>{f.name}</span>
                                        <span className="text-xs text-gray-400">{f.size_mb} MB</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div>
                            <h3 className="text-md font-bold text-gray-600 mb-2">Saved Models</h3>
                            <div className="border rounded bg-gray-50 p-2 text-gray-500 text-sm max-h-32 overflow-y-auto">
                                {models.map(f => (
                                    <div key={f.name} className="flex justify-between px-2">
                                        <span className="font-medium text-purple-700">{f.name}</span>
                                        <span className="text-xs text-gray-400">{f.size_mb} MB</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                </div>
            </div>

            {/* Plot Modal */}
            {showPlotModal && (
                <div className="fixed inset-0 bg-black bg-opacity-50 z-[100] flex items-center justify-center p-4">
                    <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full p-6 relative">
                        <button
                            onClick={() => setShowPlotModal(false)}
                            className="absolute top-4 right-4 text-gray-500 hover:text-gray-800 text-xl font-bold"
                        >
                            ✕
                        </button>
                        <h2 className="text-2xl font-bold text-gray-800 mb-4">Latent Visualization</h2>
                        {plotLoading ? (
                            <div className="flex flex-col items-center justify-center py-12">
                                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
                                <p className="text-gray-600">Generating...</p>
                            </div>
                        ) : plotImage ? (
                            <div className="flex justify-center">
                                <img src={`data:image/png;base64,${plotImage}`} alt="PCA Plot" className="max-h-[70vh] rounded border" />
                            </div>
                        ) : (
                            <div className="text-center py-12 text-red-500">{plotError || "Failed"}</div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default TrainingMode;
