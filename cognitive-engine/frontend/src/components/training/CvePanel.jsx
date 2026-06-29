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
import { Layers, Play, Square } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { API } from '../../services/api';

const CvePanel = ({
    dataRoot, getFileMeta, getExpectedCveName,
    cveEpochs, setCveEpochs,
    cveBatchSize, setCveBatchSize,
    cveLearningRate, setCveLearningRate,
    cveBeta, setCveBeta,
    cveModelSize, setCveModelSize,
    isCveTraining, setIsCveTraining,
    cveLossHistory, setCveLossHistory,
    cveValidationImage, setCveValidationImage,
    processResult, fetchFiles,
    lastCveDataRef, ignoreStaleCveRef,
    pipelineArch, setPipelineArch,
    selectedDatasets,
    activeCveName, setActiveCveName,
    handleLoadLatestCve,
    cveImageSize, setCveImageSize,
    cveNumLayers, setCveNumLayers,
    cveLatentDim, setCveLatentDim,
    pipelineArchitecture, setPipelineArchitecture,
    dreamerTag, setDreamerTag
}) => {


    const startTraining = async () => {
        // Generate Name Immediately
        let timestamp = new Date().toISOString().replace(/T/, '_').replace(/:/g, '-').split('.')[0].replace(/-/g, '').replace('_', '_');
        // Actually let's just use a clean YYYYMMDD_HHMMSS format like Python's time.strftime("%Y%m%d_%H%M%S")
        const now = new Date();
        const ts = now.getFullYear().toString() +
            (now.getMonth() + 1).toString().padStart(2, '0') +
            now.getDate().toString().padStart(2, '0') + "_" +
            now.getHours().toString().padStart(2, '0') +
            now.getMinutes().toString().padStart(2, '0') +
            now.getSeconds().toString().padStart(2, '0');

        let generatedName = "";
        let datasetName = "tinycve";
        if (dataRoot) {
            const parts = dataRoot.replace(/\\/g, '/').split('/').filter(p => !!p);
            const last = parts.length > 0 ? parts[parts.length - 1] : "";
            if (last && last !== 'data') datasetName = last;
        }

        if (pipelineArch === "dreamer") generatedName = `${dreamerTag}-dreamer_${ts}.pth`;
        else generatedName = `${datasetName}-cve_${pipelineArchitecture}_${ts}.pth`;
        
        setActiveCveName(generatedName);
        setIsCveTraining(true);
        setCveLossHistory([]);
        setCveValidationImage(null);
        lastCveDataRef.current = { epoch: -1, loss: -1 };
        ignoreStaleCveRef.current = true;


        if (!dataRoot && (!processResult || processResult.status !== 'success')) {
            const confirmed = window.confirm("⚠️ Warning: No Dataset Directory Selected. Proceed with default?");
            if (!confirmed) {
                setIsCveTraining(false);
                setActiveCveName("");
                return;
            }
        }

        try {
            let res = null;
            if (pipelineArch === "dreamer") {
                await API.post('/training/train_dreamer', {
                    num_epochs: parseInt(cveEpochs) || 50,
                    batch_size: parseInt(cveBatchSize) || 32,
                    learning_rate: parseFloat(cveLearningRate) || 0.0001,
                    root_path: dataRoot,
                    tag: dreamerTag,
                    selected_datasets: selectedDatasets,
                    model_filename: generatedName
                });
            } else {
                res = await API.trainCVE({
                    num_epochs: parseInt(cveEpochs) || 10,
                    batch_size: parseInt(cveBatchSize) || 32,
                    learning_rate: parseFloat(cveLearningRate) || 0.0001,
                    cve_beta: parseFloat(cveBeta) || 0.5,
                    model_size: cveModelSize,
                    root_path: dataRoot,
                    selected_datasets: selectedDatasets,
                    model_filename: generatedName,
                    architecture: pipelineArchitecture,
                    latent_dim: parseInt(cveLatentDim, 10)
                });
            }
            setIsCveTraining(false);
            fetchFiles(dataRoot);
            
            if (res && res.manifold_plot) {
                const pltStr = String(res.manifold_plot);
                const b64 = pltStr.startsWith("data:") ? pltStr : `data:image/png;base64,${pltStr}`;
                setCveValidationImage(b64);
            }
        } catch (e) {
            console.error(e);
            setIsCveTraining(false);
            setActiveCveName("");
        }
    };

    const stopTraining = async () => {
        await API.stopTraining();
    };

    const formatYAxis = (val) => {
        if (val === 0) return "0";
        if (Math.abs(val) >= 1000) return `${Math.round(val / 1000)}K`;
        return val.toFixed(0);
    };

    const modelName = getExpectedCveName(dataRoot);
    const modelMeta = getFileMeta(modelName);

    const modelNameDisplay = activeCveName || (isCveTraining ? "Generating..." : "");

    return (
        <div className="glass-panel p-2 flex flex-col gap-2 h-full">
            <div className="flex items-center gap-2 border-b border-indigo-100 pb-1">
                <Layers className={pipelineArch === 'latentslam' ? "text-purple-600" : (pipelineArch === 'dreamer' ? "text-orange-600" : "text-indigo-600")} size={18} />
                <select 
                    value={pipelineArch || 'cve'} 
                    onChange={(e) => setPipelineArch(e.target.value)} 
                    className={`text-lg font-semibold bg-transparent outline-none cursor-pointer focus:ring-0 appearance-none ${pipelineArch === 'latentslam' ? 'text-purple-800' : (pipelineArch === 'dreamer' ? 'text-orange-800' : 'text-slate-800')}`}
                    disabled={isCveTraining}
                >
                    <option value="cve" className="text-slate-800">Vision System (bCVE)</option>
                    <option value="dreamer" className="text-orange-800">World Model (DreamerV3)</option>
                </select>
            </div>

            <div className="flex flex-col gap-2">
                <div className="grid grid-cols-5 gap-1">
                    {/* Epochs */}
                    <div className="flex flex-col gap-0.5">
                        <span className="text-[9px] text-slate-400 font-bold uppercase">Epochs</span>
                        <input type="number" min="1" max="1000" value={cveEpochs} onChange={(e) => setCveEpochs(e.target.value)} className="w-full bg-white/50 border border-slate-200 rounded px-1 py-0.5 text-center text-[10px] font-mono focus:outline-none focus:border-indigo-400 text-slate-700" disabled={isCveTraining} />
                    </div>

                    {/* Batch Size */}
                    <div className="flex flex-col gap-0.5">
                        <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Batch</span>
                        <select value={cveBatchSize} onChange={(e) => setCveBatchSize(e.target.value)} className="w-full bg-white/50 border border-slate-200 rounded px-1 py-0.5 text-[10px] font-mono focus:outline-none focus:border-indigo-400 text-slate-700 bg-transparent" disabled={isCveTraining}>
                            <option value="16">16</option>
                            <option value="32">32</option>
                            <option value="64">64</option>
                            <option value="128">128</option>
                            <option value="256">256</option>
                            <option value="512">512</option>
                            <option value="1024">1024</option>
                        </select>
                    </div>

                    {/* LR */}
                    <div className="flex flex-col gap-0.5">
                        <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">LR</span>
                        <select value={cveLearningRate} onChange={(e) => setCveLearningRate(e.target.value)} className="w-full bg-white/50 border border-slate-200 rounded px-1 py-0.5 text-[10px] font-mono focus:outline-none focus:border-indigo-400 text-slate-700 bg-transparent" disabled={isCveTraining}>
                            <option value={0.001}>1e-3</option>
                            <option value={0.0001}>1e-4</option>
                            <option value={0.00001}>1e-5</option>
                            <option value={0.0000001}>1e-7</option>
                        </select>
                    </div>

                    {/* Beta Value */}
                    <div className="flex flex-col gap-0.5">
                        <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Beta</span>
                        <select value={cveBeta} onChange={(e) => setCveBeta(e.target.value)} className="w-full bg-white/50 border border-slate-200 rounded px-1 py-0.5 text-[10px] font-mono focus:outline-none focus:border-indigo-400 text-slate-700 bg-transparent" disabled={isCveTraining}>
                            <option value={4.0}>4.0</option>
                            <option value={2.0}>2.0</option>
                            <option value={1.0}>1.0</option>
                            <option value={0.5}>0.5</option>
                            <option value={0.25}>0.25</option>
                            <option value={0.1}>0.1</option>
                        </select>
                    </div>

                    {/* Size */}
                    <div className="flex flex-col gap-0.5">
                        <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Size</span>
                        <select value={cveModelSize} onChange={(e) => setCveModelSize(e.target.value)} className="w-full bg-white/50 border border-slate-200 rounded px-1 py-0.5 text-[10px] font-mono focus:outline-none focus:border-indigo-400 text-slate-700 bg-transparent" disabled={isCveTraining}>
                            {pipelineArch === 'dreamer' ? (
                                <>
                                    <option value="small">Small (256)</option>
                                    <option value="medium">Medium (512)</option>
                                    <option value="large">Large (1024)</option>
                                    <option value="enormous">Huge (2048)</option>
                                    <option value="tectonic">Massive (4096)</option>
                                </>
                            ) : (
                                <>
                                    <option value="small">Small (4 layers)</option>
                                    <option value="medium">Medium (4L, double)</option>
                                    <option value="large">Large (5 layers)</option>
                                    <option value="enormous">Huge (5L, double)</option>
                                    <option value="tectonic">Massive (6L)</option>
                                </>
                            )}
                        </select>
                    </div>
                </div>

                    <div className="grid grid-cols-3 gap-1 mt-0.5">
                        {/* Architecture */}
                        <div className="flex flex-col gap-0.5">
                            <span className="text-[9px] text-indigo-500 font-bold uppercase tracking-wider">Architecture</span>
                            <select value={pipelineArchitecture} onChange={(e) => setPipelineArchitecture(e.target.value)} className="w-full bg-white/50 border border-slate-200 rounded px-1 py-0.5 text-[10px] font-mono focus:outline-none focus:border-indigo-400 text-indigo-700" disabled={isCveTraining}>
                            <option value="continuous">Continuous (β-CVE)</option>
                                <option value="discrete">Discrete (VQ-CVE)</option>
                                <option value="contrastive">Contrastive (CVE)</option>
                            </select>
                        </div>
                        {/* Resolution */}
                        <div className="flex flex-col gap-0.5">
                            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Resolution</span>
                            <select value={cveImageSize} onChange={(e) => setCveImageSize(e.target.value)} className="w-full bg-white/50 border border-slate-200 rounded px-1 py-0.5 text-[10px] font-mono focus:outline-none focus:border-indigo-400 text-slate-700 bg-transparent" disabled={isCveTraining}>
                                <option value="64">64x64</option>
                                <option value="128">128x128</option>
                            </select>
                        </div>
                        {/* Latent Dimension */}
                        <div className="flex flex-col gap-0.5">
                            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Latent Dims</span>
                            <select value={cveLatentDim} onChange={(e) => setCveLatentDim(e.target.value)} className="w-full bg-white/50 border border-slate-200 rounded px-1 py-0.5 text-[10px] font-mono focus:outline-none focus:border-indigo-400 text-slate-700 bg-transparent" disabled={isCveTraining}>
                                <option value="32">32 (Lightweight)</option>
                                <option value="64">64 (Balanced)</option>
                                <option value="128">128 (Standard)</option>
                                <option value="256">256 (High Capacity)</option>
                                <option value="512">512 (Ultra Capacity)</option>
                                <option value="1024">1024 (Tectonic Capacity)</option>
                            </select>
                        </div>
                    </div>

                {pipelineArch === 'dreamer' && (
                    <div className="flex flex-col gap-1 bg-orange-50/50 p-1.5 rounded border border-orange-100">
                        <span className="text-[9px] text-orange-600 font-bold uppercase">Policy Tag (Filenaming)</span>
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={dreamerTag}
                                onChange={(e) => setDreamerTag(e.target.value)}
                                className="flex-grow bg-white border border-orange-200 rounded px-2 py-1 text-[11px] font-mono focus:outline-none focus:border-orange-400 text-slate-700"
                                placeholder="e.g. red_ball"
                                disabled={isCveTraining}
                            />
                            <span className="text-[10px] text-slate-400 self-center font-mono">_dreamer.pth</span>
                        </div>
                    </div>
                )}
            </div>

            {
                !isCveTraining ? (
                    <button onClick={startTraining} className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white py-1.5 rounded font-bold flex justify-center items-center gap-2 shadow-sm text-xs">
                        <Play size={14} /> Start Training
                    </button>
                ) : (
                    <button onClick={stopTraining} className="w-full bg-red-500 text-white py-1.5 rounded font-bold flex justify-center items-center gap-2 shadow-sm text-xs">
                        <Square size={14} /> Stop CVE Training
                    </button>
                )
            }

            {/* Plots */}
            <div className="flex flex-row gap-1 h-28">
                {/* 1. Recon Loss */}
                <div className="flex-1 bg-white/60 rounded border border-slate-200/50 p-1 relative">
                    <span className="absolute top-1 right-2 text-[8px] font-bold text-purple-500 uppercase">{pipelineArch === 'latentslam' ? 'Recon/Total' : (pipelineArch === 'imitation' ? 'Training Loss' : (pipelineArchitecture === 'contrastive' ? 'Contrastive Loss' : 'Recon Loss'))}</span>
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={cveLossHistory}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                            <XAxis
                                type="number"
                                dataKey="epoch"
                                domain={[0, Math.max(cveEpochs, cveLossHistory.length > 0 ? Math.ceil(cveLossHistory[cveLossHistory.length - 1].epoch) : 0)]}
                                stroke="#94a3b8"
                                fontSize={9}
                                tickFormatter={(val) => val % 1 === 0 ? val : val.toFixed(1)}
                            />
                            <YAxis domain={['auto', 'auto']} fontSize={9} tickFormatter={formatYAxis} width={30} />
                            <Line type="monotone" dataKey={pipelineArch === 'latentslam' ? 'recon' : "loss"} stroke="#ec4899" strokeWidth={1.5} dot={false} isAnimationActive={false} />

                        </LineChart>
                    </ResponsiveContainer>
                </div>

                {/* 2. KLD Loss */}
                <div className="flex-1 bg-white/60 rounded border border-slate-200/50 p-1 relative">
                    <span className="absolute top-1 right-2 text-[8px] font-bold text-indigo-500 uppercase">{pipelineArch === 'latentslam' ? 'KL Divergence' : (pipelineArch === 'dreamer' ? 'Val Loss' : (pipelineArchitecture === 'contrastive' ? 'Action Loss' : 'KL Divergence'))}</span>
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={cveLossHistory}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                            <XAxis
                                type="number"
                                dataKey="epoch"
                                domain={[0, Math.max(cveEpochs, cveLossHistory.length > 0 ? Math.ceil(cveLossHistory[cveLossHistory.length - 1].epoch) : 0)]}
                                stroke="#94a3b8"
                                fontSize={9}
                                tickFormatter={(val) => val % 1 === 0 ? val : val.toFixed(1)}
                            />
                            <YAxis domain={['auto', 'auto']} fontSize={9} tickFormatter={formatYAxis} width={30} />
                            <Line type="monotone" dataKey={pipelineArch === 'latentslam' ? 'kl' : (pipelineArch === 'dreamer' ? 'val_loss' : 'kld')} stroke="#6366f1" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Validation Plot */}
            <div className="flex-1 min-h-[300px] bg-white/40 rounded border border-slate-200/50 flex flex-col overflow-hidden relative group">
                {cveValidationImage ? (
                    <img src={cveValidationImage} alt="Manifold" className="w-full h-full object-contain p-1" />
                ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-400 text-xs italic">
                        Waiting for training completion...
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="mt-auto pt-1 border-t border-indigo-50 text-[10px] text-slate-500">
                <div className="flex justify-between items-center h-5">
                    <span className="font-semibold">Model:</span>
                    {modelNameDisplay ? (
                        <span
                            className="font-mono bg-indigo-50 text-indigo-700 px-1 py-0.5 rounded select-all border border-indigo-100 cursor-pointer hover:bg-indigo-100 transition-colors"
                            onClick={() => { if (handleLoadLatestCve) handleLoadLatestCve(); }}
                            title="Click to reload latest CVE model"
                        >
                            {modelNameDisplay}
                        </span>
                    ) : (
                        <span
                            className="text-slate-400 italic cursor-pointer hover:text-indigo-500 hover:underline transition-colors"
                            onClick={() => { if (handleLoadLatestCve) handleLoadLatestCve(); }}
                            title="Click to load latest model"
                        >
                            Empty - click to load latest or start training
                        </span>
                    )}
                </div>
                {modelMeta && !isCveTraining && (
                    <div className="flex justify-between text-[9px] text-slate-400 mt-0.5">
                        <span className="text-emerald-500 font-bold">FOUND</span>
                        <span>{modelMeta.size_mb} MB</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export default CvePanel;
