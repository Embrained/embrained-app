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
import { Activity, Play, Square } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
import { API } from '../../services/api';

const ForwardModelPanel = ({
    dataRoot, getFileMeta, getExpectedForwardName, getExpectedVaeName,
    forwardEpochs, setForwardEpochs,
    forwardBatchSize, setForwardBatchSize,
    forwardLearningRate, setForwardLearningRate,
    forwardApproach, setForwardApproach,
    isForwardTraining, setIsForwardTraining,
    forwardLossHistory, setForwardLossHistory,
    policyHeatmap, setPolicyHeatmap,
    fetchFiles, fetchPolicyHeatmap, lastForwardDataRef, ignoreStaleForwardRef,
    selectedDatasets,
    activeForwardName, setActiveForwardName,
    handleLoadLatestForward,
    transitionLossWeight, setTransitionLossWeight,
    contrastiveWeight, setContrastiveWeight,
    vaeBeta, setVaeBeta,
    pipelineArchitecture, vaeLatentDim, vaeImageSize, vaeNumLayers
}) => {

    const startTraining = async () => {
        const now = new Date();
        const ts = now.getFullYear().toString() +
            (now.getMonth() + 1).toString().padStart(2, '0') +
            now.getDate().toString().padStart(2, '0') + "_" +
            now.getHours().toString().padStart(2, '0') +
            now.getMinutes().toString().padStart(2, '0') +
            now.getSeconds().toString().padStart(2, '0');

        let datasetName = "topological";
        if (dataRoot) {
            const parts = dataRoot.replace(/\\/g, '/').split('/').filter(p => !!p);
            const last = parts.length > 0 ? parts[parts.length - 1] : "";
            if (last && last !== 'data') datasetName = last;
        }

        const vaeModel = getExpectedVaeName(dataRoot) || "";
        const generatedName = `topological_forward_${forwardApproach}_${ts}.pth`;

        setActiveForwardName(generatedName);
        setIsForwardTraining(true);
        setForwardLossHistory([]);
        setPolicyHeatmap(null);
        lastForwardDataRef.current = { epoch: -1, loss: -1 };
        ignoreStaleForwardRef.current = true;

        try {
            let res = null;
            if (forwardApproach === "latentslam") {
                res = await API.post('/training/train_latentslam', {
                    num_epochs: parseInt(forwardEpochs) || 20,
                    batch_size: parseInt(forwardBatchSize) || 64,
                    learning_rate: parseFloat(forwardLearningRate) || 0.0001,
                    vae_beta: parseFloat(vaeBeta) || 0.5,
                    model_size: "large", // or hardcoded fallback
                    root_path: dataRoot,
                    selected_datasets: selectedDatasets,
                    model_filename: generatedName,
                    image_size: parseInt(vaeImageSize, 10),
                    num_layers: parseInt(vaeNumLayers, 10),
                    latent_dim: parseInt(vaeLatentDim, 10),
                    transition_loss_weight: parseFloat(transitionLossWeight),
                    contrastive_weight: parseFloat(contrastiveWeight),
                    architecture: pipelineArchitecture
                });
            } else {
                res = await API.trainForward({
                    num_epochs: parseInt(forwardEpochs) || 20,
                    batch_size: parseInt(forwardBatchSize) || 128,
                    learning_rate: parseFloat(forwardLearningRate) || 0.0001,
                    forward_approach: forwardApproach,
                    root_path: dataRoot,
                    selected_datasets: selectedDatasets,
                    model_filename: generatedName,
                });
            }
            setIsForwardTraining(false);
            if (res.status === "success") {
                fetchFiles(dataRoot);
                if (res.policy_heatmap) {
                    const pltStr = String(res.policy_heatmap);
                    const b64 = pltStr.startsWith("data:") ? pltStr : `data:image/png;base64,${pltStr}`;
                    setPolicyHeatmap(b64);
                } else if (fetchPolicyHeatmap) {
                    fetchPolicyHeatmap(dataRoot, generatedName);
                } else setPolicyHeatmap(null);
            }
        } catch (e) {
            console.error(e);
            setIsForwardTraining(false);
            setActiveForwardName("");
        }
    };

    const stopTraining = async () => API.stopTraining();
    const formatYAxis = (val) => val === 0 ? "0" : val.toFixed(2);

    const modelName = getExpectedForwardName(dataRoot);
    const modelMeta = getFileMeta(modelName);

    return (
        <div className="glass-panel p-2 flex flex-col gap-2 h-full">
            <h2 className="text-lg font-semibold flex items-center gap-2 border-b border-indigo-100 pb-1 text-slate-800">
                <Activity className="text-orange-500" size={18} />Forward Modeling</h2>

            <div className="flex flex-col gap-2">
                <div className="grid grid-cols-3 gap-1">
                    {/* Epochs */}
                    <div className="flex flex-col gap-0.5">
                        <span className="text-[9px] text-slate-400 font-bold uppercase">Epochs</span>
                        <input type="number" min="1" max="1000" value={forwardEpochs} onChange={(e) => setForwardEpochs(e.target.value)} className="w-full bg-white/50 border border-slate-200 rounded px-1 py-0.5 text-center text-[10px] font-mono focus:outline-none focus:border-orange-400 text-slate-700" disabled={isForwardTraining} />
                    </div>

                    {/* Batch Size */}
                    <div className="flex flex-col gap-0.5">
                        <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Batch</span>
                        <select value={forwardBatchSize} onChange={(e) => setForwardBatchSize(e.target.value)} className="w-full bg-white/50 border border-slate-200 rounded px-1 py-0.5 text-[10px] font-mono focus:outline-none focus:border-orange-400 text-slate-700 bg-transparent" disabled={isForwardTraining}>
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
                        <select value={forwardLearningRate} onChange={(e) => setForwardLearningRate(e.target.value)} className="w-full bg-white/50 border border-slate-200 rounded px-1 py-0.5 text-[10px] font-mono focus:outline-none focus:border-orange-400 text-slate-700 bg-transparent" disabled={isForwardTraining}>
                            <option value="0.01">1e-2</option>
                            <option value="0.001">1e-3</option>
                            <option value="0.0005">5e-4</option>
                            <option value="0.0001">1e-4</option>
                            <option value="0.00001">1e-5</option>
                        </select>
                    </div>

                    {/* Approach Setup */}
                    <div className="flex flex-col gap-0.5 col-span-3">
                        <div className="flex flex-col gap-0.5 mt-1">
                            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Training Logic</span>
                            <select value={forwardApproach} onChange={(e) => setForwardApproach(e.target.value)} className="w-full bg-white/50 border border-slate-200 rounded px-1 py-0.5 text-[10px] font-mono focus:outline-none focus:border-orange-400 text-slate-700 bg-transparent" disabled={isForwardTraining}>
                                <option value="mse">Standard Generative (MSE)</option>
                                <option value="weighted">Action-Weighted Generative (MSE)</option>
                                <option value="infonce">Contrastive Discriminative (InfoNCE)</option>
                                <option value="rnn">Temporal Context (GRU/LSTM)</option>
                                <option value="latentslam">Joint Forward-Observation (LatentSLAM)</option>
                            </select>
                        </div>
                    </div>
                </div>

                {forwardApproach === 'latentslam' && (
                    <div className="flex flex-col gap-0.5 mt-1 p-2 bg-orange-50/50 rounded border border-orange-200">
                        <span className="text-[9px] text-orange-600 font-bold uppercase mb-1">LatentSLAM Joint Configuration (Inheriting Phase 1 [{pipelineArchitecture?.toUpperCase()}] • {vaeLatentDim}d • {vaeImageSize}px)</span>
                        <div className="grid grid-cols-3 gap-2">
                        {/* Transition Loss Weight */}
                        <div className="flex flex-col gap-0.5">
                            <span className="text-[8px] text-orange-500 font-bold uppercase tracking-wider" title="Prioritize Functional Next-State Consistency over Pixel Reconstruction">Trans Wht</span>
                            <select value={transitionLossWeight} onChange={(e) => setTransitionLossWeight(e.target.value)} className="w-full bg-white border border-orange-200 rounded px-1 py-0.5 text-[9px] font-mono focus:outline-none focus:border-orange-400 text-orange-700" disabled={isForwardTraining}>
                                <option value="10.0">x10.0</option>
                                <option value="5.0">x5.0</option>
                                <option value="2.0">x2.0</option>
                                <option value="1.0">x1.0</option>
                                <option value="0.5">x0.5</option>
                                <option value="0.1">x0.1</option>
                            </select>
                        </div>
                        {/* Contrastive Weight */}
                        <div className="flex flex-col gap-0.5">
                            <span className="text-[8px] text-emerald-600 font-bold uppercase tracking-wider" title="InfoNCE Topological Disentanglement">Contrastive</span>
                            <select value={contrastiveWeight} onChange={(e) => setContrastiveWeight(e.target.value)} className="w-full bg-white border border-emerald-200 rounded px-1 py-0.5 text-[9px] font-mono focus:outline-none focus:border-emerald-400 text-emerald-700" disabled={isForwardTraining}>
                                <option value="0.0">0.0</option>
                                <option value="0.1">0.1</option>
                                <option value="0.5">0.5</option>
                                <option value="1.0">1.0</option>
                            </select>
                        </div>
                        {/* LatentSLAM Beta */}
                        <div className="flex flex-col gap-0.5">
                            <span className="text-[8px] text-purple-600 font-bold uppercase tracking-wider" title="New KL Weight">Joint Beta</span>
                            <select value={vaeBeta} onChange={(e) => setVaeBeta(e.target.value)} className="w-full bg-white border border-purple-200 rounded px-1 py-0.5 text-[9px] font-mono focus:outline-none focus:border-purple-400 text-purple-700" disabled={isForwardTraining}>
                                <option value="4.0">4.0</option>
                                <option value="2.0">2.0</option>
                                <option value="1.0">1.0</option>
                                <option value="0.5">0.5</option>
                            </select>
                        </div>
                        </div>
                    </div>
                )}

                {!isForwardTraining ? (
                    <button onClick={startTraining} className="w-full bg-gradient-to-r from-orange-500 to-amber-600 text-white py-1.5 rounded font-bold flex justify-center items-center gap-2 shadow-sm text-xs">
                        <Play size={14} /> Train Forward Model
                    </button>
                ) : (
                    <button onClick={stopTraining} className="w-full bg-red-500 text-white py-1.5 rounded font-bold flex justify-center items-center gap-2 shadow-sm text-xs">
                        <Square size={14} /> Stop
                    </button>
                )}
            </div>

            <div className="h-28 bg-white/60 rounded border border-slate-200/50 p-1 relative">
                <span className="absolute top-1 right-2 text-[8px] font-bold text-orange-500 uppercase">Kinetic Error (Loss)</span>
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={forwardLossHistory}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis type="number" dataKey="epoch" domain={[0, parseInt(forwardEpochs) || 10]} stroke="#94a3b8" fontSize={9} tickFormatter={(val) => val % 1 === 0 ? val : val.toFixed(1)} />
                        <YAxis domain={['auto', 'auto']} fontSize={9} tickFormatter={formatYAxis} width={35} />
                        <Line type="monotone" dataKey="loss" stroke="#f97316" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div className="h-[250px] flex-none bg-white/40 rounded border border-slate-200/50 flex items-center justify-center p-2">
                {policyHeatmap ? (
                    <img src={policyHeatmap} alt="Evaluation Plot" className="max-h-full object-contain mix-blend-multiply" />
                ) : (
                    <span className="text-xs text-slate-400">Algorithmic Baseline Parity Plot (Train to generate)</span>
                )}
            </div>

            <div className="mt-auto pt-1 border-t border-indigo-50 text-[10px] text-slate-500">
                <div className="flex justify-between items-center h-5">
                    <span className="font-semibold">Model:</span>
                    {activeForwardName || (isForwardTraining ? "Generating..." : "") ? (
                        <span 
                            className="font-mono bg-orange-50 text-orange-700 px-1 py-0.5 rounded select-all border border-orange-100 cursor-pointer hover:bg-orange-100 transition-colors"
                            onClick={() => { if (handleLoadLatestForward) handleLoadLatestForward(); }}
                            title="Click to reload and regenerate latest Forward dashboard"
                        >
                            {activeForwardName || "Generating..."}
                        </span>
                    ) : (
                        <span 
                            className="text-slate-400 italic cursor-pointer hover:text-orange-500 hover:underline transition-colors"
                            onClick={() => { if (handleLoadLatestForward) handleLoadLatestForward(); }}
                            title="Click to load latest model"
                        >
                            Empty - click to load latest or start training
                        </span>
                    )}
                </div>
                {modelMeta && !isForwardTraining && (
                    <div className="flex justify-between text-[9px] text-slate-400 mt-0.5">
                        <span className="text-emerald-500 font-bold">FOUND</span>
                        <span>{modelMeta.size_mb} MB</span>
                    </div>
                )}
            </div>
        </div>
    );
};
export default ForwardModelPanel;
