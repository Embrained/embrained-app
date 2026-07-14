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

import React, { useState } from 'react';
import VisualizationPanel from './VisualizationPanel';
import GoalPanel from './GoalPanel';
import ActuatorStatus from './ActuatorStatus';
import CameraFeed from './CameraFeed';
import { useKeyboardControls } from '../hooks/useKeyboardControls';
import { Layers, Activity, Cpu, Disc, Square, Wifi, WifiOff, ArrowRight, AlertTriangle } from 'lucide-react';
import clsx from 'clsx';

const StatCard = ({ label, value, unit, icon: Icon, color = "blue" }) => (
    <div className="glass-panel px-2 py-1.5 rounded-lg flex items-center gap-2 bg-white/50 border border-slate-200 shadow-sm overflow-hidden">
        {Icon && <Icon className={`text-${color}-500/70 shrink-0`} size={12} />}
        <div className="flex items-center justify-between min-w-0 flex-grow">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-tight">{label}</span>
            <span className={`text-xs font-mono font-bold text-${color}-600 whitespace-nowrap`}>
                {value}{unit && <span className="text-[10px] text-slate-400 ml-0.5 font-sans">{unit}</span>}
            </span>
        </div>
    </div>
);

const API_BASE = "http://localhost:8000";

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }
    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }
    componentDidCatch(error, errorInfo) {
        console.error("LiveMode Error:", error, errorInfo);
    }
    render() {
        if (this.state.hasError) {
            return (
                <div className="flex items-center justify-center h-full w-full bg-red-50 text-red-800 p-10 flex-col gap-4">
                    <h2 className="text-2xl font-bold">Live UI Crashed</h2>
                    <pre className="text-xs bg-white p-4 rounded border border-red-200 overflow-auto max-w-2xl">
                        {this.state.error?.toString()}
                    </pre>
                    <button
                        onClick={() => { this.setState({ hasError: false }); window.location.reload(); }}
                        className="px-4 py-2 bg-red-600 text-white rounded shadow hover:bg-red-700 font-bold"
                    >
                        Reload Interface
                    </button>
                </div>
            );
        }
        return this.props.children;
    }
}

const getEvalSuffix = (isActive, data) => {
    if (!isActive || !data) return "";
    if (data.telemetry_warmup_active) {
        return ` (WARMUP - ${data.telemetry_init_frames_left} steps left)`;
    }
    return "";
};

const formatModelName = (name, isActive, isOracle, data) => {
    let baseName = name;
    if (isOracle) baseName = 'TELEMETRY ORACLE (XYO)';
    else if (name.includes('-dark-wall-cql_')) baseName = 'CQL (DARK-WALL)_' + name.split('-dark-wall-cql_')[1].replace('.pth', '');
    else if (name.includes('-cql_')) baseName = 'CQL_' + name.split('-cql_')[1].replace('.pth', '');
    else if (name.includes('-fixed_goal_cql_model')) baseName = 'FIXED-GOAL CQL (' + name.split('_')[2] + ')' + name.split('-fixed_goal_cql_model')[1].replace('.pth', '').replace('_', ' v');
    else if (name.includes('-e2e_contrastive_fixed_goal_bc_model.pth')) baseName = 'CONTRASTIVE BC (' + name.split('_')[2] + ')';
    else if (name.includes('-fixed_goal_oracle_control.pth')) baseName = 'ORACLE CONTROL (' + name.split('_')[2] + ')';
    else if (name.includes('-fixed_goal_markov_control.pth')) baseName = 'MARKOV CONTROL (' + name.split('_')[2] + ')';
    else if (name.includes('-fixed_goal_model.pth')) baseName = 'FIXED-GOAL BC (' + name.split('_')[2] + ')';
    else if (name.includes('_seek_cql')) {
        // e.g., cve_32d_...-sofa_seek_cql_model.pth -> SOFA-SEEK CQL
        const seekMatch = name.match(/-(\w+)_seek_cql/);
        if (seekMatch) baseName = seekMatch[1].toUpperCase() + '-SEEK CQL';
        else baseName = 'SEEK CQL';
    }
    
    return baseName + getEvalSuffix(isActive, data);
    
    return name
        .replace('_dreamer.pth', ' (DreamerV3)')
        .replace('-cql.pth', ' (CQL)')
        .replace('-cve', '')
        .replace('.pth', '')
        .replace('topological_forward_', '')
        .replace('tiny_', '')
        .toUpperCase();
};

const LiveMode = ({ data, connected, sendMessage }) => {
    const [models, setModels] = useState([]);
    const [cves, setCves] = useState([]);

    // Movement Timer State
    const [isMoving, setIsMoving] = useState(false);
    const [startTime, setStartTime] = useState(null);
    const [goalImageSrc, setGoalImageSrc] = useState(null);

    React.useEffect(() => {
        if (data.controller && data.controller !== 'N/A' && data.controller.includes('.pth')) {
            setGoalImageSrc(`/api/current_goal_image?t=${Date.now()}`);
        } else {
            setGoalImageSrc(null);
        }
    }, [data.controller]);

    const wrappedSendMessage = React.useCallback((type, payload) => {
        if (type === 'MOVE') {
            const isStop = payload === 0;
            if (!isStop && !isMoving) {
                setIsMoving(true);
                setStartTime(performance.now());
            } else if (isStop && isMoving) {
                setIsMoving(false);
            }
        }
        sendMessage(type, payload);
    }, [isMoving, sendMessage]);

    useKeyboardControls(wrappedSendMessage);

    React.useEffect(() => {
        let raf;
        const tick = () => {
            if (isMoving && startTime) {
                raf = requestAnimationFrame(tick);
            }
        };
        if (isMoving) {
            raf = requestAnimationFrame(tick);
        }
        return () => cancelAnimationFrame(raf);
    }, [isMoving, startTime]);

    React.useEffect(() => {
        fetch('/api/models')
            .then(res => res.json())
            .then(data => {
                if (data.models) {
                    setModels(data.models);
                }
                if (data.cves) {
                    setCves(data.cves);
                }
            })
            .catch(err => console.error("Failed to fetch models:", err));
    }, []);

    // [NEW] Enforce LIVE mode on mount / connection
    React.useEffect(() => {
        if (connected) {
            console.log("LiveMode mounted/connected: Sending SET_MODE -> LIVE");
            sendMessage('SET_MODE', 'LIVE');
        }

        return () => {
            // Cleanup on unmount (only if we are actually leaving the view, 
            // checks handled by App.jsx usually, but redundant safety is good)
            // We don't want to send IDLE if we are just re-rendering, so be careful.
            // React StrictMode mounts/unmounts twice.
            // Rely on App.jsx handleExit for explicit stop, but here we can ensure startup.
        };
    }, [connected, sendMessage]);

    const activeManifoldModel = (data.cve_model && data.cve_model !== "N/A") 
        ? data.cve_model 
        : "N/A";

    // [NEW] Enforce Threshold Scaling — distance is always in continuous z_e space
    React.useEffect(() => {
        if (!connected || !data || data.latent_thresh === undefined || data.latent_thresh === null) return;
        
        // Removed auto-correct to 2.00 since CVE uses normalized latents
    }, [connected, data.latent_thresh, activeManifoldModel, sendMessage]);

    return (
        <ErrorBoundary>
            <div className="fixed inset-0 z-50 flex flex-col h-screen w-full bg-slate-50 overflow-hidden text-slate-900">
                {/* Background Grid */}
                <div className="absolute inset-0 z-0 pointer-events-none"
                    style={{
                        backgroundImage: 'linear-gradient(rgba(0, 0, 0, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 0, 0, 0.03) 1px, transparent 1px)',
                        backgroundSize: '20px 20px'
                    }}>
                </div>

                {/* Main Cockpit Grid */}
                <div className="flex-grow grid grid-cols-12 grid-rows-1 p-4 gap-4 z-10 h-full min-h-0 overflow-hidden">

                    {/* LEFT SIDEBAR - CONTROLS & TELEMETRY (Col 1-3) */}
                    <div className="col-span-3 flex flex-col gap-4 h-full min-h-0 overflow-hidden">

                        {/* 1. Telemetry (Fixed at Top) */}
                        <div className="glass-panel zone-green p-3 rounded-xl shrink-0 flex flex-col gap-2 shadow-sm">
                            <h3 className="text-[10px] font-bold text-green-600/70 uppercase flex items-center gap-2 mb-1 accent-green">
                                <Activity size={12} /> Telemetry
                            </h3>
                            <div className="grid grid-cols-2 gap-2">
                                <StatCard label="FPS" value={data.fps ? data.fps.toFixed(0) : '0'} unit="" icon={Activity} color="green" />
                                <StatCard label="Ping" value={data.ping || '0'} unit="ms" icon={Wifi} color="yellow" />
                                <StatCard label="Distance" value={data.sensor_dist !== undefined && data.sensor_dist !== null ? Math.round(data.sensor_dist) : '0'} unit="" icon={Cpu} color="blue" />
                                <StatCard label="Batt" value={data.sensor_batt || '0'} unit="v" icon={Layers} color="red" />
                            </div>
                        </div>

                        {/* 2. Control Center (Expanded, No Status Panel) */}
                        <div className="glass-panel zone-purple p-3 rounded-xl flex-grow overflow-y-auto min-h-0 flex flex-col gap-4 shadow-sm custom-scrollbar">



                            {/* Autonomy */}
                            <div className="">
                                <h3 className="text-[10px] font-bold text-purple-600/70 uppercase mb-2 flex justify-between items-start accent-purple">
                                    <span>Autonomy</span>
                                </h3>
                                <div className="flex flex-col gap-2">
                                    {/* 1. Telemetry Oracle */}
                                    {(() => {
                                        const isOracleReq = data.controller === 'Algorithmic Oracle' || data.telemetry_source_algo === 'Algorithmic Oracle';
                                        const isWarmupPhase = data.telemetry_warmup_active;
                                        
                                        const isPulsing = isOracleReq && isWarmupPhase;
                                        const isSolid = isOracleReq && !isPulsing;
                                        
                                        const btnClass = isPulsing ? 'bg-orange-500 border-orange-400 text-white shadow-orange-200 animate-pulse' : 
                                                         isSolid ? 'bg-purple-600 border-purple-500 text-white shadow-purple-200' : 
                                                         data.use_webcam ? 'bg-white border-slate-200 text-slate-500 hover:bg-purple-50 hover:border-purple-200 hover:text-purple-600' : 
                                                         'bg-slate-50 border-slate-100 text-slate-300 cursor-not-allowed grayscale';
                                                         
                                        return (
                                            <button
                                                disabled={!data.use_webcam}
                                                title={data.use_webcam ? "Telemetry Oracle" : "Requires Webcam Telemetry"}
                                                onClick={() => wrappedSendMessage('SET_CONTROLLER', isOracleReq ? null : 'Algorithmic Oracle')}
                                                className={`px-2 py-1.5 rounded-lg text-[10px] leading-tight font-bold transition-all border shadow-sm flex items-center gap-1 ${btnClass}`}
                                            >
                                                <span className="break-all flex-grow text-left">{formatModelName('Algorithmic Oracle', isOracleReq, true, data)}</span>
                                            </button>
                                        );
                                    })()}

                                    {/* 2. Markov Randomizer */}
                                    {(() => {
                                        const isMarkov = data.controller === 'Markov';
                                        
                                        const btnClass = isMarkov ? 'bg-purple-600 border-purple-500 text-white shadow-purple-200' :
                                                         'bg-white border-slate-200 text-slate-500 hover:bg-purple-50 hover:border-purple-200 hover:text-purple-600';
                                                         
                                        const btnText = `MARKOV RANDOMIZER${getEvalSuffix(isMarkov, data)}`;

                                        return (
                                            <button
                                                onClick={() => wrappedSendMessage('SET_CONTROLLER', isMarkov ? null : 'Markov')}
                                                className={`px-2 py-1.5 rounded-lg text-[10px] leading-tight font-bold transition-all border shadow-sm flex items-center gap-1 ${btnClass}`}
                                            >
                                                <span className="break-all flex-grow text-left">{btnText}</span>
                                            </button>
                                        );
                                    })()}

                                    {/* 3. MarkovWASD */}
                                    {(() => {
                                        // If telemetry warmup is active, MarkovWASD is internally active, but we shouldn't highlight this manual button
                                        const isMarkovWASD = data.controller === 'MarkovWASD' && !data.telemetry_warmup_active && !data.telemetry_source_algo;
                                        const isWarmupControl = data.telemetry_warmup_active;
                                        
                                        const btnClass = isMarkovWASD ? 'bg-purple-600 border-purple-500 text-white shadow-purple-200' :
                                                         isWarmupControl ? 'bg-orange-500 border-orange-400 text-white shadow-orange-200 animate-pulse' :
                                                         'bg-white border-slate-200 text-slate-500 hover:bg-purple-50 hover:border-purple-200 hover:text-purple-600';
                                                         
                                        const btnText = isWarmupControl ? `MARKOVWASD (TELEOP) (ACTIVE)` : `MARKOVWASD (TELEOP)${getEvalSuffix(isMarkovWASD, data)}`;

                                        return (
                                            <button
                                                onClick={() => wrappedSendMessage('SET_CONTROLLER', isMarkovWASD ? null : 'MarkovWASD')}
                                                className={`px-2 py-1.5 rounded-lg text-[10px] leading-tight font-bold transition-all border shadow-sm flex items-center gap-1 ${btnClass}`}
                                            >
                                                <span className="break-all flex-grow text-left">{btnText}</span>
                                            </button>
                                        );
                                    })()}

                                    {/* 4. CVEs */}
                                    {cves.map((cve) => {
                                        const isActiveCve = data.cve_model === cve.name;
                                        return (
                                            <button
                                                key={cve.name}
                                                onClick={() => {
                                                    wrappedSendMessage('SET_CVE_MODEL', isActiveCve ? null : cve.name);
                                                }}
                                                className={`px-2 py-1.5 rounded-lg text-[10px] leading-tight font-bold transition-all border shadow-sm flex items-center gap-1 ${isActiveCve ? 'bg-emerald-600 border-emerald-500 text-white shadow-emerald-200' : 'bg-white border-slate-200 text-slate-500 hover:bg-emerald-50 hover:border-emerald-200 hover:text-emerald-600'}`}
                                            >
                                                <span className="break-all flex-grow text-left">CVE: {cve.name.replace('.pth', '').replace('_', ' ').toUpperCase()}</span>
                                            </button>
                                        );
                                    })}

                                    {/* 5. CQL Models */}
                                    {models.filter(m => m.name.includes('cql') || m.name.includes('oracle_control')).map((model) => {
                                        const isActive = data.controller === model.name || data.telemetry_source_algo === model.name;
                                        
                                        const isWarmupPhase = isActive && data.telemetry_warmup_active;
                                        
                                        let btnClass = 'bg-white border-slate-200 text-slate-500 hover:bg-purple-50 hover:border-purple-200 hover:text-purple-600';
                                        if (isActive) {
                                            if (isWarmupPhase) {
                                                btnClass = 'bg-orange-500 border-orange-400 text-white shadow-orange-200 animate-pulse';
                                            } else {
                                                btnClass = 'bg-purple-600 border-purple-500 text-white shadow-purple-200';
                                            }
                                        }
                                        
                                        return (
                                            <button
                                                key={model.name}
                                                onClick={() => {
                                                    wrappedSendMessage('SET_CONTROLLER', isActive ? null : model.name);
                                                }}
                                                className={`px-2 py-1.5 rounded-lg text-[10px] leading-tight font-bold transition-all border shadow-sm flex items-center gap-1 ${btnClass}`}
                                            >
                                                <span className="break-all flex-grow text-left">{formatModelName(model.name, isActive, false, data)}</span>
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* CENTER - MAIN VIEW (Col 4-8) */}
                    <div className="col-span-5 flex flex-col h-full min-h-0 overflow-hidden relative">
                        {data.use_webcam ? (
                            <div className="flex-grow grid grid-rows-2 gap-3 h-full min-h-0 relative">
                                <div className="rounded-2xl overflow-hidden glass-panel relative p-1 bg-white/30 border-white/50 shadow-lg flex flex-col h-full min-h-0">

                                    <CameraFeed image={data.webcam_image} />
                                </div>
                                <div className="rounded-2xl overflow-hidden glass-panel relative p-1 bg-white/30 border-white/50 shadow-lg flex flex-col h-full min-h-0">

                                    <CameraFeed
                                        image={data.image}
                                        goalImage={goalImageSrc || data.goal_image}
                                        isRecording={data.is_recording}
                                        recordingFrames={data.recording_frames}
                                    />
                                </div>
                            </div>
                        ) : (
                            <div className="flex-grow w-full rounded-2xl overflow-hidden glass-panel relative p-1 bg-white/30 border-white/50 shadow-lg flex flex-col">
                                <CameraFeed
                                    image={data.image}
                                    goalImage={goalImageSrc || data.goal_image}
                                    isRecording={data.is_recording}
                                    recordingFrames={data.recording_frames}
                                />
                            </div>
                        )}
                    </div>

                    {/* RIGHT SIDEBAR - ANALYTICS (Col 9-12) */}
                    <div className="col-span-4 flex flex-col gap-4 h-full min-h-0 overflow-hidden">

                        {/* Top: Vision System (Header + Visualization) */}
                        <div className="flex-grow min-h-[40%] flex flex-col glass-panel zone-blue rounded-xl shadow-sm overflow-hidden">
                            {/* Header (Matches CvePanel style) */}
                            <div className="px-4 py-3 border-b border-indigo-100/50 flex flex-wrap items-center justify-between shrink-0 bg-white/40 gap-2">
                                <div className="flex items-center gap-2 min-w-0">
                                    <h2 className="text-xs font-bold flex items-center gap-2 text-slate-700 uppercase tracking-wide whitespace-nowrap accent-blue">
                                        <Layers size={14} className="text-purple-600" />
                                        Latent Space
                                    </h2>
                                    {data.cve_model && data.cve_model !== "N/A" && (
                                        <span className="text-[9px] font-mono text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-100 truncate max-w-[300px]" title={data.cve_model}>
                                            {data.cve_model}
                                        </span>
                                    )}
                                </div>

                                <div className="flex items-center gap-3 ml-auto shrink-0">
                                    {/* [NEW] Stop Threshold Dropdown */}
                                    {(data.controller && data.controller !== "N/A") && (() => {
                                        const thresholds = [
                                            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5
                                        ];
                                        
                                        const currentVal = data.latent_thresh !== undefined && data.latent_thresh !== null ? data.latent_thresh : 0.80;
                                        
                                        // Ensure current value is in the dropdown to avoid blank selections
                                        const optionsToRender = [...thresholds];
                                        if (!optionsToRender.includes(currentVal)) {
                                            optionsToRender.push(currentVal);
                                            optionsToRender.sort((a, b) => a - b);
                                        }

                                        return (
                                            <div className="flex items-center gap-2">
                                                <label htmlFor="stop-threshold-select" className="text-[9px] font-bold text-slate-500 uppercase whitespace-nowrap">Stop Thr:</label>
                                                <select
                                                    id="stop-threshold-select"
                                                    value={currentVal.toFixed(2)}
                                                    onChange={(e) => {
                                                        const val = parseFloat(e.target.value);
                                                        wrappedSendMessage('SET_THRESHOLD', val);
                                                    }}
                                                    className="px-2 py-0.5 text-[10px] font-bold text-slate-700 bg-white border border-slate-200 rounded shadow-sm focus:outline-none focus:border-indigo-300 focus:ring-1 focus:ring-indigo-300"
                                                >
                                                    {optionsToRender.map(val => {
                                                        const label = val.toFixed(2);
                                                        return <option key={val} value={label}>{label}</option>;
                                                    })}
                                                </select>
                                            </div>
                                        );
                                    })()}


                                </div>
                            </div>

                            {/* Visualization Content */}
                            <div className="relative flex-grow min-h-0 w-full bg-slate-50/50">
                                {activeManifoldModel !== "N/A" ? (
                                    <VisualizationPanel
                                        keypoints={data.current_latent}
                                        manifoldCoord={data.manifold_coord}
                                        goalCoords={(data.goal_manifold_coords && data.goal_manifold_coords.length > 0) ? data.goal_manifold_coords : (data.active_goal_coord ? [data.active_goal_coord] : [])}
                                        goalIdx={data.goal_idx || 0}
                                        matchCoord={data.match_manifold_coord}
                                        activeCve={activeManifoldModel}

                                        bounds={data.manifold_bounds}
                                        latentDist={data.latent_dist}
                                        latentThreshold={data.latent_thresh}
                                        visualizationMode='default'
                                        onPointClick={(data.controller && (data.controller.includes('fixed_goal') || data.controller.includes('_seek_cql'))) ? undefined : (idx) => wrappedSendMessage('SET_MANIFOLD_GOAL', { index: idx })}
                                    />
                                ) : (
                                    <div className="h-full flex flex-col items-center justify-center text-slate-400 p-4">
                                        <AlertTriangle size={32} className="mb-2 opacity-50 text-amber-500" />
                                        <span className="text-[10px] font-bold uppercase tracking-wider text-amber-600">No CVE Loaded</span>
                                        <span className="text-[9px] opacity-70">A Background CVE is required for latent visualization.</span>
                                    </div>
                                )}
                            </div>
                        </div>



                        {/* Bottom: Motor Output (Compact) */}
                        <div className="shrink-0 glass-panel zone-yellow rounded-xl p-3 overflow-hidden flex flex-col shadow-sm">
                            <h3 className="text-[10px] font-bold text-yellow-600/70 uppercase mb-2 accent-yellow">Motor Output</h3>
                            <ActuatorStatus data={data} />
                        </div>
                        
                        {/* System Controls */}
                        <div className="shrink-0 flex items-center justify-between mt-auto glass-panel zone-orange rounded-xl p-2 shadow-sm">
                            <div className="flex items-center gap-2">
                                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/60 border border-slate-200 shadow-sm backdrop-blur-md">
                                    {connected ? <Wifi size={16} className="text-green-500" /> : <WifiOff size={16} className="text-red-500" />}
                                </div>
                                <button
                                    onClick={async () => {
                                        try { await fetch(data.is_recording ? '/record/stop' : '/record/start', { method: 'POST' }); }
                                        catch (e) { console.error("Record toggle failed", e); }
                                    }}
                                    className={clsx(
                                        "flex items-center gap-2 px-3 py-2 rounded-lg transition-colors uppercase tracking-wider border text-xs font-bold shadow-sm backdrop-blur-md",
                                        data.is_recording ? "bg-red-50 text-red-600 border-red-200 hover:bg-red-100" : "bg-white/60 text-slate-500 border-slate-200 hover:bg-white hover:text-slate-900"
                                    )}
                                >
                                    {data.is_recording ? <Square size={14} fill="currentColor" /> : <Disc size={14} fill="currentColor" />}
                                    REC
                                </button>
                                <button
                                    onClick={() => {
                                        sendMessage('SET_CONTROLLER', null);
                                        sendMessage('MOVE', 0);
                                    }}
                                    className="bg-orange-50 hover:bg-orange-100 text-orange-600 border border-orange-200 px-3 py-2 rounded-lg transition-colors uppercase tracking-wider text-xs font-bold shadow-sm backdrop-blur-md"
                                >
                                    PAUSE
                                </button>
                                <button
                                    onClick={async () => {
                                        try { await fetch('/shutdown', { method: 'POST' }); }
                                        catch (e) { console.error("Shutdown failed", e); }
                                    }}
                                    className="bg-red-50 hover:bg-red-100 text-red-600 border border-red-200 px-3 py-2 rounded-lg transition-colors uppercase tracking-wider text-xs font-bold shadow-sm backdrop-blur-md"
                                >
                                    STOP
                                </button>
                            </div>
                        </div>
                    </div>

                </div>

            </div>
        </ErrorBoundary >
    );
};

export default LiveMode;
