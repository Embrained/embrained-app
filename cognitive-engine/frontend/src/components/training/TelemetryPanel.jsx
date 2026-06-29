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
import { Layers, Play, Square } from 'lucide-react';
import { API } from '../../services/api';

const TelemetryPanel = ({
    dataRoot,
    selectedDatasets,
    activeCveName,
    setActiveCveName,
    cveValidationImage,
    setCveValidationImage,
    fetchFiles
}) => {
    const [isExtracting, setIsExtracting] = useState(false);

    const startExtraction = async () => {
        if (!selectedDatasets || selectedDatasets.length === 0) {
            alert("⚠️ Please select at least one dataset to extract telemetry from.");
            return;
        }

        setIsExtracting(true);
        setCveValidationImage(null);
        
        try {
            const result = await API.post('/training/extract_telemetry', {
                root_path: dataRoot,
                datasets: selectedDatasets
            });
            
            if (result.status === "success") {
                if (result.image) {
                    setCveValidationImage(`data:image/png;base64,${result.image}`);
                }
                // We use the activeCveName to store the master_telemetry state
                setActiveCveName("master_telemetry.csv");
                fetchFiles(dataRoot);
            } else {
                alert(`⚠️ Extraction Failed: ${result.message}`);
            }
        } catch (e) {
            console.error(e);
            alert("⚠️ API Request Failed");
        } finally {
            setIsExtracting(false);
        }
    };

    return (
        <div className="glass-panel p-2 flex flex-col gap-2 h-full">
            <h2 className="text-lg font-semibold flex items-center gap-2 border-b border-indigo-100 pb-1 text-slate-800">
                <Layers className="text-emerald-600" size={18} />
                Ground Truth Telemetry Tracker
            </h2>

            <div className="flex flex-col gap-2 text-sm text-slate-600 mb-2 p-2 bg-emerald-50 rounded border border-emerald-100">
                <p>
                    <strong>Allocentric Extraction Mode:</strong> The neural vision encoder is bypassed. 
                    The physical XY coordinate and Yaw orientation will be analytically extracted from the overhead `webcam_frame` via geometric blob template matching.
                </p>
                <p>
                    Click below to evaluate the selected datasets and compile <code>master_telemetry.csv</code> for the CQL engine to train off of directly.
                </p>
            </div>

            {
                !isExtracting ? (
                    <button onClick={startExtraction} className="w-full bg-gradient-to-r from-emerald-500 to-teal-600 text-white py-2 rounded font-bold flex justify-center items-center gap-2 shadow-sm text-sm hover:from-emerald-400 hover:to-teal-500 transition-colors">
                        <Play size={16} /> Compute Physics Telemetry
                    </button>
                ) : (
                    <button disabled className="w-full bg-slate-300 text-white py-2 rounded font-bold flex justify-center items-center gap-2 shadow-sm text-sm cursor-not-allowed">
                        <Square size={16} /> Computing Matrices (approx 1s/frame)...
                    </button>
                )
            }

            {/* Validation Plot */}
            <div className="flex-1 min-h-[300px] mt-2 bg-white/40 rounded border border-slate-200/50 flex flex-col overflow-hidden relative group">
                {cveValidationImage ? (
                    <img src={cveValidationImage} alt="Telemetry Constraints" className="w-full h-full object-contain p-1" />
                ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-400 text-xs italic">
                        Waiting for physics computation...
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="mt-auto pt-2 border-t border-emerald-50 text-[10px] text-slate-500 flex justify-between">
                <div>
                    <span className="font-semibold">Target File: </span>
                    <span className="font-mono text-emerald-700">master_telemetry.csv</span>
                </div>
                {activeCveName === "master_telemetry.csv" && (
                    <span className="font-bold text-emerald-500">READY</span>
                )}
            </div>
        </div>
    );
};

export default TelemetryPanel;
