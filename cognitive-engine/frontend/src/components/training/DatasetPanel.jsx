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

import React, { useMemo, useState } from 'react';
import { Database, RefreshCw, Play, CheckSquare, Target, ChevronRight, ChevronDown } from 'lucide-react';
import { API } from '../../services/api';

const DatasetPanel = ({
    dataRoot, setDataRoot,
    datasets, setDatasets,
    selectedDatasets, setSelectedDatasets,
    isProcessing, setIsProcessing,
    processResult, setProcessResult,
    isLoadingDatasets, fetchFiles,
    defaultRoot
}) => {
    // State to keep track of expanded groups
    const [expandedGroups, setExpandedGroups] = useState({});


    const handleProcess = async () => {
        setIsProcessing(true);
        setProcessResult(null);
        try {
            const json = await API.processDatasets(selectedDatasets, dataRoot);
            setProcessResult(json);
            fetchFiles(dataRoot);
        } catch (e) {
            setProcessResult({ status: "error", message: e.message });
        }
        setIsProcessing(false);
    };

    const toggleDataset = (name) => {
        setSelectedDatasets(prev =>
            prev.includes(name) ? prev.filter(d => d !== name) : [...prev, name]
        );
    };

    const handleSelectAll = () => {
        if (selectedDatasets.length === datasets.length) {
            setSelectedDatasets([]);
        } else {
            setSelectedDatasets(datasets.map(d => d.name));
        }
    };

    const groupedDatasets = useMemo(() => {
        const groups = {};
        datasets.forEach(d => {
            const groupName = d.name; // Use the exact name provided by the backend without truncation
            if (!groups[groupName]) {
                groups[groupName] = { name: groupName, items: [], totalCount: 0 };
            }
            groups[groupName].items.push(d);
            groups[groupName].totalCount += d.count || 0;
        });
        return Object.values(groups).sort((a, b) => a.name.localeCompare(b.name));
    }, [datasets]);

    const toggleGroupExpand = (groupName) => {
        setExpandedGroups(prev => ({
            ...prev,
            [groupName]: !prev[groupName]
        }));
    };

    const toggleGroupSelection = (groupName, items, e) => {
        e.stopPropagation(); // prevent expanding/collapsing when clicking the checkbox/selection area
        const itemNames = items.map(i => i.name);
        // Check if all items in this group are currently selected
        const allSelected = itemNames.every(name => selectedDatasets.includes(name));

        if (allSelected) {
            // Deselect all in group
            setSelectedDatasets(prev => prev.filter(name => !itemNames.includes(name)));
        } else {
            // Select all in group
            setSelectedDatasets(prev => {
                const newSelection = new Set([...prev, ...itemNames]);
                return Array.from(newSelection);
            });
        }
    };

    return (
        <div className="glass-panel p-2 flex flex-col gap-2 relative h-full">
            <h2 className="text-lg font-semibold flex items-center gap-2 border-b border-indigo-100 pb-1 justify-between text-slate-800">
                <div className="flex items-center gap-2">
                    <Database className="text-cyan-600" size={18} />
                    Dataset Sync
                </div>
                <div className="flex items-center gap-1">
                    {dataRoot !== defaultRoot && (
                        <button
                            onClick={() => setDataRoot(defaultRoot)}
                            className="text-[10px] px-2 py-1 rounded transition-colors font-bold shadow-sm border bg-white border-slate-200 text-slate-400 hover:text-blue-500 hover:border-blue-200"
                            title="Reset to global data dir"
                        >
                            RESET
                        </button>
                    )}
                    <button
                        onClick={handleSelectAll}
                        className="text-xs px-2 py-1 rounded transition-colors font-bold shadow-sm border bg-slate-50 border-slate-200 text-slate-500 hover:bg-slate-100 flex items-center gap-1"
                        title="Select All"
                    >
                        <CheckSquare size={12} />
                        {selectedDatasets.length === datasets.length ? "All" : "All"}
                    </button>
                </div>
            </h2>

            {dataRoot && (
                <div className="text-[10px] text-slate-400 font-mono truncate px-1 bg-slate-50 rounded border border-slate-100 py-0.5">
                    {dataRoot}
                </div>
            )}

            <div className="flex-1 overflow-y-auto min-h-0 bg-white/40 rounded p-2 border border-slate-200/50 relative">
                {isLoadingDatasets ? (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 gap-2 bg-white/60 backdrop-blur-sm z-10">
                        <RefreshCw className="animate-spin text-indigo-500" size={24} />
                        <span className="text-sm font-semibold animate-pulse">Indexing...</span>
                    </div>
                ) : groupedDatasets.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-slate-400 gap-2">
                        <Database size={24} className="opacity-50" />
                        <span className="text-sm font-medium">No Recordings Found</span>
                    </div>
                ) : (
                    groupedDatasets.map(group => {
                        const isExpanded = expandedGroups[group.name];
                        const itemNames = group.items.map(i => i.name);
                        const allSelected = itemNames.every(name => selectedDatasets.includes(name));
                        const someSelected = itemNames.some(name => selectedDatasets.includes(name));

                        return (
                            <div key={group.name} className="mb-1">
                                {/* Group Header */}
                                <div
                                    className={`flex justify-between items-center px-2 py-2 rounded cursor-pointer transition-colors border ${someSelected && !allSelected ? 'bg-indigo-50 border-indigo-200' :
                                            allSelected ? 'bg-blue-100 border-blue-200 shadow-sm' :
                                                'hover:bg-slate-100 border-transparent'
                                        }`}
                                    onClick={() => toggleGroupExpand(group.name)}
                                >
                                    <div className="flex items-center min-w-0 gap-2 flex-1">
                                        <div className="text-slate-400">
                                            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                                        </div>
                                        <div
                                            className="flex items-center mt-0.5 cursor-pointer"
                                            onClick={(e) => toggleGroupSelection(group.name, group.items, e)}
                                        >
                                            {allSelected ? (
                                                <CheckSquare size={14} className="text-blue-600 mr-2" />
                                            ) : someSelected ? (
                                                <div className="w-3.5 h-3.5 rounded-sm bg-indigo-500 text-white flex items-center justify-center mr-2">
                                                    <div className="w-2 h-0.5 bg-white rounded-full"></div>
                                                </div>
                                            ) : (
                                                <div className="w-3.5 h-3.5 rounded-sm border border-slate-300 mr-2"></div>
                                            )}
                                        </div>
                                        <span className={`font-mono text-sm font-bold truncate ${allSelected ? 'text-blue-900' : 'text-slate-700'}`}>
                                            {group.name}*
                                        </span>
                                        <span className="text-[10px] text-slate-400">({group.items.length})</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="text-[10px] text-slate-500 bg-white/60 px-1.5 py-0.5 rounded border border-slate-100 whitespace-nowrap font-medium">
                                            {group.totalCount.toLocaleString()} transitions
                                        </span>
                                    </div>
                                </div>

                                {/* Expanded Items */}
                                {isExpanded && (
                                    <div className="pl-6 pr-1 py-1 flex flex-col gap-1 border-l-2 border-slate-100 ml-3">
                                        {group.items.map(d => (
                                            <div
                                                key={d.name}
                                                className={`flex justify-between items-center px-2 py-1 rounded cursor-pointer transition-colors border text-xs ${selectedDatasets.includes(d.name)
                                                        ? 'bg-blue-50 border-blue-100 text-blue-900'
                                                        : 'hover:bg-slate-50 border-transparent text-slate-600'
                                                    }`}
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    toggleDataset(d.name);
                                                }}
                                            >
                                                <div className="flex items-baseline min-w-0 flex-1 gap-2">
                                                    <span className="font-mono truncate">{d.name}</span>
                                                    <span className="text-[9px] text-slate-400 italic hidden sm:inline">{d.format || 'standard'}</span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <span className="text-[9px] text-slate-400">{d.count}</span>
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            const cleanRoot = dataRoot.replace(/[\/\\]$/, '');
                                                            const newRoot = `${cleanRoot}/${d.name}`.replace(/[\/\\]+/g, '/');
                                                            setDataRoot(newRoot);
                                                        }}
                                                        className="p-1 hover:bg-blue-200 rounded text-blue-500 transition-all"
                                                        title="Set as Focus Training Root"
                                                    >
                                                        <Target size={12} />
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        );
                    })
                )}
            </div>

            <div className="flex flex-col gap-2">
                <button
                    onClick={handleProcess}
                    disabled={isProcessing || selectedDatasets.length === 0}
                    className={`flex items-center justify-center gap-2 py-3 rounded-lg font-bold transition-all shadow-sm ${isProcessing
                        ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                        : 'bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-white shadow-emerald-500/20'
                        }`}
                >
                    {isProcessing ? <RefreshCw className="animate-spin" /> : <Play size={20} />}
                    {isProcessing ? "Processing..." : "Process Selection"}
                </button>
                {processResult && (
                    <div className={`text-xs p-2 rounded border font-medium ${processResult.status === 'success'
                        ? 'bg-green-50 border-green-200 text-green-700'
                        : 'bg-red-50 border-red-200 text-red-700'
                        }`}>
                        {processResult.message || (typeof processResult.detail === 'string' ? processResult.detail : JSON.stringify(processResult.detail)) || `Processed ${processResult.episodes_count} episodes.`}
                    </div>
                )}
            </div>
        </div>
    );
};
export default DatasetPanel;
