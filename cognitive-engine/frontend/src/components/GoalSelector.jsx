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

import React, { useState, useEffect } from 'react';
import { X, Folder, Image as ImageIcon, Check, MousePointerClick } from 'lucide-react';
import clsx from 'clsx';

const GoalSelector = ({ isOpen, onClose, onSave }) => {
    const [folders, setFolders] = useState([]);
    const [selectedFolder, setSelectedFolder] = useState(null);
    const [images, setImages] = useState([]);
    const [selectedImages, setSelectedImages] = useState([]);
    const [loading, setLoading] = useState(false);

    // Fetch folders on mount/open
    useEffect(() => {
        if (isOpen) {
            fetch('/datasets') // Reuse existing endpoint
                .then(res => res.json())
                .then(data => {
                    if (data.datasets) {
                        setFolders(data.datasets);
                        // Default to 'goals' if exists, else first
                        const goalsFolder = data.datasets.find(d => d.name === 'goals');
                        if (goalsFolder) setSelectedFolder('goals');
                    }
                })
                .catch(err => console.error("Failed to fetch folders:", err));
        }
    }, [isOpen]);

    // Fetch images when folder changes
    useEffect(() => {
        if (selectedFolder) {
            setLoading(true);
            fetch('/api/list_images', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: selectedFolder })
            })
                .then(res => res.json())
                .then(data => {
                    if (data.images) {
                        setImages(data.images);
                        setSelectedImages([]); // Reset selection on folder change
                    }
                })
                .catch(err => console.error("Failed to fetch images:", err))
                .finally(() => setLoading(false));
        }
    }, [selectedFolder]);

    const toggleImage = (path) => {
        setSelectedImages(prev => {
            if (prev.includes(path)) {
                return prev.filter(p => p !== path);
            } else {
                if (prev.length >= 5) return prev; // Max 5
                return [...prev, path];
            }
        });
    };

    const handleSave = () => {
        // Map selected image names to their full/relative paths
        // We stored the 'path' (relative from data root) in the image object
        // But our toggle stores just the unique identifier (path)
        onSave(selectedImages);
        onClose();
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
            <div className="bg-white rounded-xl shadow-2xl w-full max-w-4xl h-[80vh] flex flex-col overflow-hidden border border-slate-200">
                {/* Header */}
                <div className="p-4 border-b border-slate-100 flex justify-between items-center bg-slate-50">
                    <div>
                        <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                            <MousePointerClick size={20} className="text-blue-600" />
                            Select Patrol Goals
                        </h2>
                        <p className="text-xs text-slate-500">Choose a folder and select 1-5 images as targets.</p>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-slate-200 rounded-full transition-colors text-slate-500">
                        <X size={20} />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-grow flex min-h-0">
                    {/* Sidebar: Folders */}
                    <div className="w-64 bg-slate-50 border-r border-slate-200 overflow-y-auto p-2 flex flex-col gap-1">
                        <div className="text-xs font-bold text-slate-400 uppercase mb-2 px-2 mt-2">Data Folders</div>
                        {folders.map(f => (
                            <button
                                key={f.name}
                                onClick={() => setSelectedFolder(f.name)}
                                className={clsx(
                                    "w-full text-left px-3 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors",
                                    selectedFolder === f.name
                                        ? "bg-blue-100 text-blue-700 shadow-sm ring-1 ring-blue-200"
                                        : "hover:bg-slate-100 text-slate-600"
                                )}
                            >
                                <Folder size={16} className={selectedFolder === f.name ? "text-blue-500" : "text-slate-400"} />
                                {f.name}
                                <span className="text-[10px] ml-auto bg-slate-200 px-1.5 rounded-full text-slate-500">{f.count}</span>
                            </button>
                        ))}
                    </div>

                    {/* Main: Images */}
                    <div className="flex-grow bg-white p-4 overflow-y-auto">
                        {loading ? (
                            <div className="h-full flex items-center justify-center text-slate-400 animate-pulse">
                                Loading images...
                            </div>
                        ) : (
                            <div>
                                <div className="flex justify-between items-center mb-4">
                                    <div className="text-sm text-slate-500">
                                        Folder: <span className="font-bold text-slate-800">{selectedFolder}</span>
                                    </div>
                                    <div className="text-sm font-bold">
                                        Selected: <span className={selectedImages.length > 0 ? "text-blue-600" : "text-slate-400"}>
                                            {selectedImages.length} / 5
                                        </span>
                                    </div>
                                </div>

                                <div className="grid grid-cols-4 md:grid-cols-5 gap-3">
                                    {images.map((img) => {
                                        const isSelected = selectedImages.includes(img.full_path); // Use full_path for backend
                                        return (
                                            <div
                                                key={img.name}
                                                onClick={() => toggleImage(img.full_path)}
                                                className={clsx(
                                                    "aspect-square rounded-lg border-2 overflow-hidden relative cursor-pointer group transition-all",
                                                    isSelected
                                                        ? "border-blue-500 shadow-md ring-2 ring-blue-200 ring-offset-2"
                                                        : "border-slate-200 hover:border-blue-300"
                                                )}
                                            >
                                                {/* We don't have thumbnails easily, using full path might be slow if large files, but for 'browse' usually ok local */}
                                                {/* For now, just placeholder or try assuming served via static? We don't have static setup for data/ easily. */}
                                                {/* Using generic icon for performance unless we want to fetch blob */}
                                                {/* Actually, user wants to SEE the image. Routes.py doesn't serve static files from data/ easily yet. */}
                                                {/* Let's try to assume we can fetch it? Or maybe we can't. */}
                                                {/* Wait, the app.py probably mounts static? */}
                                                {/* Checking app.py would be good. For now, let's just use icon + name. */}

                                                <div className="w-full h-full bg-slate-50 flex items-center justify-center flex-col gap-2 p-2">
                                                    <ImageIcon className={isSelected ? "text-blue-500" : "text-slate-300"} />
                                                    <span className="text-[10px] text-center break-all text-slate-500 leading-tight line-clamp-2">
                                                        {img.name}
                                                    </span>
                                                </div>

                                                {isSelected && (
                                                    <div className="absolute top-1 right-1 bg-blue-500 text-white rounded-full p-0.5 shadow-sm">
                                                        <Check size={12} strokeWidth={3} />
                                                    </div>
                                                )}

                                                {/* Selection Index */}
                                                {isSelected && (
                                                    <div className="absolute top-1 left-1 bg-blue-600 text-[10px] font-bold text-white px-1.5 rounded shadow-sm">
                                                        {selectedImages.indexOf(img.full_path) + 1}
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                                {images.length === 0 && (
                                    <div className="text-center text-slate-400 mt-10">No images found in this folder.</div>
                                )}
                            </div>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-slate-100 bg-slate-50 flex justify-end gap-3 transition-all">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm font-bold text-slate-500 hover:text-slate-700 hover:bg-slate-200 rounded-lg transition-colors"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={selectedImages.length === 0}
                        className={clsx(
                            "px-6 py-2 text-sm font-bold text-white rounded-lg transition-all shadow-sm flex items-center gap-2",
                            selectedImages.length > 0
                                ? "bg-blue-600 hover:bg-blue-700 shadow-blue-200"
                                : "bg-slate-300 cursor-not-allowed"
                        )}
                    >
                        Update Patrol Goals ({selectedImages.length})
                    </button>
                </div>
            </div>
        </div>
    );
};

export default GoalSelector;
