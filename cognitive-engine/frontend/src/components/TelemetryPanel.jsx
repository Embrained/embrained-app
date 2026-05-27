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

import { LineChart, Line, YAxis, ResponsiveContainer, ComposedChart, Scatter, Bar, Tooltip } from 'recharts';
import { Brain, Activity } from 'lucide-react';

const GoalDash = (props) => {
    const { cx, cy, width } = props;
    const w = 6; // Fixed width for dash
    return <line x1={cx - w} y1={cy} x2={cx + w} y2={cy} stroke="#ef4444" strokeWidth={2} />;
};

const TelemetryPanel = ({ data, history }) => {
    return (
        <div className="flex-1 bg-white border border-gray-200 rounded p-4 flex flex-col gap-4 overflow-y-auto shadow-sm h-full">
            {/* Neural State */}
            <div className="flex-1 bg-gray-50 rounded border border-gray-200 p-2 flex flex-col gap-2 relative overflow-hidden">
                <span className="text-xs text-gray-500 flex items-center gap-2">
                    <Brain size={12} /> NEURAL STATE
                </span>
                <div className="flex flex-col gap-3 px-1">
                    <div className="flex flex-col">
                        <span className="text-gray-600 font-semibold text-xs uppercase tracking-wider">Variational Autoencoder</span>
                        <span className="text-blue-600 font-bold text-sm">{data.bvae_model}</span>
                    </div>
                    <div className="flex flex-col">
                        <span className="text-gray-600 font-semibold text-xs uppercase tracking-wider">Conservative Q-Learning</span>
                        <span className="text-blue-600 font-bold text-sm">{data.cql_model}</span>
                    </div>
                </div>
            </div>

            {/* Distance Display (Text Only) */}
            <div className="flex-1 bg-gray-50 rounded border border-gray-200 p-4 flex flex-col items-center justify-center gap-2">
                <span className="text-xs text-gray-500 tracking-wider">DISTANCE TO GOAL</span>
                <span className="text-4xl font-bold font-mono text-gray-800">
                    {data.distance.toFixed(3)}m
                </span>
            </div>


        </div>
    );
};

export default TelemetryPanel;
