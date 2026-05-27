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

import { useState, useEffect, useRef, useCallback } from 'react';

export function useRobotConnection(url) {
    const [data, setData] = useState({
        image: null,
        goal_image: null,
        action: 'STOP',
        distance: 0,
        goal_idx: 0,
        led_color: 'N/A',
        mode: 'LIVE',
        fps: 0,
        bvae_model: 'N/A',
        cql_model: 'N/A',
        current_latent: [],
        goal_latent: [],
        is_recording: false,
        match_image: null,
        match_dist: 0,
        match_name: 'N/A'
    });

    const [history, setHistory] = useState([]);
    const [connected, setConnected] = useState(false);
    const socketRef = useRef(null);
    const maxHistory = 50;

    useEffect(() => {
        const connect = () => {
            const socket = new WebSocket(url);
            socketRef.current = socket;

            socket.onopen = () => setConnected(true);
            socket.onclose = () => {
                setConnected(false);
                setTimeout(connect, 1000); // Reconnect
            };

            socket.onmessage = (event) => {
                try {
                    const packet = JSON.parse(event.data);
                    setData(packet);

                    // Update History for Plot
                    setHistory(prev => {
                        const next = [...prev, { t: Date.now(), d: packet.distance }];
                        if (next.length > maxHistory) next.shift();
                        return next;
                    });
                } catch (e) {
                    console.error("Parse Error", e);
                }
            };
        };

        connect();
        return () => socketRef.current?.close();
    }, [url]);

    const sendMessage = useCallback((type, payload) => {
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
            socketRef.current.send(JSON.stringify({ type, payload }));
        }
    }, []);

    return { data, history, connected, sendMessage };
}
