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
        is_recording: false
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
