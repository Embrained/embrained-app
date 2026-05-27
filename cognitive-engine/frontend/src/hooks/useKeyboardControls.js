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

import { useEffect, useRef } from 'react';

export const useKeyboardControls = (sendMessage) => {
    const ledStateRef = useRef({ r: 0, g: 0, b: 0 });
    const activeKeyRef = useRef(null);

    useEffect(() => {
        const handleKeyDown = (e) => {
            const key = e.key.toLowerCase();
            // Prevent default scrolling for w/a/s/d/space
            if (['w', 'a', 's', 'd', ' ', '0'].includes(key)) {
                e.preventDefault();
            }

            // Ignore key repeats to prevent rapid over-acceleration
            if (e.repeat) return;

            let cmd = null;
            let isMoveKey = false;
            switch (key) {
                case 'w': cmd = 1; isMoveKey = true; break; // Forward -> ID 1
                case 's': cmd = 2; isMoveKey = true; break; // Backward -> ID 2
                case 'a': cmd = 3; isMoveKey = true; break; // Hard Left -> ID 3
                case 'd': cmd = 4; isMoveKey = true; break; // Hard Right -> ID 4
                case ' ': cmd = 5; isMoveKey = true; break; // Intentional Stop -> ID 5
                case '0': cmd = 0; break; // Stop -> ID 0
                default: break;
            }

            if (cmd !== null) {
                if (isMoveKey) {
                    activeKeyRef.current = key;
                } else if (cmd === 0) {
                    activeKeyRef.current = null;
                }

                sendMessage('MOVE', cmd);
                // If Stop (0), reset LED ref to match physical shutdown
                if (cmd === 0) {
                    ledStateRef.current = { r: 0, g: 0, b: 0 };
                }
            }

            // Sound Keys
            const soundMap = { 'u': 261, 'i': 329, 'o': 392, 'p': 523 };
            if (soundMap[key]) {
                sendMessage('SOUND', soundMap[key]);
            }

            // LED Keys (R, G, B toggle)
            if (['r', 'g', 'b'].includes(key)) {
                const current = ledStateRef.current;

                if (key === 'r') current.r = current.r ? 0 : 255;
                if (key === 'g') current.g = current.g ? 0 : 255;
                if (key === 'b') current.b = current.b ? 0 : 255;

                sendMessage('LED', [current.r, current.g, current.b]);
            }
        };

        const handleKeyUp = (e) => {
            const key = e.key.toLowerCase();
            if (['w', 'a', 's', 'd'].includes(key)) {
                if (activeKeyRef.current === key) {
                    activeKeyRef.current = null;
                    sendMessage('MOVE', 0);
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        window.addEventListener('keyup', handleKeyUp);

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
        };
    }, [sendMessage]);
};
