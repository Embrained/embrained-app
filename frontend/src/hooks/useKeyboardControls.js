import { useEffect, useRef } from 'react';

export const useKeyboardControls = (sendMessage) => {
    const ledStateRef = useRef({ r: 0, g: 0, b: 0 });

    useEffect(() => {
        const handleKeyDown = (e) => {
            // Prevent default scrolling for w/a/s/d/space
            if (['w', 'a', 's', 'd', ' ', '0'].includes(e.key.toLowerCase())) {
                e.preventDefault();
            }

            let cmd = null;
            switch (e.key.toLowerCase()) {
                case 'w': cmd = 0; break; // Forward -> ID 0
                case 'a': cmd = 1; break; // Left -> ID 1
                case 'd': cmd = 2; break; // Right -> ID 2
                case 's': cmd = 4; break; // Backward -> ID 4
                case ' ':
                case '0': cmd = 3; break; // Stop -> ID 3
                default: break;
            }

            if (cmd !== null) {
                sendMessage('MOVE', cmd);
                // If Stop (3), reset LED ref to match physical shutdown
                if (cmd === 3) {
                    ledStateRef.current = { r: 0, g: 0, b: 0 };
                }
            }

            // Sound Keys
            const soundMap = { 'u': 261, 'i': 329, 'o': 392, 'p': 523 };
            if (soundMap[e.key.toLowerCase()]) {
                sendMessage('SOUND', soundMap[e.key.toLowerCase()]);
            }

            // LED Keys (R, G, B toggle)
            if (['r', 'g', 'b'].includes(e.key.toLowerCase())) {
                const key = e.key.toLowerCase();
                const current = ledStateRef.current;

                if (key === 'r') current.r = current.r ? 0 : 255;
                if (key === 'g') current.g = current.g ? 0 : 255;
                if (key === 'b') current.b = current.b ? 0 : 255;

                sendMessage('LED', [current.r, current.g, current.b]);
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [sendMessage]);
};
