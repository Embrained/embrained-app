import React, { useEffect, useRef } from 'react';

interface Node {
    id: number;
    x: number;
    y: number;
}

interface Edge {
    source: number;
    target: number;
}

interface Ghost {
    prior: { x: number; y: number };
    posterior: { x: number; y: number };
}

interface LatentGraphProps {
    graphData: {
        nodes: Node[];
        edges: Edge[];
        active_node: number;
        ghost?: Ghost;
    };
}

const LatentGraph: React.FC<LatentGraphProps> = ({ graphData }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const { nodes, edges, active_node, ghost } = graphData;
        const width = canvas.width;
        const height = canvas.height;

        ctx.clearRect(0, 0, width, height);

        if (nodes.length === 0) return;

        // Determine bounds
        const xs = nodes.map(n => n.x);
        const ys = nodes.map(n => n.y);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);

        const pad = 0.1;
        const xRange = Math.max(maxX - minX, 0.0001) * (1 + pad * 2);
        const yRange = Math.max(maxY - minY, 0.0001) * (1 + pad * 2);

        const mapX = (x: number) => ((x - (minX - pad * (maxX - minX))) / xRange) * width;
        const mapY = (y: number) => height - ((y - (minY - pad * (maxY - minY))) / yRange) * height;

        // Draw Edges
        edges.forEach(edge => {
            const source = nodes.find(n => n.id === edge.source);
            const target = nodes.find(n => n.id === edge.target);

            if (source && target) {
                const sx = mapX(source.x);
                const sy = mapY(source.y);
                const tx = mapX(target.x);
                const ty = mapY(target.y);

                // Loop closure detection: if indices are far apart but nodes are connected
                const distIndex = Math.abs(edge.source - edge.target);
                const isLoopClosure = distIndex > 50; // Heuristic for loop closure

                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.lineTo(tx, ty);
                ctx.strokeStyle = isLoopClosure ? 'rgba(234, 179, 8, 0.8)' : 'rgba(148, 163, 184, 0.4)'; // Yellow vs Slate
                ctx.lineWidth = isLoopClosure ? 2 : 1;
                ctx.stroke();
            }
        });

        // Draw Nodes
        nodes.forEach(node => {
            const nx = mapX(node.x);
            const ny = mapY(node.y);
            const isActive = node.id === active_node;

            ctx.beginPath();
            ctx.arc(nx, ny, isActive ? 5 : 3, 0, 2 * Math.PI);
            ctx.fillStyle = isActive ? '#06b6d4' : '#64748b'; // Cyan vs Slate 500
            ctx.fill();

            if (isActive) {
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        });

        // Draw Ghost Node if present
        if (ghost) {
            const px = mapX(ghost.prior.x);
            const py = mapY(ghost.prior.y);
            const ox = mapX(ghost.posterior.x);
            const oy = mapY(ghost.posterior.y);

            // Prior Ghost (Transparent Gray)
            ctx.beginPath();
            ctx.arc(px, py, 4, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(100, 116, 139, 0.3)';
            ctx.fill();

            // Dotted line for "Surprisal"
            ctx.beginPath();
            ctx.setLineDash([5, 5]);
            ctx.moveTo(px, py);
            ctx.lineTo(ox, oy);
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.5)';
            ctx.stroke();
            ctx.setLineDash([]);
        }

    }, [graphData]);

    return (
        <div className="w-full h-full bg-slate-50/50 rounded-lg overflow-hidden border border-slate-200">
            <canvas
                ref={canvasRef}
                width={400}
                height={400}
                className="w-full h-full block"
            />
        </div>
    );
};

export default LatentGraph;
