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


const API_BASE = "";

export const API = {
    async post(endpoint, data) {
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        return res.json();
    },

    async fetchDatasets(path) {
        const query = path ? `?path=${encodeURIComponent(path)}` : "";
        const res = await fetch(`${API_BASE}/datasets${query}`);
        return res.json();
    },

    async fetchTrainingFiles(path) {
        const query = path ? `?path=${encodeURIComponent(path)}` : "";
        const res = await fetch(`${API_BASE}/training/files${query}`);
        return res.json();
    },

    async browseFolder() {
        const res = await fetch(`${API_BASE}/api/browse`, { method: "POST" });
        return res.json();
    },

    async processDatasets(datasets, rootPath) {
        const res = await fetch(`${API_BASE}/training/process`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                datasets,
                root_path: rootPath
            })
        });
        return res.json();
    },

    async trainVAE(params) {
        // params: { num_epochs, batch_size, learning_rate, model_size, root_path }
        const res = await fetch(`${API_BASE}/training/train_vae`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params)
        });
        return res.json();
    },

    async trainForward(params) {
        // params: { num_epochs, batch_size, learning_rate, forward_approach, root_path, selected_datasets, model_filename }
        const res = await fetch(`${API_BASE}/training/train_forward`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params)
        });
        return res.json();
    },

    async stopTraining() {
        await fetch(`${API_BASE}/training/stop`, { method: "POST" });
    },

    async verifyManifold(params) {
        // params: { force, only_cache, model_filename, dataset, root_path }
        const res = await fetch(`${API_BASE}/training/verify_manifold`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                force: params.force,
                only_cache: params.only_cache,
                model_filename: params.model_filename,
                dataset: params.dataset,
                root_path: params.root_path
            })
        });
        return res.json();
    },

    async verifyForwardModel(params) {
        // params: { model_filename, vae_filename, approach, root_path }
        const res = await fetch(`${API_BASE}/training/verify_forward_model`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params)
        });
        return res.json();
    },

    async visualizePolicy(params) {
        // params: { path, model }
        const queryParams = new URLSearchParams();
        if (params.path) queryParams.append("path", params.path);
        if (params.model) queryParams.append("model", params.model);

        const res = await fetch(`${API_BASE}/training/visualize_policy?${queryParams.toString()}`);
        return res.json();
    }
};
