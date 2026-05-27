# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import logging
from sklearn.decomposition import PCA

logger = logging.getLogger("LatentSLAMService")

class ExperienceMap:
    def __init__(self):
        self.nodes = [] # List of 32D latent vectors (mu)
        self.edges = [] # List of tuples (source_index, target_index)
        self.current_node_index = -1
        self.prior_latent = None # Ghost node: Prior prediction
        self.posterior_latent = None # Ghost node: Posterior correction

    def add_node(self, latent, index=None):
        if index is None:
            self.nodes.append(latent)
            return len(self.nodes) - 1
        else:
            # For fixed-size maps or updates
            if index < len(self.nodes):
                self.nodes[index] = latent
            else:
                self.nodes.append(latent)
            return index

    def add_edge(self, source, target):
        self.edges.append((source, target))

class LatentSLAMService:
    def __init__(self):
        self.experience_map = ExperienceMap()
        self.pca = PCA(n_components=2)
        self.is_fitted = False

    def update_map(self, nodes, edges, current_index, prior=None, posterior=None):
        """
        Updates the internal ExperienceMap structure with full data.
        """
        self.experience_map.nodes = nodes
        self.experience_map.edges = edges
        self.experience_map.current_node_index = current_index
        self.experience_map.prior_latent = prior
        self.experience_map.posterior_latent = posterior
        
        # Re-fit PCA
        if len(nodes) >= 2:
            try:
                self.pca.fit(np.array(nodes))
                self.is_fitted = True
            except Exception as e:
                logger.error(f"PCA fit failed: {e}")

    def update_live_state(self, latent_mu, prior_mu=None):
        """
        Helper for live updates during robot navigation.
        Adds/Updates nodes and ghost predictions.
        """
        # Add new node for current posterior
        self.experience_map.nodes.append(latent_mu)
        self.experience_map.current_node_index = len(self.experience_map.nodes) - 1
        
        # Add edge from previous
        if len(self.experience_map.nodes) > 1:
            self.experience_map.edges.append((len(self.experience_map.nodes)-2, len(self.experience_map.nodes)-1))
            
        self.experience_map.posterior_latent = latent_mu
        self.experience_map.prior_latent = prior_mu
        
        # Limit nodes for performance?
        if len(self.experience_map.nodes) > 500:
            self.experience_map.nodes = self.experience_map.nodes[-500:]
            # Note: edges would need re-indexing, but for live viz 500 is plenty.
        
        # Re-fit PCA every N steps
        if len(self.experience_map.nodes) % 5 == 0:
            try:
                self.pca.fit(np.array(self.experience_map.nodes))
                self.is_fitted = True
            except: pass

    def get_graph_payload(self):
        """
        Projects the 32D nodes down to 2D and returns a JSON-compatible payload.
        """
        if not self.is_fitted or len(self.experience_map.nodes) < 2:
            return {
                "nodes": [],
                "edges": [],
                "active_node": -1
            }

        try:
            nodes_2d = self.pca.transform(np.array(self.experience_map.nodes))
            
            payload_nodes = []
            for i, coord in enumerate(nodes_2d):
                payload_nodes.append({
                    "id": i,
                    "x": float(coord[0]),
                    "y": float(coord[1])
                })

            payload_edges = []
            for source, target in self.experience_map.edges:
                payload_edges.append({
                    "source": source,
                    "target": target
                })

            payload = {
                "nodes": payload_nodes,
                "edges": payload_edges,
                "active_node": self.experience_map.current_node_index
            }

            # Ghost Node Logic
            if self.experience_map.prior_latent is not None and self.experience_map.posterior_latent is not None:
                combined = np.array([self.experience_map.prior_latent, self.experience_map.posterior_latent])
                ghost_2d = self.pca.transform(combined)
                payload["ghost"] = {
                    "prior": {"x": float(ghost_2d[0, 0]), "y": float(ghost_2d[0, 1])},
                    "posterior": {"x": float(ghost_2d[1, 0]), "y": float(ghost_2d[1, 1])}
                }

            return payload
        except Exception as e:
            logger.error(f"Graph projection failed: {e}")
            return {"nodes": [], "edges": [], "active_node": -1}
