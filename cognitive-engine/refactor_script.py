import sys

with open('backend/engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

start_idx = content.find("    def _decide(self, current_mode, z_cur, img=None, last_action_sent=(0, 0)):")
end_idx = content.find("    def _act(self, current_mode, target_action, last_action_sent, last_sent_time):")

if start_idx == -1 or end_idx == -1:
    print("Could not find boundaries.")
    sys.exit(1)

# Helper definitions
new_methods = """    def _extract_explicit_state(self, last_action_sent):
        curr_l, curr_r = 0.0, 0.0
        if last_action_sent is not None and len(last_action_sent) == 2:
             curr_l = float(last_action_sent[0])
             curr_r = float(last_action_sent[1])
             
        best_action = 0
        best_dist = float('inf')
        import math
        for act_id, (map_l, map_r) in ACTION_PWM_MAP.items():
            diff = math.hypot(curr_l - map_l, curr_r - map_r)
            if diff < best_dist:
                best_dist = diff
                best_action = act_id
                
        MAX_ACTION = 4.0
        action_norm = float(best_action) / MAX_ACTION
        
        curr_sonar = 0.0
        with self.state_lock:
             try: 
                 curr_sonar = float(self.state.get('sensor_dist', 0.0))
             except Exception: pass
                 
        dist_norm = curr_sonar / 1024.0
        state_vec = np.array([action_norm, dist_norm], dtype=np.float32)
        return state_vec, curr_sonar

    def _decide_cql_policy(self, z_cur, state_vec, curr_sonar):
        action, dist, eff_thresh, goal_idx, active_goal_dict, reflex_triggered = self.planner.decide(
            z_cur, state_vec=state_vec, dist_threshold=self.stop_threshold
        )
        target_action = 0
        is_bout_start = False
        
        if self.active_model_name and '_markov_control' in self.active_model_name:
            if action != 5 and not reflex_triggered:
                import random
                action = random.choices([1, 2, 3, 4], weights=[0.6, 0.1, 0.15, 0.15], k=1)[0]
        
        prev_state = getattr(self.cql_controller, 'state', None)
        if reflex_triggered or action == 5:
            self.cql_controller.state = 'WAITING'
            self.cql_controller.state_start_time = 0
            self.cql_controller.current_action_id = 0
            if reflex_triggered:
                target_action = 0
        else:
            effective_sonar = curr_sonar if self.reflex_enabled else 0.0
            if self.active_model_name and 'fixed_goal' in self.active_model_name:
                effective_sonar = curr_sonar
                
            if self.active_model_name and '_oracle_control' in self.active_model_name:
                try:
                    import os
                    coords_path = self.active_model_path.replace("_oracle_control.pth", "_oracle_coords.npy")
                    feats = getattr(self, 'live_telemetry_cache', None)
                    if os.path.exists(coords_path) and feats is not None:
                        oracle_coords = np.load(coords_path)
                        t_arr = np.array([feats['cx_norm'], feats['cy_norm'], feats['cos_yaw'], feats['sin_yaw']], dtype=np.float32)
                        
                        if getattr(self, 'explorer', None) and getattr(self.explorer, 'algo_oracle', None):
                            if action == 5:
                                target_action = self.cql_controller.get_action(effective_sonar, teleop_action=5)
                            else:
                                target_action = self.explorer.algo_oracle.get_action(
                                    sensor_dist=effective_sonar, teleop_action=action, z_cur=t_arr, z_goal=oracle_coords
                                )
                                self.cql_controller.state = self.explorer.algo_oracle.pacer.state
                        else:
                            target_action = self.cql_controller.get_action(effective_sonar, teleop_action=action)
                    else:
                        target_action = self.cql_controller.get_action(effective_sonar, teleop_action=action)
                except Exception as e:
                    logging.error(f"Algorithmic Oracle bypass failed: {e}")
                    target_action = self.cql_controller.get_action(effective_sonar, teleop_action=action)
            else:
                target_action = self.cql_controller.get_action(effective_sonar, teleop_action=action)
        
        new_state = getattr(self.cql_controller, 'state', None)
        if new_state == 'MOVE' and prev_state in ['STOP', 'WAITING']:
            is_bout_start = True
            
        return target_action, dist, goal_idx, reflex_triggered, is_bout_start

    def _decide_latent_slam(self, z_cur, curr_sonar):
        target_action = 0
        dist = 0.0
        with self.state_lock:
            goals = self.state.get('goal_slam_latents', [])
            if goals and len(goals) > 0:
                z_goal = np.array(goals[self.state.get('goal_idx', 0)]).squeeze()
                latent_dim = self.slam_inference.model.latent_dim
                if z_goal.shape[0] > latent_dim:
                    z_goal = z_goal[:latent_dim]
                    
                z_cur_slam = z_cur.squeeze()
                if z_cur_slam.shape[0] > latent_dim:
                    z_cur_slam = z_cur_slam[:latent_dim]

                alpha = 0.3
                if not hasattr(self, 'slam_z_smoothed') or self.slam_z_smoothed is None:
                    self.slam_z_smoothed = z_cur_slam.copy()
                else:
                    self.slam_z_smoothed = alpha * z_cur_slam + (1.0 - alpha) * self.slam_z_smoothed

                dist = float(np.linalg.norm(self.slam_z_smoothed - z_goal))
                
                if dist < self.stop_threshold:
                    action = 0
                else:
                    best_action = 0
                    min_dist = float('inf')
                    valid_actions = [0, 1, 3, 4]
                    
                    for act_id in valid_actions:
                        pwm = ACTION_PWM_MAP.get(act_id, (0,0))
                        z_next = self.slam_inference.predict_next_state(pwm)
                        z_next_sq = z_next.squeeze()
                        if z_next_sq.shape[0] > latent_dim:
                            z_next_sq = z_next_sq[:latent_dim]
                            
                        d = float(np.linalg.norm(z_next_sq - z_goal))
                        if d < min_dist:
                            min_dist = d
                            best_action = act_id
                            
                    action = best_action
                target_action = self.cql_controller.get_action(curr_sonar, teleop_action=action)
        return target_action, dist

    def _decide_exploration(self, current_mode, z_cur, img, dist):
        target_action = 0
        is_bout_start = False
        
        if getattr(self, 'dreamer_ctrl', None) and current_mode == 'INFERENCE':
            target_action = self.dreamer_ctrl.get_action(img=img, latent=z_cur)
        elif self.policy_server and current_mode == 'INFERENCE':
            target_action = self.vla.get_action(img)
        elif self.explorer and self.explorer.current_algo:
            try:
                s_dist = float(self.state.get('sensor_dist', '999'))
            except Exception:
                s_dist = 999.0
            teleop_val = self.current_live_action
            
            algo = self.explorer.current_algo
            if algo == "Markov": ctrl = self.explorer.markov
            elif algo == "MarkovTelemetry": ctrl = getattr(self.explorer, 'markov_telemetry', None)
            elif algo == "MarkovWASD": ctrl = getattr(self.explorer, 'markov_wasd', None)
            elif algo == "MarkovSweep": ctrl = getattr(self.explorer, 'markov_sweep', None)
            elif algo == "Algorithmic Oracle": ctrl = getattr(self.explorer, 'algo_oracle', None)
            elif isinstance(algo, str) and algo.startswith("Neural Oracle"): ctrl = getattr(self.explorer, 'neural_oracle', None)
            else: ctrl = None
            
            prev_state = getattr(ctrl, 'state', None) if ctrl else None
            effector_dist = s_dist if self.reflex_enabled else 0.0
            
            current_goal = None
            current_img_goal = None
            if hasattr(self, 'planner') and self.planner and self.planner.goals:
                if 0 <= self.planner.current_goal_idx < len(self.planner.goals):
                    current_goal = self.planner.goals[self.planner.current_goal_idx].get('latent', None)
                    current_img_goal = self.planner.goals[self.planner.current_goal_idx].get('img', None)
                    
            if current_goal is None and z_cur is not None:
                with self.state_lock:
                    goals = self.state.get('goal_latents', [])
                    idx = self.state.get('goal_idx', 0)
                    if goals and len(goals) > 0 and 0 <= idx < len(goals):
                        current_goal = np.array(goals[idx]).squeeze()
                        
            target_action = self.explorer.get_action(
                sensor_dist=effector_dist, teleop_action=teleop_val, 
                z_cur=z_cur, z_goal=current_goal, 
                img_cur=img, img_goal=current_img_goal, latent_dist=dist
            )
            
            new_state = getattr(ctrl, 'state', None) if ctrl else None
            if new_state == 'MOVE' and prev_state in ['STOP', 'WAITING']:
                is_bout_start = True
        else:
            target_action = self.current_live_action
            
        return target_action, is_bout_start

    def _project_manifold(self, z_cur):
        if self.manifold:
            latent_to_project = None
            if z_cur is not None:
                latent_to_project = z_cur
            elif getattr(self, 'dreamer_ctrl', None) and self.dreamer_ctrl.last_latent is not None:
                latent_to_project = self.dreamer_ctrl.last_latent
            
            if latent_to_project is not None:
                pca_dim = getattr(self.manifold.pca, 'n_features_in_', 32) if hasattr(self.manifold, 'pca') else 32
                if isinstance(latent_to_project, np.ndarray) and latent_to_project.shape[0] > pca_dim:
                    latent_to_project = latent_to_project[:pca_dim]
                    
                coords = self.manifold.project(latent_to_project)
                with self.state_lock:
                    self.state['manifold_coord'] = coords if coords else None
            else:
                with self.state_lock:
                    self.state['manifold_coord'] = None
                    
            m_name = getattr(self.planner, 'model_name', '') if hasattr(self, 'planner') else ''
            fallback_latent = None
            if m_name and ('group-goal' in m_name or 'group_goal' in m_name or 'fixed_goal' in m_name):
                if hasattr(self, 'planner'): fallback_latent = getattr(self.planner, 'mu_goal', None)
            elif getattr(self, 'explorer', None) and self.explorer.current_algo == "Neural Oracle":
                n_oracle = getattr(self.explorer, 'neural_oracle', None)
                if n_oracle: fallback_latent = getattr(n_oracle, 'goal_latent', None)
                
            if fallback_latent is not None:
                with self.state_lock:
                    if not self.state.get('goal_manifold_coords'):
                        if hasattr(fallback_latent, 'detach'):
                            fallback_latent = fallback_latent.detach().cpu().numpy().squeeze()
                        c_coord = self.manifold.project(fallback_latent)
                        if c_coord:
                            self.state['goal_manifold_coords'] = [c_coord]
                            self.state['goal_idx'] = 0

    def _decide(self, current_mode, z_cur, img=None, last_action_sent=(0, 0)):
        target_action = 0 
        dist = 0.0
        goal_idx = 0
        reflex_triggered = False
        is_bout_start = False
        
        state_vec, curr_sonar = self._extract_explicit_state(last_action_sent)

        is_latentslam_active = hasattr(self, 'slam_inference') and self.slam_inference is not None and self.active_model_name and 'latentslam' in self.active_model_name.lower()
        
        if self.active_model_name and self.planner and not is_latentslam_active and not getattr(self, 'telemetry_warmup_active', False):
             if z_cur is not None:
                 target_action, dist, goal_idx, reflex_triggered, is_bout_start = self._decide_cql_policy(z_cur, state_vec, curr_sonar)
        elif is_latentslam_active:
             if z_cur is not None:
                 target_action, dist = self._decide_latent_slam(z_cur, curr_sonar)
        else:
             if z_cur is not None:
                 with self.state_lock:
                     goals = self.state.get('goal_latents', [])
                     if goals and len(goals) > 0:
                         z_goal_tmp = np.array(goals[self.state.get('goal_idx', 0)]).squeeze()
                         dist = float(np.linalg.norm(z_cur.squeeze() - z_goal_tmp))
                     elif getattr(self, 'explorer', None) and self.explorer.current_algo == "Neural Oracle":
                         n_oracle = getattr(self.explorer, 'neural_oracle', None)
                         if n_oracle and getattr(n_oracle, 'goal_latent', None) is not None:
                             try:
                                 dist = float(np.linalg.norm(z_cur.squeeze() - n_oracle.goal_latent.detach().cpu().numpy().squeeze()))
                             except Exception: pass
             
             target_action, is_bout_start = self._decide_exploration(current_mode, z_cur, img, dist)

        if self.reflex_enabled and curr_sonar > AUTONOMY_THRESHOLD:
             reflex_triggered = True

        self._project_manifold(z_cur)

        active_ctrl = self.active_model_name if self.active_model_name else (self.explorer.current_algo if self.explorer else None)
        with self.state_lock:
             self.state['controller'] = active_ctrl
             self.state['fg_eval_phase'] = getattr(self, 'fg_eval_phase', 'MODEL')
             self.state['latent_dist'] = dist
             self.state['latent_thresh'] = self.stop_threshold
             
        return target_action, dist, goal_idx, reflex_triggered, is_bout_start
"""

new_content = content[:start_idx] + new_methods + content[end_idx:]

with open('backend/engine.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
print("Successfully refactored _decide in backend/engine.py")
