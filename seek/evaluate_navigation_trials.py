import os
import sys
import glob
import json
import ast
import pandas as pd
import numpy as np

# Path to data directory
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def get_action(row):
    act_raw = str(row.get('action_id', '0'))
    try:
        return int(act_raw)
    except:
        try:
            tup = ast.literal_eval(act_raw)
            if tup == (0, 0): return 5
            return 1
        except:
            return 1

def evaluate_session(session_dir):
    csv_path = os.path.join(session_dir, 'episode_data.csv')
    if not os.path.exists(csv_path):
        print(f"No episode_data.csv found in {session_dir}")
        return
    
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"Empty dataset in {session_dir}")
        return

    df['macro_action'] = df.apply(get_action, axis=1)
    
    # Clean actions: remove isolated 5s (length 1 stop bouts are mistakes)
    actions = df['macro_action'].tolist()
    cleaned_actions = []
    n = len(actions)
    for i in range(n):
        if actions[i] == 5:
            is_valid_stop = False
            if i > 0 and actions[i-1] == 5: is_valid_stop = True
            if i < n - 1 and actions[i+1] == 5: is_valid_stop = True
            if is_valid_stop:
                cleaned_actions.append(5)
            else:
                cleaned_actions.append(1)
        else:
            cleaned_actions.append(1)
            
    df['cleaned_action'] = cleaned_actions
    df['is_stop'] = (df['cleaned_action'] == 5)
    df['bout_id'] = (df['is_stop'] != df['is_stop'].shift()).cumsum()
    
    bout_stats = df.groupby(['bout_id', 'is_stop']).size().reset_index(name='length')
    
    approach_bouts = bout_stats[bout_stats['is_stop'] == False]['length'].tolist()
    stop_bouts = bout_stats[bout_stats['is_stop'] == True]['length'].tolist()
    
    num_trials = len(approach_bouts)
    avg_approach_steps = np.mean(approach_bouts) if num_trials > 0 else 0
    avg_stops_per_bout = np.mean(stop_bouts) if len(stop_bouts) > 0 else 0
    
    print(f"\n==================================================")
    print(f" EVALUATION COMPLETE: {os.path.basename(session_dir)}")
    print(f"==================================================")
    print(f" Total Approach Sessions (Trials): {num_trials}")
    print(f" Average Duration (Steps):         {avg_approach_steps:.1f}")
    print(f" Average Intentional Stops / Bout: {avg_stops_per_bout:.1f}")
    print(f"==================================================\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_navigation_trials.py <session_dir_or_name>")
        return
        
    session_arg = sys.argv[1]
    
    if os.path.exists(session_arg):
        session_dir = session_arg
    else:
        session_dir = os.path.join(DATA_ROOT, session_arg)
        
    if not os.path.exists(session_dir):
        print(f"ERROR: Session directory not found: {session_dir}")
        return
        
    evaluate_session(session_dir)

if __name__ == '__main__':
    main()
