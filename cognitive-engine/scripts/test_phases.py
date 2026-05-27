import pandas as pd
import glob

def get_phases(df):
    is_zero = (df['pwm_left'] == 0) & (df['pwm_right'] == 0)
    
    phases = []
    
    # 0 = Trial, 1 = Randomizer
    current_moving_phase = 0 
    
    # Let's find blocks of zeros
    block_start = 0
    curr_z = is_zero.iloc[0]
    
    for i in range(len(df)):
        z = is_zero.iloc[i]
        if z != curr_z:
            length = i - block_start
            if curr_z and length >= 9:
                # It was a DWELL
                phases.extend(['Dwell'] * length)
                # Next moving block will switch type
                current_moving_phase = 1 - current_moving_phase
            else:
                # It was moving, or a short stop
                phase_name = 'Trial' if current_moving_phase == 0 else 'Randomizer'
                phases.extend([phase_name] * length)
            
            curr_z = z
            block_start = i
            
    # last block
    length = len(df) - block_start
    if curr_z and length >= 9:
        phases.extend(['Dwell'] * length)
    else:
        phase_name = 'Trial' if current_moving_phase == 0 else 'Randomizer'
        phases.extend([phase_name] * length)
        
    return phases

for d in ['data/markov_2026-05-01_13-37-31', 'data/markov_2026-05-01_13-18-11', 'data/markov_2026-05-01_15-12-21']:
    df = pd.read_csv(d + '/episode_data.csv')
    phases = get_phases(df)
    df['phase'] = phases
    
    print(d)
    # Check alignment with active_controller
    print(df.groupby(['active_controller', 'phase']).size())
    print()
