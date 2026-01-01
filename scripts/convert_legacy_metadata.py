
import os
import csv
import re

LEGACY_DIR = "legacydata"

def convert_folder(folder_path):
    info_file = os.path.join(folder_path, "_InformationData.txt")
    if not os.path.exists(info_file):
        return

    print(f"Processing {folder_path}...")
    
    with open(info_file, 'r') as f:
        lines = f.readlines()
        
    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line: # Skip empty lines
            i += 1
            continue
            
        # Line 1: ID (e.g. 2025-06-07 05_58_39-1749333519808)
        if not re.match(r'\d{4}-\d{2}-\d{2}', line):
           i += 1
           continue
           
        id_str = line
        img_file = f"{id_str}.jpg"
        
        # Parse timestamp from ID (last part after -)
        try:
            ts_ms_str = id_str.split('-')[-1]
            ts = float(ts_ms_str) / 1000.0
        except:
            ts = 0.0
            
        # Line 2: Sensors
        sensor_line = ""
        if i + 1 < len(lines):
            sensor_line = lines[i+1].strip()
        
        # Line 3: Commands
        cmd_line = ""
        if i + 2 < len(lines):
            cmd_line = lines[i+2].strip()
        
        # Process Sensor: "0,0,12,757, ;..."
        # Split by ;
        sensor_groups = sensor_line.split(';')
        ir = 0
        batt = 0
        if sensor_groups:
             # Take first valid group?
             parts = sensor_groups[0].split(',')
             if len(parts) >= 4:
                 try:
                     ir = parts[2]
                     batt = parts[3]
                 except: pass
                 
        # Process Command: "l:0;r:200;s:0;d:0,0,0,0;..."
        l_val = '0'
        r_val = '0'
        s_val = 's:0;'
        d_val = 'd:N/A;'
        
        # Extract l, r
        m = re.search(r'l:(-?\d+);r:(-?\d+);', cmd_line)
        if m:
            motor_cmd = f"l:{m.group(1)};r:{m.group(2)};"
        else:
            motor_cmd = "l:0;r:0;"
            
        # Extract s
        m_s = re.search(r's:(\d+);', cmd_line)
        if m_s:
            sound_cmd = f"s:{m_s.group(1)};"
        else:
            sound_cmd = "s:0;"
            
        # Extract d
        m_d = re.search(r'(d:[\d,]+;)', cmd_line)
        if m_d:
            led_cmd = m_d.group(1)
        else:
            led_cmd = "d:N/A;"

        entries.append({
            'timestamp': ts,
            'img_file': img_file,
            'ir': ir,
            'battery': batt,
            'motor_cmd': motor_cmd,
            'led_cmd': led_cmd,
            'sound_cmd': sound_cmd
        })
        
        i += 3 # Move past block
        
    # Write CSV
    if entries:
        csv_path = os.path.join(folder_path, "log.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'img_file', 'ir', 'battery', 'motor_cmd', 'led_cmd', 'sound_cmd']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entries)
        print(f"Created log.csv with {len(entries)} entries in {folder_path}")

def main():
    # legacydata is at root/legacydata
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    legacy_dir = os.path.join(root_dir, LEGACY_DIR)
    
    if not os.path.exists(legacy_dir):
        print(f"Legacy directory not found: {legacy_dir}")
        return

    for d in os.listdir(legacy_dir):
        path = os.path.join(legacy_dir, d)
        if os.path.isdir(path):
            convert_folder(path)

if __name__ == "__main__":
    main()
