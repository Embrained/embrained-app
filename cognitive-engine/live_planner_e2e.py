import os
import sys
import time
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

# Setup paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import ACTION_PWM_MAP
from modules.comms import NervousSystem

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing End-To-End CNN on {device}...")
    
    # Load ResNet18
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "e2e_tv_model.pth"))
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        print(f"Loaded trained End-To-End model: {model_path}")
    else:
        print(f"Model not found at {model_path}!")
        print("Please run: python backend/training/train_end_to_end_bc.py")
        return

    # ImageNet preprocessing matching the training setup
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Connecting to robot nervous system...")
    ns = NervousSystem()
    time.sleep(2.0) # Wait for connection and buffer to fill
    
    print("Beginning End-To-End Navigation Loop. Press Ctrl+C to stop.")

    try:
        while True:
            frame = ns.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Convert OpenCV BGR array to PIL RGB Image
            img = Image.fromarray(frame[..., ::-1])
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs.data, 1)
                action_id = predicted.item()

            print(f"\rPredicted Action ID: {action_id}       ", end="", flush=True)
            
            # Translate action ID to motor PWM values
            if action_id in ACTION_PWM_MAP:
                left, right = ACTION_PWM_MAP[action_id]
                ns.send_pwm(left, right)
            else:
                ns.send_pwm(0, 0)
                
            time.sleep(0.1) # 10Hz control loop

    except KeyboardInterrupt:
        print("\nStopping robot...")
        ns.send_pwm(0, 0)
        ns.close()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
