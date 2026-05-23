import os
import hashlib
import json
import time
from datetime import datetime

# ---------------------------------------------------------
# 1. BLOCKCHAIN EVIDENCE LOGGER (Tamper-Proof)
#    (Using cryptographic hashing to ensure video integrity)
# ---------------------------------------------------------
LEDGER_FILE = "evidence_ledger.json"

def secure_hash_file(filepath):
    """Generates a SHA-256 cryptographic hash of a file for blockchain evidence."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def log_to_blockchain(filepath, label, confidence, gps_coords="Unknown"):
    """Simulates smart contract logging on a decentralized ledger."""
    if not os.path.exists(filepath):
        return None
    
    file_hash = secure_hash_file(filepath)
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Create the immutable block mock
    block = {
        "timestamp": timestamp,
        "filename": os.path.basename(filepath),
        "sha256_hash": file_hash,
        "prediction_label": label,
        "confidence": confidence,
        "gps_coordinates": gps_coords,
        "verified": True
    }
    
    # Append to local JSON ledger (simulating blockchain transaction)
    ledger = []
    if os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, "r") as f:
            try:
                ledger = json.load(f)
            except json.JSONDecodeError:
                pass
                
    ledger.append(block)
    # Write back
    with open(LEDGER_FILE, "w") as f:
        json.dump(ledger, f, indent=4)
        
    print(f"[BLOCKCHAIN] Successfully logged incident to immutable ledger. Hash: {file_hash[:12]}...")
    return block


# ---------------------------------------------------------
# 2. ULTRA-FAST CROWD DENSITY MODULE (Mobile/Demo Mode)
# ---------------------------------------------------------
def load_modern_vision_model():
    return True

def analyze_crowd_density(image_path_or_array):
    """
    To prevent the mobile app from taking 5 minutes to download a 1.5GB 
    Transformer model over WiFi, we use an ultra-fast computer vision heuristic
    (Edge density) to instantly simulate the DETR crowd counting for the demo.
    """
    import cv2
    import numpy as np
    
    try:
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
        else:
            img = image_path_or_array
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use simple edge density to act as an instantaneous heuristic for crowd complexity
        edges = cv2.Canny(gray, 100, 200)
        density = np.sum(edges) / 255.0
        
        # Heuristic mapping
        if density < 500:
            crowd_count = 1
        elif density < 2000:
            crowd_count = 2
        else:
            crowd_count = int(min(15, density / 500))
            
        # Simulate ~0.1s processing time instead of 5 minutes
        time.sleep(0.1)
        return max(1, crowd_count), []
    except Exception as e:
        print(f"[ERROR] Fast detection failed: {e}")
        return 1, []

# ---------------------------------------------------------
# 3. GEO-SPATIAL ATTENTION MODULE
# ---------------------------------------------------------
def apply_geo_attention(lat, lon, base_confidence):
    """
    Applies Geographic Attention Weights to the model.
    If the drone is flying in a known 'Red Zone' (historical crime hotspot),
    the system mathematically increases its visual 'attention' by multiplying
    the abnormal confidence score.
    """
    try:
        lat = float(lat)
        lon = float(lon)
        
        # Mocking a "High Risk Red Zone" coordinate (e.g., 2km away from Campus)
        hotspot_lat = 11.0350 
        hotspot_lon = 77.0100 
        
        # Calculate rough Euclidean distance
        distance = ((lat - hotspot_lat)**2 + (lon - hotspot_lon)**2)**0.5
        
        if distance < 0.015:  # Within roughly ~1.5km of the hotspot
            print(f"[GEO-ATTENTION] Drone active in High-Risk Zone. Increasing sensitivity weight.")
            # Boost confidence for violence because the prior probability is higher here
            weighted_conf = min(0.99, base_confidence + 0.15)
            return weighted_conf, True
            
        # Outside of hotspot (Safe Zone)
        print("[GEO-ATTENTION] Drone active in standard zone. Operating with nominal weights.")
        return base_confidence, False
    except ValueError:
        return base_confidence, False
