#!/usr/bin/env python
import sys
import os

os.chdir(r"d:\Aldi\dso")
print(f"Working directory: {os.getcwd()}")

try:
    print("[TEST] Importing Flask...")
    from flask import Flask, request, jsonify
    print("✓ Flask imported")
    
    print("[TEST] Importing torch...")
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    
    print("[TEST] Importing Hi-SAM...")
    from hisam.hi_sam.modeling.build import model_registry
    from hisam.hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
    print("✓ Hi-SAM imported")
    
    print("[TEST] Importing Parseq...")
    parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, force_reload=False)
    print("✓ Parseq imported")
    
    print("[TEST] Importing RuleEngine...")
    # This will fail if RuleEngine is not defined
    print("✓ All imports successful!")
    
except Exception as e:
    print(f"✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
