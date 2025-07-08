#!/usr/bin/env python3
"""
Debug Script to inspect pickle file contents
"""

import pickle
import jax.numpy as jnp
import numpy as np

def debug_pickle_file(filename):
    """Inspects the contents of a pickle file"""
    print(f"🔍 Inspecting pickle file: {filename}")
    print("="*60)
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        print(f"📦 Root type: {type(data)}")
        print(f"📦 Root content: {data if not hasattr(data, 'keys') else 'Dict with keys'}")
        
        if isinstance(data, dict):
            print(f"\n📋 Dictionary with {len(data)} keys:")
            for key, value in data.items():
                print(f"\n  🔑 Key: '{key}'")
                print(f"     Type: {type(value)}")
                
                if hasattr(value, 'shape'):
                    print(f"     Shape: {value.shape}")
                    print(f"     Dtype: {value.dtype}")
                    print(f"     Min/Max: {np.min(value):.4f} / {np.max(value):.4f}")
                elif isinstance(value, (str, int, float, bool)):
                    print(f"     Value: {value}")
                elif isinstance(value, (list, tuple)):
                    print(f"     Length: {len(value)}")
                    if len(value) > 0:
                        print(f"     First element type: {type(value[0])}")
                elif isinstance(value, dict):
                    print(f"     Nested dict with keys: {list(value.keys())}")
                else:
                    print(f"     Content: {str(value)[:100]}...")
        
        elif hasattr(data, '__dict__'):
            print(f"\n📋 Object attributes:")
            for attr in dir(data):
                if not attr.startswith('_'):
                    value = getattr(data, attr)
                    print(f"  🔑 {attr}: {type(value)}")
                    if hasattr(value, 'shape'):
                        print(f"     Shape: {value.shape}")
        
        else:
            print(f"\n📋 Direct content: {str(data)[:200]}...")
            
    except Exception as e:
        print(f"❌ Error loading pickle file: {e}")

if __name__ == "__main__":
    debug_pickle_file('roarm_working_params.pkl')