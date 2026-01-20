#!/usr/bin/env python3
"""
System health check - Verify all modules load correctly
"""

import sys
import importlib

def test_imports():
    """Test that all modules import correctly"""
    modules = [
        'indicators',
        'conviction_monitor',
        'strategy_engine',
        'backtester',
        'walkforward',
        'live_ws',
        'db',
        'alerts',
        'download_history'
    ]
    
    failed = []
    
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"✓ {mod}")
        except Exception as e:
            print(f"✗ {mod}: {e}")
            failed.append(mod)
    
    return len(failed) == 0


def test_config():
    """Test that config.yaml loads correctly"""
    try:
        import yaml
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
        
        # Check required keys
        required = ['indicators', 'signal_scoring', 'conviction', 'walkforward', 'risk']
        for key in required:
            if key not in config:
                print(f"✗ config.yaml missing: {key}")
                return False
        
        print("✓ config.yaml loads correctly")
        return True
    except Exception as e:
        print(f"✗ config.yaml error: {e}")
        return False


def test_dirs():
    """Test that directories exist"""
    from pathlib import Path
    
    dirs = ['data', 'logs', 'models']
    for d in dirs:
        p = Path(d)
        if p.exists():
            print(f"✓ {d}/ exists")
        else:
            print(f"✗ {d}/ missing")
            return False
    
    return True


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║ V4 PRO MAX ULTRA - SYSTEM HEALTH CHECK                         ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    results = []
    
    print("1. Testing imports...")
    results.append(test_imports())
    
    print("\n2. Testing config...")
    results.append(test_config())
    
    print("\n3. Testing directories...")
    results.append(test_dirs())
    
    print("\n" + "="*70)
    
    if all(results):
        print("✓ ALL CHECKS PASSED - System is ready!")
        print("\nNext steps:")
        print("  1. python main.py download           # Download data")
        print("  2. python main.py backtest --symbol PAXGUSDT --tf 15m")
        print("  3. python main.py backtest --symbol PAXGUSDT --tf 15m --walkforward")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Fix errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
