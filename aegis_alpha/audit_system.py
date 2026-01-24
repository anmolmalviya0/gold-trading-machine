import requests
import sys
import time
from datetime import datetime

# CONFIG
BASE_URL = "http://localhost:8000"
ENDPOINTS = {
    "STATUS": "/api/status",
    "NEWS": "/api/news",
    "BTC_PREDICT": "/api/predict/BTC",
    "ETH_PREDICT": "/api/predict/ETH",
    "LOGS": "/api/logs?lines=5"
}

print(f"🕵️ STARTING FORENSIC AUDIT PROTOCOL")
print(f"📅 Date: {datetime.now()}")
print(f"🎯 Target: {BASE_URL}")
print("-" * 50)

failures = 0

for name, endpoint in ENDPOINTS.items():
    url = f"{BASE_URL}{endpoint}"
    print(f"Testing {name}...", end=" ")
    start_time = time.time()
    try:
        res = requests.get(url, timeout=2)
        latency = (time.time() - start_time) * 1000
        
        if res.status_code == 200:
            data = res.json()
            # Deep Validation
            if name == "NEWS":
                if len(data.get('headlines', [])) > 0:
                     print(f"✅ PASS ({latency:.1f}ms) | {len(data['headlines'])} headlines")
                else:
                    print(f"⚠️ FAIL (Empty News)")
                    failures += 1
            elif "PREDICT" in name:
                if 'confidence' in data:
                    print(f"✅ PASS ({latency:.1f}ms) | Conf: {data['confidence']:.1%}")
                else:
                    print(f"⚠️ FAIL (Invalid Prediction Structure)")
                    failures += 1
            elif name == "STATUS":
                if data.get('running') == True:
                    print(f"✅ PASS ({latency:.1f}ms) | PID: {data.get('pid')}")
                else:
                    print(f"⚠️ FAIL (Daemon Not Running)")
                    failures += 1
            else:
                 print(f"✅ PASS ({latency:.1f}ms)")
        else:
            print(f"❌ FAIL (Status {res.status_code})")
            failures += 1
            
    except Exception as e:
        print(f"❌ CRITICAL FAIL: {e}")
        failures += 1

print("-" * 50)
if failures == 0:
    print("🚀 AUDIT RESULT: SYSTEM HEALTHY")
    sys.exit(0)
else:
    print(f"🚨 AUDIT RESULT: {failures} FAILURES DETECTED")
    sys.exit(1)
