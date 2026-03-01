import psutil
import time
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class CPUMonitor:
    def __init__(self):
        self.history = []
        self.cores = psutil.cpu_count(logical=True)
    
    def get_metrics(self) -> Dict:
        """Get CPU metrics"""
        # Get per-core percentages
        per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Get total CPU usage
        total_usage = psutil.cpu_percent(interval=0.1)
        
        # Get CPU frequency
        freq = psutil.cpu_freq()
        
        # Detect CPU spikes
        current_total = total_usage
        spike_detected = False
        
        if self.history:
            avg_usage = np.mean(self.history[-10:]) if len(self.history) >= 10 else np.mean(self.history)
            if current_total > avg_usage + 30:  # 30% spike
                spike_detected = True
        
        self.history.append(current_total)
        if len(self.history) > 100:
            self.history.pop(0)
        
        return {
            'total_usage': total_usage,
            'per_core': per_core,
            'frequency': freq.current if freq else 0,
            'cores': self.cores,
            'spike_detected': spike_detected,
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0,0,0)
        }
    
    def get_core_affinity(self) -> List[int]:
        """Get recommended core affinity for AMD"""
        # For AMD Ryzen, recommend using physical cores first
        physical_cores = psutil.cpu_count(logical=False)
        return list(range(physical_cores))
    
    def detect_burst(self) -> bool:
        """Detect CPU burst activity"""
        if len(self.history) < 5:
            return False
        
        recent = self.history[-5:]
        if max(recent) - min(recent) > 40:  # 40% variation
            return True
        return False