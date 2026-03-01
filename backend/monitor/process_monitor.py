import psutil
import os
import hashlib
import numpy as np
from typing import List, Dict
import time
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class ProcessMonitor:
    def __init__(self):
        self.history = defaultdict(list)  # Store historical process data
        self.file_write_counts = defaultdict(int)
        self.last_check = time.time()
    
    def get_processes(self) -> List[Dict]:
        """Get all running processes with metrics"""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 
                                        'num_threads']):  # Removed 'connections' temporarily
            try:
                pinfo = proc.info
                
                # Get file write rate (simplified for now)
                file_writes = 0  # Placeholder
                
                # Calculate entropy for process memory (simplified)
                entropy = 0.1 + (hash(proc.name()) % 100) / 1000  # Simple hash-based entropy
                
                # Get CPU delta
                cpu_delta = self._get_cpu_delta(proc.pid, pinfo['cpu_percent'] or 0)
                
                # Get number of connections (simplified)
                try:
                    connections = len(proc.connections(kind='inet'))
                except:
                    connections = 0
                
                # Check for file type changes
                file_changes = 0  # Placeholder
                
                process_data = {
                    'pid': proc.pid,
                    'name': pinfo['name'] or 'unknown',
                    'cpu': pinfo['cpu_percent'] or 0,
                    'cpu_delta': cpu_delta,
                    'memory': pinfo['memory_percent'] or 0,
                    'memory_percent': pinfo['memory_percent'] or 0,
                    'num_threads': pinfo['num_threads'] or 1,
                    'connections': connections,
                    'file_writes': file_writes,
                    'entropy': entropy,
                    'file_type_changes': file_changes,
                    'create_time': proc.create_time(),
                    'status': proc.status()
                }
                
                processes.append(process_data)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        return processes
    
    def get_process_by_pid(self, pid: int) -> Dict:
        """Get specific process details"""
        try:
            proc = psutil.Process(pid)
            return {
                'pid': pid,
                'name': proc.name(),
                'cpu': proc.cpu_percent(interval=0.1),
                'memory': proc.memory_percent(),
                'num_threads': proc.num_threads(),
                'connections': len(proc.connections(kind='inet')),
                'create_time': proc.create_time(),
                'exe': proc.exe(),
                'cmdline': proc.cmdline(),
                'status': proc.status()
            }
        except:
            return None
    
    def _get_file_write_rate(self, proc) -> float:
        """Estimate file write rate for process"""
        try:
            # This is a simplified version - in reality, you'd use system hooks
            io_counters = proc.io_counters()
            if io_counters:
                write_bytes = io_counters.write_bytes
                current_time = time.time()
                
                # Store in history
                key = f"{proc.pid}_write"
                if key in self.history:
                    last_bytes, last_time = self.history[key]
                    rate = (write_bytes - last_bytes) / (current_time - last_time) / 1024  # KB/s
                    self.history[key] = (write_bytes, current_time)
                    return min(rate, 1000)  # Cap at 1000 KB/s
                else:
                    self.history[key] = (write_bytes, current_time)
                    return 0
        except:
            pass
        return np.random.randint(0, 100)  # Fallback for demo
    
    def _calculate_entropy(self, proc) -> float:
        """Calculate entropy for process (simplified)"""
        # In reality, you'd sample process memory
        # For demo, generate based on process name and activity
        try:
            # Higher entropy for suspicious process names
            suspicious_names = ['powershell', 'cmd', 'wscript', 'mshta', 'rundll32']
            name = proc.name().lower()
            
            if any(s in name for s in suspicious_names):
                base_entropy = 0.6 + np.random.random() * 0.3
            else:
                base_entropy = 0.1 + np.random.random() * 0.3
            
            # Adjust based on CPU usage
            cpu_usage = proc.cpu_percent() or 0
            if cpu_usage > 50:
                base_entropy += 0.2
            
            return min(base_entropy, 1.0)
        except:
            return np.random.random() * 0.5
    
    def _get_cpu_delta(self, pid: int, current_cpu: float) -> float:
        """Get change in CPU usage"""
        key = f"{pid}_cpu"
        if key in self.history:
            last_cpu = self.history[key]
            delta = current_cpu - last_cpu
        else:
            delta = 0
        
        self.history[key] = current_cpu
        return delta
    
    def _get_file_changes(self, proc) -> int:
        """Get number of file type changes (simplified)"""
        # In reality, monitor file extensions being written
        # For demo, return random value based on process
        try:
            if 'ransom' in proc.name().lower():
                return np.random.randint(50, 200)
            elif 'powershell' in proc.name().lower():
                return np.random.randint(10, 50)
            else:
                return np.random.randint(0, 10)
        except:
            return 0
    
    def _get_status(self, proc) -> str:
        """Get process status"""
        try:
            return proc.status()
        except:
            return 'unknown'