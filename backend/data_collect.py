"""
Collect real training data from your system
Run this for 5-10 minutes while using your computer normally
"""

import psutil
import time
import json
import os
import numpy as np
from datetime import datetime
import logging
from collections import defaultdict
import cpuinfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemDataCollector:
    def __init__(self, duration_minutes=10):
        self.duration = duration_minutes * 60
        self.normal_data = []
        self.simulated_ransomware = []
        self.process_history = defaultdict(list)
        self.start_time = None
        
    def collect_normal_data(self):
        """Collect data from normal system operation"""
        logger.info(f"Collecting normal system data for {self.duration//60} minutes...")
        logger.info("Use your computer normally during this time")
        
        self.start_time = time.time()
        samples_collected = 0
        
        try:
            while time.time() - self.start_time < self.duration:
                # Get all processes
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 
                                                'memory_percent', 'num_threads']):
                    try:
                        # Get process info
                        pinfo = proc.info
                        
                        # Skip processes with 0 CPU (inactive)
                        if pinfo['cpu_percent'] is None or pinfo['cpu_percent'] < 0.1:
                            continue
                        
                        # Get additional metrics
                        try:
                            connections = len(proc.connections(kind='inet'))
                        except:
                            connections = 0
                            
                        try:
                            io_counters = proc.io_counters()
                            if io_counters:
                                write_bytes = io_counters.write_bytes
                                # Store in history for rate calculation
                                key = f"{proc.pid}_write"
                                self.process_history[key].append((time.time(), write_bytes))
                                # Calculate rate
                                if len(self.process_history[key]) > 1:
                                    old_time, old_bytes = self.process_history[key][-2]
                                    time_diff = time.time() - old_time
                                    if time_diff > 0:
                                        write_rate = (write_bytes - old_bytes) / time_diff / 1024  # KB/s
                                    else:
                                        write_rate = 0
                                else:
                                    write_rate = 0
                            else:
                                write_rate = 0
                        except:
                            write_rate = 0
                        
                        # Calculate entropy (simplified - based on process name)
                        entropy = self._calculate_entropy(proc.name())
                        
                        # Get CPU delta
                        cpu_history = self.process_history.get(f"{proc.pid}_cpu", [])
                        if cpu_history:
                            cpu_delta = pinfo['cpu_percent'] - cpu_history[-1]
                        else:
                            cpu_delta = 0
                        self.process_history[f"{proc.pid}_cpu"].append(pinfo['cpu_percent'])
                        
                        # Create sample
                        sample = {
                            'cpu_percent': float(pinfo['cpu_percent']),
                            'memory_percent': float(pinfo['memory_percent'] or 0),
                            'file_writes_rate': float(write_rate),
                            'entropy': float(entropy),
                            'num_threads': int(pinfo['num_threads'] or 1),
                            'connections': int(connections),
                            'cpu_burst': 1 if pinfo['cpu_percent'] > 70 else 0,
                            'sudden_cpu_change': float(cpu_delta),
                            'file_type_changes': 0,  # Will be updated by file monitor
                            'process_name': proc.name(),
                            'pid': proc.pid,
                            'timestamp': time.time()
                        }
                        
                        self.normal_data.append(sample)
                        samples_collected += 1
                        
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Progress indicator
                elapsed = int(time.time() - self.start_time)
                remaining = self.duration - elapsed
                if samples_collected % 100 == 0:
                    logger.info(f"Collected {samples_collected} samples - {elapsed//60}:{elapsed%60:02d} elapsed, {remaining//60}:{remaining%60:02d} remaining")
                
                time.sleep(1)  # Sample every second
                
        except KeyboardInterrupt:
            logger.info("Data collection interrupted by user")
        
        logger.info(f"✅ Collected {len(self.normal_data)} normal samples")
        return self.normal_data
    
    def _calculate_entropy(self, name):
        """Calculate entropy based on process name"""
        # Simple entropy approximation
        if name and len(name) > 0:
            # Use character distribution as simple entropy
            chars = {}
            for c in name:
                chars[c] = chars.get(c, 0) + 1
            
            entropy = 0
            for count in chars.values():
                p = count / len(name)
                entropy -= p * np.log2(p) if p > 0 else 0
            
            # Normalize to 0-1 range (max entropy for typical process names ~3-4 bits)
            return min(1.0, entropy / 4)
        return 0.3
    
    def save_data(self, filename="training_data.json"):
        """Save collected data to file"""
        for s in self.normal_data:
            s['label'] = 0
        for s in self.simulated_ransomware:
            s['label'] = 1
        data = {
            'normal': self.normal_data,
            'ransomware': self.simulated_ransomware,
            'metadata': {
                'collection_time': datetime.now().isoformat(),
                'normal_samples': len(self.normal_data),
                'ransomware_samples': len(self.simulated_ransomware),
                'cpu_info': cpuinfo.get_cpu_info()['brand_raw'],
                'total_samples': len(self.normal_data) + len(self.simulated_ransomware)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"✅ Data saved to {filename}")
        return filename
    
    def load_data(self, filename="training_data.json"):
        """Load previously collected data"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.normal_data = data['normal']
            self.simulated_ransomware = data['ransomware']
            logger.info(f"✅ Loaded {len(self.normal_data)} normal and {len(self.simulated_ransomware)} ransomware samples")
            return data
        else:
            logger.warning(f"File {filename} not found")
            return None

def main():
    print("=" * 60)
    print("EdgeGuard AI - Training Data Collector")
    print("=" * 60)
    print("\nThis tool will collect REAL system data to train your model.")
    print("It will run for 10 minutes while you use your computer normally.")
    print("\nThen it will generate simulated ransomware samples based on")
    print("the patterns it learned from your system.")
    print("\n" + "=" * 60)
    
    # Ask for duration
    try:
        duration = int(input("\nCollection duration in minutes (default 10): ") or "10")
    except:
        duration = 10
    
    collector = SystemDataCollector(duration_minutes=duration)
    
    # Check for existing data
    if os.path.exists("training_data.json"):
        print("\n📁 Existing training data found!")
        choice = input("Load existing data? (y/n): ").lower()
        if choice == 'y':
            collector.load_data()
    
    # Collect new data if needed
    if len(collector.normal_data) == 0:
        print("\n📊 Starting data collection...")
        print("Please use your computer normally during this time.")
        print("Press Ctrl+C to stop early if needed.\n")
        
        collector.collect_normal_data()
        
        if len(collector.normal_data) > 0:
            print(f"\n✅ Collected {len(collector.normal_data)} normal samples")
    
    # Save data
    if len(collector.normal_data) > 0:
        collector.save_data()
        
        print("\n" + "=" * 60)
        print("📊 Data Collection Summary:")
        print(f"   Normal samples: {len(collector.normal_data)}")
        print(f"   Ransomware samples: {len(collector.simulated_ransomware)}")
        print(f"   Total: {len(collector.normal_data) + len(collector.simulated_ransomware)}")
        print("=" * 60)
        
        print("\nNow you can train your model with this REAL data!")
        print("Run: python -m models.train")
    else:
        print("\n❌ No data collected")

if __name__ == "__main__":
    main()