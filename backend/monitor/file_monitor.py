import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)

class FileMonitorHandler(FileSystemEventHandler):
    def __init__(self):
        self.events = deque(maxlen=1000)
        self.file_entropy = {}
        
    def on_modified(self, event):
        if not event.is_directory:
            self.events.append({
                'type': 'modified',
                'path': event.src_path,
                'time': time.time(),
                'entropy': self.calculate_entropy(event.src_path)
            })
    
    def on_created(self, event):
        if not event.is_directory:
            self.events.append({
                'type': 'created',
                'path': event.src_path,
                'time': time.time(),
                'entropy': self.calculate_entropy(event.src_path)
            })
    
    def calculate_entropy(self, filepath):
        """Calculate Shannon entropy of file"""
        try:
            if os.path.exists(filepath) and os.path.getsize(filepath) < 10*1024*1024:  # <10MB
                with open(filepath, 'rb') as f:
                    data = f.read()
                    if data:
                        # Calculate entropy
                        entropy = 0
                        for x in range(256):
                            p_x = data.count(x) / len(data)
                            if p_x > 0:
                                entropy += - p_x * (p_x.bit_length() - 1)  # log2 approximation
                        return entropy / 8  # Normalize to 0-1
        except:
            pass
        return 0

class FileMonitor:
    def __init__(self, paths_to_watch=None):
        if paths_to_watch is None:
            paths_to_watch = [
                os.path.expanduser("~/Documents"),
                os.path.expanduser("~/Desktop"),
                os.path.expanduser("~/Downloads")
            ]
        
        self.paths_to_watch = paths_to_watch
        self.handler = FileMonitorHandler()
        self.observer = Observer()
        self.is_running = False
    
    def start(self):
        """Start file system monitoring"""
        for path in self.paths_to_watch:
            if os.path.exists(path):
                self.observer.schedule(self.handler, path, recursive=True)
                logger.info(f"Watching path: {path}")
        
        self.observer.start()
        self.is_running = True
    
    def stop(self):
        """Stop monitoring"""
        if self.is_running:
            self.observer.stop()
            self.observer.join()
            self.is_running = False
    
    def get_recent_events(self, seconds=10):
        """Get events from last N seconds"""
        current_time = time.time()
        recent = [e for e in self.handler.events if current_time - e['time'] <= seconds]
        return recent
    
    def get_file_write_rate(self):
        """Get file write rate per second"""
        recent = self.get_recent_events(1)
        return len(recent)
    
    def check_ransomware_patterns(self):
        """Check for ransomware-like patterns"""
        recent = self.get_recent_events(5)
        
        if len(recent) < 10:
            return False
        
        # Check for rapid file modifications
        file_paths = [e['path'] for e in recent]
        unique_paths = len(set(file_paths))
        
        # If many unique files modified quickly, could be ransomware
        if unique_paths > 20:
            return True
        
        # Check for high entropy (encryption)
        avg_entropy = sum(e['entropy'] for e in recent) / len(recent)
        if avg_entropy > 0.7:
            return True
        
        return False