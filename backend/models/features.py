"""
Feature Extraction Module for Ransomware Detection
Extracts behavioral features from system activity
"""

import numpy as np
import psutil
import os
import hashlib
import time
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import logging
import math
from scipy import stats
import json

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extract features from system activity for ML model
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        
        # History buffers for temporal features
        self.cpu_history = defaultdict(lambda: deque(maxlen=window_size))
        self.file_write_history = defaultdict(lambda: deque(maxlen=window_size))
        self.entropy_history = defaultdict(lambda: deque(maxlen=window_size))
        self.io_history = defaultdict(lambda: deque(maxlen=window_size))
        
        # Feature cache
        self.feature_cache = {}
        self.last_extraction = {}
        
        # Feature names (must match model input)
        self.feature_names = [
            'cpu_percent',
            'cpu_delta',
            'cpu_std',
            'cpu_burst_detected',
            'memory_percent',
            'memory_delta',
            'num_threads',
            'thread_delta',
            'file_writes_rate',
            'file_writes_delta',
            'file_writes_acceleration',
            'entropy',
            'entropy_delta',
            'entropy_std',
            'io_read_rate',
            'io_write_rate',
            'io_total_rate',
            'connections',
            'connection_delta',
            'file_type_changes',
            'file_type_diversity',
            'process_age',
            'parent_child_ratio',
            'sudden_termination_risk',
            'network_activity_score',
            'registry_changes',  # Windows only
            'dll_injections',     # Windows only
            'code_injection_score'
        ]
        
        # Feature thresholds (for normalization)
        self.feature_ranges = {
            'cpu_percent': (0, 100),
            'cpu_delta': (-50, 50),
            'cpu_std': (0, 30),
            'cpu_burst_detected': (0, 1),
            'memory_percent': (0, 100),
            'memory_delta': (-20, 20),
            'num_threads': (1, 100),
            'thread_delta': (-10, 10),
            'file_writes_rate': (0, 1000),
            'file_writes_delta': (-500, 500),
            'file_writes_acceleration': (-200, 200),
            'entropy': (0, 1),
            'entropy_delta': (-0.5, 0.5),
            'entropy_std': (0, 0.3),
            'io_read_rate': (0, 1000),
            'io_write_rate': (0, 1000),
            'io_total_rate': (0, 2000),
            'connections': (0, 50),
            'connection_delta': (-20, 20),
            'file_type_changes': (0, 100),
            'file_type_diversity': (0, 10),
            'process_age': (0, 3600),
            'parent_child_ratio': (0, 10),
            'sudden_termination_risk': (0, 1),
            'network_activity_score': (0, 1),
            'registry_changes': (0, 100),
            'dll_injections': (0, 10),
            'code_injection_score': (0, 1)
        }
    
    def extract_features(self, process: Dict, system_state: Dict) -> np.ndarray:
        """
        Extract all features for a process
        
        Args:
            process: Process data dictionary
            system_state: Overall system state
            
        Returns:
            Feature vector as numpy array
        """
        pid = process['pid']
        current_time = time.time()
        
        # Update history buffers
        self._update_histories(pid, process, current_time)
        
        # Extract all features
        features = []
        
        # CPU-based features
        features.extend(self._extract_cpu_features(pid, process))
        
        # Memory features
        features.extend(self._extract_memory_features(pid, process))
        
        # Thread features
        features.extend(self._extract_thread_features(pid, process))
        
        # File I/O features
        features.extend(self._extract_file_features(pid, process))
        
        # Entropy features
        features.extend(self._extract_entropy_features(pid, process))
        
        # Network features
        features.extend(self._extract_network_features(pid, process))
        
        # Behavioral features
        features.extend(self._extract_behavioral_features(pid, process, system_state))
        
        # Security features
        features.extend(self._extract_security_features(pid, process))
        
        # Cache features
        self.feature_cache[pid] = features
        self.last_extraction[pid] = current_time
        
        return np.array(features, dtype=np.float32)
    
    def _update_histories(self, pid: int, process: Dict, current_time: float):
        """Update feature history buffers"""
        
        # CPU history
        self.cpu_history[pid].append({
            'time': current_time,
            'value': process.get('cpu', 0)
        })
        
        # File write history
        self.file_write_history[pid].append({
            'time': current_time,
            'value': process.get('file_writes', 0)
        })
        
        # Entropy history
        self.entropy_history[pid].append({
            'time': current_time,
            'value': process.get('entropy', 0)
        })
    
    def _extract_cpu_features(self, pid: int, process: Dict) -> List[float]:
        """Extract CPU-related features"""
        cpu_current = process.get('cpu', 0)
        cpu_history = list(self.cpu_history[pid])
        
        # Current CPU
        cpu_percent = min(100, max(0, cpu_current))
        
        # CPU delta (change from last measurement)
        if len(cpu_history) >= 2:
            cpu_delta = cpu_current - cpu_history[-2]['value']
        else:
            cpu_delta = 0
        
        # CPU standard deviation (variability)
        if len(cpu_history) >= 5:
            cpu_values = [h['value'] for h in cpu_history[-5:]]
            cpu_std = np.std(cpu_values)
        else:
            cpu_std = 0
        
        # CPU burst detection
        cpu_burst = 1 if (cpu_current > 70 and cpu_delta > 30) else 0
        
        return [cpu_percent, cpu_delta, cpu_std, cpu_burst]
    
    def _extract_memory_features(self, pid: int, process: Dict) -> List[float]:
        """Extract memory-related features"""
        memory_current = process.get('memory', 0)
        
        # Memory percent
        memory_percent = min(100, max(0, memory_current))
        
        # Memory delta (simplified)
        memory_delta = process.get('memory_delta', 0)
        
        return [memory_percent, memory_delta]
    
    def _extract_thread_features(self, pid: int, process: Dict) -> List[float]:
        """Extract thread-related features"""
        threads_current = process.get('num_threads', 1)
        
        # Number of threads
        num_threads = min(100, max(1, threads_current))
        
        # Thread delta (simplified)
        thread_delta = 0
        
        return [num_threads, thread_delta]
    
    def _extract_file_features(self, pid: int, process: Dict) -> List[float]:
        """Extract file I/O features"""
        file_writes = process.get('file_writes', 0)
        file_history = list(self.file_write_history[pid])
        
        # Current write rate
        file_writes_rate = min(1000, max(0, file_writes))
        
        # Write rate delta
        if len(file_history) >= 2:
            file_delta = file_writes - file_history[-2]['value']
        else:
            file_delta = 0
        
        # Write acceleration (rate of change of rate)
        if len(file_history) >= 3:
            recent = [h['value'] for h in file_history[-3:]]
            deltas = [recent[i] - recent[i-1] for i in range(1, 3)]
            acceleration = deltas[1] - deltas[0] if len(deltas) == 2 else 0
        else:
            acceleration = 0
        
        # I/O rates (simplified)
        io_read_rate = process.get('io_read', 0)
        io_write_rate = process.get('io_write', 0)
        io_total_rate = io_read_rate + io_write_rate
        
        return [
            file_writes_rate,
            file_delta,
            acceleration,
            io_read_rate,
            io_write_rate,
            io_total_rate
        ]
    
    def _extract_entropy_features(self, pid: int, process: Dict) -> List[float]:
        """Extract entropy-based features"""
        entropy_current = process.get('entropy', 0)
        entropy_history = list(self.entropy_history[pid])
        
        # Current entropy
        entropy = min(1, max(0, entropy_current))
        
        # Entropy delta
        if len(entropy_history) >= 2:
            entropy_delta = entropy_current - entropy_history[-2]['value']
        else:
            entropy_delta = 0
        
        # Entropy standard deviation
        if len(entropy_history) >= 5:
            entropy_values = [h['value'] for h in entropy_history[-5:]]
            entropy_std = np.std(entropy_values)
        else:
            entropy_std = 0
        
        return [entropy, entropy_delta, entropy_std]
    
    def _extract_network_features(self, pid: int, process: Dict) -> List[float]:
        """Extract network-related features"""
        connections = process.get('connections', 0)
        
        # Current connections
        conn_count = min(50, max(0, connections))
        
        # Connection delta
        conn_delta = 0  # Would need history
        
        # Network activity score (0-1)
        network_score = min(1, connections / 20) if connections > 0 else 0
        
        return [conn_count, conn_delta, network_score]
    
    def _extract_behavioral_features(self, pid: int, process: Dict, system_state: Dict) -> List[float]:
        """Extract behavioral patterns"""
        
        # File type changes
        file_changes = process.get('file_type_changes', 0)
        file_type_changes = min(100, max(0, file_changes))
        
        # File type diversity (number of different file types accessed)
        file_diversity = process.get('file_diversity', 1)
        
        # Process age (seconds since creation)
        create_time = process.get('create_time', time.time())
        process_age = min(3600, max(0, time.time() - create_time))
        
        # Parent-child ratio (simplified)
        parent_child_ratio = process.get('child_count', 0) / max(1, process.get('parent_count', 1))
        
        # Sudden termination risk (simplified)
        term_risk = 1 if process.get('status') == 'zombie' else 0
        
        return [
            file_type_changes,
            file_diversity,
            process_age,
            parent_child_ratio,
            term_risk
        ]
    
    def _extract_security_features(self, pid: int, process: Dict) -> List[float]:
        """Extract security-related features"""
        
        # Registry changes (Windows only - placeholder)
        registry_changes = process.get('registry_changes', 0)
        
        # DLL injections (simplified)
        dll_injections = 1 if 'injection' in str(process.get('name', '')).lower() else 0
        
        # Code injection score
        injection_score = 0
        suspicious_indicators = [
            'alloc' in str(process.get('cmdline', '')).lower(),
            'write' in str(process.get('cmdline', '')).lower(),
            'execute' in str(process.get('cmdline', '')).lower()
        ]
        if any(suspicious_indicators):
            injection_score = 0.5
        if all(suspicious_indicators):
            injection_score = 1.0
        
        return [registry_changes, dll_injections, injection_score]
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range"""
        normalized = []
        
        for i, value in enumerate(features):
            if i < len(self.feature_names):
                min_val, max_val = self.feature_ranges[self.feature_names[i]]
                
                # Clip to range
                clipped = max(min_val, min(max_val, value))
                
                # Normalize
                if max_val > min_val:
                    normalized_val = (clipped - min_val) / (max_val - min_val)
                else:
                    normalized_val = 0.5
                
                normalized.append(normalized_val)
            else:
                normalized.append(value)
        
        return np.array(normalized, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()
    
    def get_feature_importance_weights(self) -> np.ndarray:
        """Get default feature importance weights"""
        # Higher weight for features more indicative of ransomware
        weights = {
            'cpu_percent': 0.5,
            'cpu_delta': 0.7,
            'cpu_std': 0.3,
            'cpu_burst_detected': 1.0,
            'memory_percent': 0.4,
            'memory_delta': 0.5,
            'num_threads': 0.3,
            'thread_delta': 0.4,
            'file_writes_rate': 1.0,
            'file_writes_delta': 0.9,
            'file_writes_acceleration': 0.8,
            'entropy': 1.0,
            'entropy_delta': 0.9,
            'entropy_std': 0.6,
            'io_read_rate': 0.3,
            'io_write_rate': 0.8,
            'io_total_rate': 0.6,
            'connections': 0.5,
            'connection_delta': 0.6,
            'file_type_changes': 0.9,
            'file_type_diversity': 0.7,
            'process_age': 0.3,
            'parent_child_ratio': 0.5,
            'sudden_termination_risk': 0.7,
            'network_activity_score': 0.6,
            'registry_changes': 0.8,
            'dll_injections': 1.0,
            'code_injection_score': 1.0
        }
        
        return np.array([weights.get(name, 0.5) for name in self.feature_names])

# Advanced feature extractor with more sophisticated algorithms
class AdvancedFeatureExtractor(FeatureExtractor):
    """Enhanced feature extractor with advanced statistical features"""
    
    def __init__(self, window_size: int = 20):
        super().__init__(window_size)
        
        # Additional advanced features
        self.advanced_feature_names = [
            'cpu_fft_peak',  # Frequency domain features
            'cpu_autocorr',
            'file_write_pattern_score',
            'entropy_velocity',
            'process_similarity_score',
            'anomaly_temporal_consistency',
            'io_burst_pattern',
            'thread_spawn_rate',
            'memory_allocation_pattern',
            'encryption_pattern_score'
        ]
        
        self.feature_names.extend(self.advanced_feature_names)
    
    def extract_advanced_features(self, pid: int, process: Dict) -> List[float]:
        """Extract advanced statistical features"""
        
        # CPU frequency domain features
        cpu_fft_peak = self._compute_fft_peak(pid)
        cpu_autocorr = self._compute_autocorrelation(pid)
        
        # File write pattern analysis
        file_pattern = self._analyze_file_write_pattern(pid)
        
        # Entropy velocity (rate of change of entropy change)
        entropy_velocity = self._compute_entropy_velocity(pid)
        
        # Process similarity (compared to known ransomware patterns)
        similarity_score = self._compute_process_similarity(process)
        
        # Temporal consistency of anomalies
        temporal_consistency = self._compute_temporal_consistency(pid)
        
        # I/O burst pattern detection
        io_burst = self._detect_io_burst(pid)
        
        # Thread spawn rate
        thread_rate = self._compute_thread_rate(pid, process)
        
        # Memory allocation pattern
        memory_pattern = self._analyze_memory_pattern(process)
        
        # Encryption pattern detection
        encryption_score = self._detect_encryption_pattern(pid, process)
        
        return [
            cpu_fft_peak,
            cpu_autocorr,
            file_pattern,
            entropy_velocity,
            similarity_score,
            temporal_consistency,
            io_burst,
            thread_rate,
            memory_pattern,
            encryption_score
        ]
    
    def _compute_fft_peak(self, pid: int) -> float:
        """Compute peak frequency from CPU usage FFT"""
        history = list(self.cpu_history[pid])
        if len(history) < 10:
            return 0
        
        values = [h['value'] for h in history]
        
        # Compute FFT
        fft = np.fft.fft(values)
        magnitude = np.abs(fft)
        
        # Find peak frequency (excluding DC)
        if len(magnitude) > 1:
            peak = np.max(magnitude[1:len(magnitude)//2])
            return float(peak / np.sum(magnitude))
        return 0
    
    def _compute_autocorrelation(self, pid: int) -> float:
        """Compute autocorrelation of CPU usage"""
        history = list(self.cpu_history[pid])
        if len(history) < 5:
            return 0
        
        values = np.array([h['value'] for h in history])
        
        # Compute autocorrelation at lag 1
        if len(values) > 1:
            autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
            return float(autocorr) if not np.isnan(autocorr) else 0
        return 0
    
    def _analyze_file_write_pattern(self, pid: int) -> float:
        """Analyze pattern of file writes"""
        history = list(self.file_write_history[pid])
        if len(history) < 5:
            return 0
        
        values = [h['value'] for h in history]
        
        # Check for burst pattern (ransomware often writes in bursts)
        diffs = np.diff(values)
        burst_count = np.sum(diffs > np.mean(diffs) * 2)
        
        return min(1, burst_count / len(diffs))
    
    def _compute_entropy_velocity(self, pid: int) -> float:
        """Compute rate of change of entropy"""
        history = list(self.entropy_history[pid])
        if len(history) < 3:
            return 0
        
        values = [h['value'] for h in history[-3:]]
        diffs = np.diff(values)
        
        if len(diffs) > 1:
            velocity = diffs[1] - diffs[0]
            return float(np.clip(velocity, -1, 1))
        return 0
    
    def _compute_process_similarity(self, process: Dict) -> float:
        """Compute similarity to known ransomware patterns"""
        ransomware_indicators = {
            'name': ['ransom', 'crypt', 'lock', 'encrypt', 'wanna', 'bad'],
            'cmdline': ['-enc', 'decode', 'hidden', 'silent'],
            'path': ['temp', 'appdata', 'programdata']
        }
        
        score = 0
        name = process.get('name', '').lower()
        cmdline = ' '.join(process.get('cmdline', [])).lower()
        path = process.get('exe', '').lower()
        
        # Check indicators
        for indicator in ransomware_indicators['name']:
            if indicator in name:
                score += 0.3
        
        for indicator in ransomware_indicators['cmdline']:
            if indicator in cmdline:
                score += 0.2
        
        for indicator in ransomware_indicators['path']:
            if indicator in path:
                score += 0.1
        
        return min(1, score)
    
    def _compute_temporal_consistency(self, pid: int) -> float:
        """Compute how consistently anomalous the process is"""
        if len(self.cpu_history[pid]) < 5:
            return 0
        
        # Get anomaly history (simplified - using CPU as proxy)
        cpu_values = [h['value'] for h in self.cpu_history[pid]]
        high_cpu = [1 if v > 50 else 0 for v in cpu_values]
        
        # Check for consistent high CPU
        if len(high_cpu) > 0:
            consistency = sum(high_cpu) / len(high_cpu)
            return float(consistency)
        return 0
    
    def _detect_io_burst(self, pid: int) -> float:
        """Detect I/O burst patterns"""
        if len(self.io_history[pid]) < 5:
            return 0
        
        io_values = [h['value'] for h in self.io_history[pid]]
        
        # Check for burst (sudden spike)
        mean_io = np.mean(io_values)
        if mean_io > 0:
            max_io = np.max(io_values)
            burst_ratio = max_io / mean_io
            return float(min(1, burst_ratio / 10))
        return 0
    
    def _compute_thread_rate(self, pid: int, process: Dict) -> float:
        """Compute thread spawn rate"""
        # Simplified - would need thread creation history
        return min(1, process.get('num_threads', 0) / 50)
    
    def _analyze_memory_pattern(self, process: Dict) -> float:
        """Analyze memory allocation pattern"""
        # Simplified - would need memory allocation tracking
        return 0.0
    
    def _detect_encryption_pattern(self, pid: int, process: Dict) -> float:
        """Detect patterns consistent with encryption"""
        score = 0
        
        # High entropy + high file writes + high CPU is suspicious
        entropy = process.get('entropy', 0)
        file_writes = process.get('file_writes', 0)
        cpu = process.get('cpu', 0)
        
        if entropy > 0.7:
            score += 0.4
        if file_writes > 200:
            score += 0.3
        if cpu > 70:
            score += 0.3
        
        return min(1, score)

# Factory function to create feature extractor
def create_feature_extractor(advanced: bool = False) -> FeatureExtractor:
    """
    Create feature extractor
    
    Args:
        advanced: Use advanced feature extraction
        
    Returns:
        Configured feature extractor
    """
    if advanced:
        return AdvancedFeatureExtractor()
    return FeatureExtractor()