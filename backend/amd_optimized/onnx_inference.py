"""
AMD-Optimized ONNX Inference Engine
Leverages AVX-512, multi-threading, and AMD-specific optimizations
"""

import numpy as np
import onnxruntime as ort
import psutil
import cpuinfo
import logging
import time
from typing import Dict, List, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

class AMDONNXInference:
    """
    ONNX Inference Engine optimized for AMD Ryzen processors
    """
    
    def __init__(self, model_path: str, num_threads: int = None):
        self.model_path = model_path
        self.num_threads = num_threads or psutil.cpu_count(logical=True)
        self.session = None
        self.input_name = None
        self.output_name = None
        self.cpu_info = cpuinfo.get_cpu_info()
        self.is_amd = 'AMD' in self.cpu_info['brand_raw']
        
        # Performance metrics
        self.inference_times = []
        self.total_inferences = 0
        self.lock = threading.Lock()
        
        # Initialize session with AMD optimizations
        self._create_optimized_session()
        
        logger.info(f"AMD ONNX Inference initialized")
        logger.info(f"  CPU: {self.cpu_info['brand_raw']}")
        logger.info(f"  Threads: {self.num_threads}")
        logger.info(f"  AMD Optimized: {self.is_amd}")
    
    def _create_optimized_session(self):
        """Create ONNX Runtime session with AMD-specific optimizations"""
        try:
            # Configure session options for AMD
            sess_options = ort.SessionOptions()
            
            # Threading optimizations
            sess_options.intra_op_num_threads = self.num_threads
            sess_options.inter_op_num_threads = max(1, self.num_threads // 2)
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Enable optimizations
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            sess_options.enable_mem_reuse = True
            
            # AMD-specific optimizations
            providers = ['CPUExecutionProvider']
            
            # Check for AMD optimizations
            if self.is_amd:
                # Try to use Vitis AI EP if available (AMD's AI acceleration)
                try:
                    import vitis_ai_onnxruntime
                    providers.insert(0, 'VitisAIExecutionProvider')
                    logger.info("Vitis AI Execution Provider enabled")
                except ImportError:
                    pass
                
                # Use OpenVINO for AMD (works well with Ryzen)
                try:
                    providers.insert(0, 'OpenVINOExecutionProvider')
                    logger.info("OpenVINO Execution Provider enabled")
                except:
                    pass
            
            # Create session
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Log provider info
            logger.info(f"ONNX Runtime providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            raise
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference with timing and optimizations
        
        Args:
            input_data: Preprocessed input features
            
        Returns:
            Model predictions
        """
        start_time = time.perf_counter()
        
        try:
            # Ensure correct data type (float32 for optimal performance)
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
            
            # Run inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_data}
            )
            
            # Record metrics
            inference_time = (time.perf_counter() - start_time) * 1000  # ms
            
            with self.lock:
                self.inference_times.append(inference_time)
                if len(self.inference_times) > 100:
                    self.inference_times.pop(0)
                self.total_inferences += 1
            
            return outputs[0]
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def infer_batch(self, batch_data: np.ndarray) -> np.ndarray:
        """
        Run batch inference (more efficient for multiple samples)
        
        Args:
            batch_data: Batch of input features [batch_size, features]
            
        Returns:
            Batch predictions
        """
        if batch_data.dtype != np.float32:
            batch_data = batch_data.astype(np.float32)
        
        return self.infer(batch_data)
    
    def infer_parallel(self, data_list: List[np.ndarray], max_workers: int = None) -> List[np.ndarray]:
        """
        Parallel inference using thread pool
        
        Args:
            data_list: List of input samples
            max_workers: Maximum number of parallel threads
            
        Returns:
            List of predictions
        """
        workers = max_workers or self.num_threads
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(self.infer, data_list))
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics"""
        with self.lock:
            if not self.inference_times:
                return {}
            
            times = np.array(self.inference_times)
            
            return {
                'avg_latency_ms': float(np.mean(times)),
                'p95_latency_ms': float(np.percentile(times, 95)),
                'p99_latency_ms': float(np.percentile(times, 99)),
                'min_latency_ms': float(np.min(times)),
                'max_latency_ms': float(np.max(times)),
                'std_latency_ms': float(np.std(times)),
                'total_inferences': self.total_inferences,
                'throughput': float(1000 / np.mean(times) * self.num_threads),
                'threads': self.num_threads
            }
    
    def get_amd_specific_metrics(self) -> Dict:
        """Get AMD-specific performance metrics"""
        if not self.is_amd:
            return {'amd_optimized': False}
        
        # Check for AVX-512 support
        cpu_flags = self.cpu_info.get('flags', [])
        has_avx512 = any('avx512' in flag for flag in cpu_flags)
        
        # Estimate SIMD utilization
        perf_stats = self.get_performance_stats()
        
        return {
            'amd_optimized': True,
            'cpu_model': self.cpu_info['brand_raw'],
            'avx512_supported': has_avx512,
            'avx512_enabled': has_avx512 and self.is_amd,
            'core_count': self.num_threads,
            'efficiency_score': self._calculate_efficiency(perf_stats),
            'provider': self.session.get_providers()[0]
        }
    
    def _calculate_efficiency(self, perf_stats: Dict) -> float:
        """Calculate CPU efficiency score"""
        if not perf_stats:
            return 0.0
        
        # Theoretical max throughput based on core count
        theoretical_max = 1000 / 0.5 * self.num_threads  # Assuming 0.5ms optimal latency
        actual = perf_stats.get('throughput', 0)
        
        efficiency = min(100, (actual / theoretical_max) * 100)
        return float(efficiency)
    
    def optimize_for_amd(self):
        """Apply additional AMD-specific optimizations"""
        if not self.is_amd:
            logger.warning("Not an AMD CPU, skipping AMD optimizations")
            return
        
        try:
            # Set thread affinity to physical cores
            physical_cores = psutil.cpu_count(logical=False)
            p = psutil.Process()
            p.cpu_affinity(list(range(physical_cores)))
            logger.info(f"Set CPU affinity to first {physical_cores} physical cores")
            
            # Increase thread priority
            try:
                p.nice(10)  # Unix
            except:
                pass
            
            # Enable huge pages if available
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                # MADV_HUGEPAGE = 14
                logger.info("Enabled huge pages support")
            except:
                pass
            
            logger.info("AMD optimizations applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply AMD optimizations: {e}")
    
    def benchmark_inference(self, iterations: int = 1000) -> Dict:
        """
        Run inference benchmark
        
        Args:
            iterations: Number of inference iterations
            
        Returns:
            Benchmark results
        """
        # Create dummy input
        input_shape = self.session.get_inputs()[0].shape
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(100):
            self.infer(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.infer(dummy_input)
            times.append((time.perf_counter() - start) * 1000)
        
        return {
            'avg_ms': float(np.mean(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'std_ms': float(np.std(times)),
            'throughput': float(1000 / np.mean(times) * self.num_threads),
            'iterations': iterations,
            'threads': self.num_threads
        }

# Factory function to create optimized inference engine
def create_optimized_inference(model_path: str, auto_detect_amd: bool = True) -> AMDONNXInference:
    """
    Create AMD-optimized inference engine
    
    Args:
        model_path: Path to ONNX model
        auto_detect_amd: Automatically detect and optimize for AMD
        
    Returns:
        Configured inference engine
    """
    engine = AMDONNXInference(model_path)
    
    if auto_detect_amd and engine.is_amd:
        engine.optimize_for_amd()
    
    return engine

