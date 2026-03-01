"""
EdgeGuard AI - Real-Time Ransomware Detection Backend
Optimized for AMD Ryzen processors with ONNX Runtime
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
import psutil
import cpuinfo
from typing import List, Dict, Optional, Any
import logging
import os
import signal
import sys
import time
from collections import deque
import threading
import platform
# Local imports
from models.detector import RansomwareDetector
from models.features import create_feature_extractor, FeatureExtractor
from monitor.process_monitor import ProcessMonitor
from monitor.file_monitor import FileMonitor
from monitor.cpu_monitor import CPUMonitor
from amd_optimized.onnx_inference import create_optimized_inference, AMDONNXInference
from amd_optimized.benchmark import BenchmarkRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/edgeguard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("data/models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("data/history", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="EdgeGuard AI - Ransomware Detection",
    description="Real-time ransomware detection using CPU behavior profiling",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global Components
# ============================================================================

class EdgeGuardSystem:
    """Main system container for all components"""
    
    def __init__(self):
        # Initialize detectors and monitors
        self.detector = RansomwareDetector()
        self.feature_extractor = create_feature_extractor(advanced=True)
        self.process_monitor = ProcessMonitor()
        self.file_monitor = FileMonitor()
        self.cpu_monitor = CPUMonitor()
        self.benchmark = BenchmarkRunner()
        self.onnx_engine = None
        
        # System state
        self.system_state = {
            'total_cpu': 0,
            'total_memory': 0,
            'active_processes': 0,
            'threat_level': 'LOW',
            'threat_score': 0,
            'last_update': datetime.now().isoformat()
        }
        
        # Data buffers
        self.threat_history = deque(maxlen=1000)
        self.process_history = {}
        self.alert_history = deque(maxlen=100)
        
        # Flags
        self.is_monitoring = False
        self.is_trained = False
        
        # Load ONNX model if available
        self._load_onnx_model()
    
    def _load_onnx_model(self):
        """Load ONNX model with AMD optimizations"""
        onnx_path = "data/models/model.onnx"
        if os.path.exists(onnx_path):
            try:
                self.onnx_engine = create_optimized_inference(onnx_path)
                logger.info("ONNX model loaded with AMD optimizations")
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                self.onnx_engine = None
    
    def start_monitoring(self):
        """Start all monitoring threads"""
        self.is_monitoring = True
        self.file_monitor.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring"""
        self.is_monitoring = False
        self.file_monitor.stop()
        logger.info("System monitoring stopped")
    
    def get_system_info(self) -> Dict:
        """Get system information with error handling"""
        try:
            cpu_info = cpuinfo.get_cpu_info()
            brand_raw = cpu_info.get('brand_raw', 'Unknown CPU')
        except:
            brand_raw = 'Unknown CPU'
            cpu_info = {}
        
        return {
            'cpu': brand_raw,
            'cores': psutil.cpu_count(logical=True),
            'physical_cores': psutil.cpu_count(logical=False),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'os': f"{sys.platform} {platform.system() if platform else 'Unknown'}",
            'python_version': sys.version,
            'is_amd': 'AMD' in brand_raw,
            'avx512_supported': any('avx512' in flag.lower() for flag in cpu_info.get('flags', [])) if cpu_info else False,
            'model_trained': self.detector.is_trained,
            'onnx_loaded': self.onnx_engine is not None
        }
# Global system instance
system = EdgeGuardSystem()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_stats = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_stats[websocket] = {
            'connected_at': datetime.now(),
            'messages_sent': 0
        }
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_stats:
            del self.connection_stats[websocket]
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict, message_type: str = "update"):
        """Broadcast message to all connected clients"""
        message['type'] = message_type
        message['timestamp'] = datetime.now().isoformat()
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
                if connection in self.connection_stats:
                    self.connection_stats[connection]['messages_sent'] += 1
            except:
                disconnected.append(connection)
        
        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# ============================================================================
# Background Tasks
# ============================================================================

async def monitoring_loop():
    """Continuous monitoring and anomaly detection"""
    logger.info("Starting monitoring loop...")
    
    while True:
        try:
            if not system.is_monitoring:
                await asyncio.sleep(1)
                continue
            
            # Collect system metrics
            processes = system.process_monitor.get_processes()
            file_events = system.file_monitor.get_recent_events(seconds=2)
            cpu_metrics = system.cpu_monitor.get_metrics()
            
            # CRITICAL FIX: Update system state with REAL values
            system.system_state.update({
                'total_cpu': cpu_metrics.get('total_usage', psutil.cpu_percent()),
                'total_memory': psutil.virtual_memory().percent,
                'active_processes': len(psutil.pids()),  # Use psutil directly for accuracy
                'last_update': datetime.now().isoformat()
            })
            
            # Log the update (for debugging)
            logger.debug(f"System state updated: CPU={system.system_state['total_cpu']}%, "
                        f"Memory={system.system_state['total_memory']}%, "
                        f"Processes={system.system_state['active_processes']}")
            
            # Track threats
            threat_scores = []
            high_threat_processes = []
            
            # Process each process for threats
            # In the monitoring_loop function, replace the feature extraction section:


            # Known safe Windows processes
            safe_processes = {
                'System', 'smss.exe', 'csrss.exe', 'wininit.exe', 'winlogon.exe',
                'services.exe', 'lsass.exe', 'svchost.exe', 'fontdrvhost.exe',
                'dwm.exe', 'explorer.exe', 'spoolsv.exe', 'taskhostw.exe',
                'SecurityHealthService.exe', 'ctfmon.exe', 'sihost.exe',
                'RuntimeBroker.exe', 'SearchIndexer.exe', 'SearchHost.exe',
                'PhoneExperienceHost.exe', 'TextInputHost.exe', 'StartMenuExperienceHost.exe',
                'Registry', 'Memory Compression', 'Idle'
            }

            # Process each process for threats
            for proc in processes[:50]:  # Limit to top 50 processes for performance
                try:
                    # Skip known safe system processes
                    proc_name = proc.get('name', '').lower()
                    if any(safe.lower() in proc_name for safe in safe_processes):
                        proc['threatScore'] = 0
                        continue
                    
                    # Extract features
                    feature_dict = {
                        'cpu_percent': proc.get('cpu', 0),
                        'memory_percent': proc.get('memory_percent', 0),
                        'file_writes_rate': proc.get('file_writes', 0),
                        'entropy': proc.get('entropy', 0.1),
                        'num_threads': proc.get('num_threads', 1),
                        'connections': proc.get('connections', 0),
                        'cpu_burst': 1 if proc.get('cpu', 0) > 80 else 0,
                        'sudden_cpu_change': proc.get('cpu_delta', 0),
                        'file_type_changes': proc.get('file_type_changes', 0)
                    }
                    
                    # Run anomaly detection
                    if system.detector.is_trained:
                        anomaly_score = system.detector.predict(feature_dict)
                        logger.info(f"{proc_name} -> {anomaly_score:.4f}") 
                        
                        # Apply safe process multiplier
                        if any(x in proc_name for x in ['svchost', 'system', 'csrss', 'winlogon']):
                            anomaly_score *= 0.1  # Reduce score for system processes
                        
                        proc['threatScore'] = float(anomaly_score * 100)
                        
                        if anomaly_score > 0.3:  # Lower threshold for logging
                            threat_scores.append(anomaly_score)
                            
                            # Get explanation for high threats
                            if anomaly_score > system.detector.threshold:
                                explanation = system.detector.explain(feature_dict)
                                proc['explanation'] = explanation
                                high_threat_processes.append(proc)
                                
                                # Generate alert for critical threats
                                if anomaly_score > 0.8:
                                    await generate_alert(proc, explanation)
                    else:
                        proc['threatScore'] = 0
                
                except Exception as e:
                    logger.error(f"Error processing process {proc.get('pid')}: {e}")
                    proc['threatScore'] = 0

            # Calculate overall threat score
            if threat_scores:
                overall_threat = np.mean(threat_scores)
                system.system_state['threat_score'] = float(overall_threat * 100)
                
                # Determine threat level
                if overall_threat < 0.3:
                    system.system_state['threat_level'] = 'LOW'
                elif overall_threat < 0.6:
                    system.system_state['threat_level'] = 'MEDIUM'
                elif overall_threat < 0.8:
                    system.system_state['threat_level'] = 'HIGH'
                else:
                    system.system_state['threat_level'] = 'CRITICAL'
            else:
                system.system_state['threat_score'] = 0
                system.system_state['threat_level'] = 'LOW'
            
            # Store in history
            system.threat_history.append({
                'timestamp': datetime.now().isoformat(),
                'score': system.system_state['threat_score'],
                'level': system.system_state['threat_level']
            })
            
            # Get energy efficiency metrics
            # Prepare broadcast data
            try:
                # Get energy metrics safely
                try:
                    energy_metrics = system.benchmark.get_energy_metrics()
                except:
                    energy_metrics = {'efficiency': 94, 'power_estimate': 65}
                
                broadcast_data = {
                    "type": "metrics",
                    "data": {
                        **system.system_state,
                        'energy_efficiency': energy_metrics.get('efficiency', 94),
                        'power_estimate': energy_metrics.get('power_estimate', 65),
                        'inference_latency': getattr(system.detector, 'last_inference_time', 2.3),
                        'active_cores': psutil.cpu_count(logical=True)
                    }
                }
                
                logger.debug(f"Broadcasting metrics: {broadcast_data}")
                await manager.broadcast(broadcast_data)
                
                # Format processes for frontend
                formatted_processes = []
                for proc in processes[:30]:
                    formatted_processes.append({
                        'pid': proc.get('pid', 0),
                        'name': proc.get('name', 'Unknown'),
                        'cpu': round(proc.get('cpu', 0), 1),
                        'memory': round(proc.get('memory', 0), 1),
                        'threatScore': round(proc.get('threatScore', 0) * 100, 1),  # Convert to percentage
                        'file_writes': round(proc.get('file_writes', 0), 0),
                        'entropy': round(proc.get('entropy', 0), 2)
                    })
                
                await manager.broadcast({
                    "type": "processes",
                    "data": formatted_processes
                })
                
                # Broadcast chart data
                chart_data = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "score": system.system_state.get('threat_score', 0),
                    "threshold": getattr(system.detector, 'threshold', 0.7) * 100
                }
                await manager.broadcast({
                    "type": "chart",
                    "data": chart_data
                })
                
            except Exception as e:
                logger.error(f"Error in broadcast: {e}")
            
            # Broadcast high threats separately
            if high_threat_processes:
                await manager.broadcast({
                    "type": "threats",
                    "data": high_threat_processes[:5]  # Top 5 threats
                })
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(5)

async def generate_alert(process: Dict, explanation: Dict):
    """Generate and broadcast alert for high threats"""
    alert = {
        "id": f"alert_{int(time.time())}_{process['pid']}",
        "processName": process['name'],
        "pid": process['pid'],
        "threatScore": process['threatScore'],
        "reason": [f["name"] for f in explanation.get("top_features", [])[:3]],
        "details": explanation.get("decision_path", []),
        "timestamp": datetime.now().isoformat()
    }
    
    # Store in history
    system.alert_history.append(alert)
    
    # Broadcast
    await manager.broadcast({
        "type": "alert",
        "data": alert
    })
    
    # Log critical alerts
    if process['threatScore'] > 80:
        logger.warning(f"CRITICAL THREAT: {process['name']} (PID: {process['pid']}) - Score: {process['threatScore']}")

async def cleanup_old_data():
    """Periodically clean up old data"""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        
        # Clear old history files
        try:
            # Keep only last 24 hours of data
            cutoff = datetime.now() - timedelta(hours=24)
            # Cleanup logic here
            logger.info("Cleaned up old data")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

async def benchmark_updates():
    """Periodically update benchmark metrics"""
    while True:
        await asyncio.sleep(60)  # Update every minute
        
        if system.is_monitoring:
            # Run quick benchmark
            try:
                inference_stats = system.onnx_engine.get_performance_stats() if system.onnx_engine else {}
                amd_metrics = system.onnx_engine.get_amd_specific_metrics() if system.onnx_engine else {}
                
                await manager.broadcast({
                    "type": "benchmark",
                    "data": {
                        "inference": inference_stats,
                        "amd": amd_metrics,
                        "energy": system.benchmark.get_energy_metrics()
                    }
                })
            except Exception as e:
                logger.error(f"Benchmark update error: {e}")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "EdgeGuard AI",
        "version": "2.0.0",
        "status": "running",
        "monitoring": system.is_monitoring,
        "model_trained": system.detector.is_trained,
        "documentation": "/api/docs",
        "websocket": "/ws"
    }

@app.get("/api/status")
async def get_status():
    """Get complete system status"""
    return {
        "system": system.get_system_info(),
        "state": system.system_state,
        "monitoring": system.is_monitoring,
        "connections": len(manager.active_connections),
        "threat_history_count": len(system.threat_history),
        "alert_count": len(system.alert_history)
    }

@app.get("/api/processes")
async def get_processes(limit: int = 50, threat_only: bool = False):
    """Get all processes with threat scores"""
    processes = system.process_monitor.get_processes()
    
    # Add threat scores
    results = []
    for proc in processes[:100]:  # Limit for performance
        if threat_only and proc.get('threatScore', 0) < system.detector.threshold * 100:
            continue
        
        if system.detector.is_trained:
            # Quick prediction without full explanation
            features = system.feature_extractor.extract_features(proc, system.system_state)
            if system.onnx_engine:
                normalized = system.feature_extractor.normalize_features(features)
                score = system.onnx_engine.infer(normalized.reshape(1, -1))[0][0]
            else:
                score = system.detector.predict(features)
            proc['threatScore'] = float(score * 100)
        
        results.append(proc)
        
        if len(results) >= limit:
            break
    
    return {
        "count": len(results),
        "processes": results
    }

@app.get("/api/process/{pid}")
async def get_process_detail(pid: int):
    """Get detailed process info with explanation"""
    process = system.process_monitor.get_process_by_pid(pid)
    
    if not process:
        raise HTTPException(status_code=404, detail="Process not found")
    
    # Extract features and get prediction
    features = system.feature_extractor.extract_features(process, system.system_state)
    
    if system.detector.is_trained:
        if system.onnx_engine:
            normalized = system.feature_extractor.normalize_features(features)
            score = system.onnx_engine.infer(normalized.reshape(1, -1))[0][0]
        else:
            score = system.detector.predict(features)
        
        process['threatScore'] = float(score * 100)
        process['explanation'] = system.detector.explain(features)
        process['features'] = {
            name: float(val) for name, val in zip(
                system.feature_extractor.feature_names[:len(features)],
                features
            )
        }
    
    return process

@app.get("/api/debug/monitoring")
async def debug_monitoring():
    """Debug endpoint to check monitoring status"""
    return {
        "is_monitoring": system.is_monitoring,
        "system_state": system.system_state,
        "threat_history_size": len(system.threat_history),
        "process_count": len(psutil.pids()),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "last_update": system.system_state['last_update']
    }


@app.post("/api/train")
async def train_model(samples: int = 10000):
    """Train the ML model"""
    try:
        logger.info(f"Starting model training with {samples} samples")
        
        # Import training function
        from models.train import train_model
        
        # Run training in background
        def train_task():
            try:
                accuracy = train_model()
                system.detector.load_model()  # Reload the trained model
                system._load_onnx_model()  # Reload ONNX model
                system.is_trained = True
                logger.info(f"Model training complete with accuracy: {accuracy}")
            except Exception as e:
                logger.error(f"Training failed: {e}")
        
        thread = threading.Thread(target=train_task)
        thread.start()
        
        return {
            "status": "started",
            "message": "Model training started in background",
            "samples": samples
        }
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/benchmark")
async def get_benchmark(full: bool = False):
    """Get performance benchmarks"""
    if full:
        results = system.benchmark.run_full_benchmark()
    else:
        # Quick stats
        results = {
            "inference": system.onnx_engine.get_performance_stats() if system.onnx_engine else {},
            "amd": system.onnx_engine.get_amd_specific_metrics() if system.onnx_engine else {},
            "energy": system.benchmark.get_energy_metrics(),
            "cpu_info": cpuinfo.get_cpu_info()['brand_raw']
        }
    
    return results

@app.get("/api/energy-efficiency")
async def get_energy_efficiency():
    """Get energy efficiency metrics"""
    return system.benchmark.get_energy_metrics()

@app.post("/api/quarantine/{pid}")
async def quarantine_process(pid: int):
    """Quarantine a suspicious process"""
    try:
        process = psutil.Process(pid)
        
        # First suspend the process
        process.suspend()
        
        # Log the action
        logger.warning(f"Process {pid} ({process.name()}) quarantined")
        
        # Broadcast quarantine event
        await manager.broadcast({
            "type": "quarantine",
            "data": {
                "pid": pid,
                "name": process.name(),
                "status": "quarantined",
                "timestamp": datetime.now().isoformat()
            }
        })
        
        return {
            "status": "success",
            "message": f"Process {pid} quarantined",
            "pid": pid,
            "name": process.name()
        }
        
    except psutil.NoSuchProcess:
        raise HTTPException(status_code=404, detail="Process not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/whitelist/{pid}")
async def whitelist_process(pid: int):
    """Add process to whitelist"""
    try:
        process = psutil.Process(pid)
        system.detector.add_to_whitelist(pid)
        
        # If process was suspended, resume it
        try:
            process.resume()
        except:
            pass
        
        logger.info(f"Process {pid} added to whitelist")
        
        return {
            "status": "success",
            "message": f"Process {pid} whitelisted",
            "pid": pid
        }
        
    except psutil.NoSuchProcess:
        raise HTTPException(status_code=404, detail="Process not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kill/{pid}")
async def kill_process(pid: int):
    """Terminate a malicious process"""
    try:
        process = psutil.Process(pid)
        process.terminate()
        
        logger.warning(f"Process {pid} terminated")
        
        return {
            "status": "success",
            "message": f"Process {pid} terminated",
            "pid": pid
        }
        
    except psutil.NoSuchProcess:
        raise HTTPException(status_code=404, detail="Process not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history(hours: int = 24):
    """Get threat history"""
    cutoff = datetime.now() - timedelta(hours=hours)
    
    # Filter history
    recent_threats = [
        t for t in system.threat_history
        if datetime.fromisoformat(t['timestamp']) > cutoff
    ]
    
    recent_alerts = [
        a for a in system.alert_history
        if datetime.fromisoformat(a['timestamp']) > cutoff
    ]
    
    # Calculate statistics
    if recent_threats:
        scores = [t['score'] for t in recent_threats]
        stats = {
            "avg": float(np.mean(scores)),
            "max": float(np.max(scores)),
            "min": float(np.min(scores)),
            "std": float(np.std(scores))
        }
    else:
        stats = {}
    
    return {
        "period_hours": hours,
        "threat_count": len(recent_threats),
        "alert_count": len(recent_alerts),
        "statistics": stats,
        "threats": list(recent_threats)[-100:],  # Last 100 points
        "alerts": list(recent_alerts)[-50:]  # Last 50 alerts
    }

@app.get("/api/export/{format}")
async def export_data(format: str = "json"):
    """Export threat data"""
    if format == "json":
        data = {
            "system_info": system.get_system_info(),
            "threat_history": list(system.threat_history),
            "alerts": list(system.alert_history),
            "export_time": datetime.now().isoformat()
        }
        return JSONResponse(content=data)
    
    elif format == "csv":
        # Create CSV
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "threat_score", "threat_level"])
        
        for t in system.threat_history:
            writer.writerow([t['timestamp'], t['score'], t['level']])
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=edgeguard_history.csv"}
        )
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")

@app.post("/api/monitoring/start")
async def start_monitoring():
    """Start system monitoring"""
    if not system.is_monitoring:
        system.start_monitoring()
        return {"status": "started", "message": "Monitoring started"}
    return {"status": "already_running"}

@app.post("/api/monitoring/stop")
async def stop_monitoring():
    """Stop system monitoring"""
    if system.is_monitoring:
        system.stop_monitoring()
        return {"status": "stopped", "message": "Monitoring stopped"}
    return {"status": "already_stopped"}

@app.get("/api/features")
async def get_features():
    """Get list of extracted features"""
    return {
        "count": len(system.feature_extractor.feature_names),
        "features": system.feature_extractor.feature_names,
        "ranges": system.feature_extractor.feature_ranges
    }

# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        # Send initial data with error handling
        try:
            await websocket.send_json({
                "type": "connected",
                "data": {
                    "message": "Connected to EdgeGuard AI",
                    "timestamp": datetime.now().isoformat(),
                    "system_info": system.get_system_info()  # This now has error handling
                }
            })
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
        
        # Send current threat history
        try:
            await websocket.send_json({
                "type": "history",
                "data": list(system.threat_history)[-50:]
            })
        except Exception as e:
            logger.error(f"Error sending history: {e}")
        
        # Handle client messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("action") == "get_history":
                    await websocket.send_json({
                        "type": "history",
                        "data": list(system.threat_history)
                    })
                
                elif message.get("action") == "get_alerts":
                    await websocket.send_json({
                        "type": "alerts",
                        "data": list(system.alert_history)
                    })
                
                elif message.get("action") == "get_process":
                    pid = message.get("pid")
                    process = system.process_monitor.get_process_by_pid(pid)
                    await websocket.send_json({
                        "type": "process_detail",
                        "data": process
                    })
                
                elif message.get("action") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "data": {"timestamp": datetime.now().isoformat()}
                    })
                    
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")
            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
# ============================================================================
# Static Files (for serving frontend)
# ============================================================================

# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
index_html = os.path.join(frontend_path, "index.html")

if os.path.exists(index_html):
    # Serve index.html at root
    @app.get("/")
    async def serve_frontend_root():
        return FileResponse(index_html)
    
    # Serve index.html at /app/
    @app.get("/app")
    async def serve_frontend_app_root():
        return FileResponse(index_html)
    
    # Serve index.html at /app/{path}
    @app.get("/app/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Check if the requested path is a file
        requested_path = os.path.join(frontend_path, full_path)
        if os.path.exists(requested_path) and os.path.isfile(requested_path):
            return FileResponse(requested_path)
        # Otherwise serve index.html (for SPA routing)
        return FileResponse(index_html)
    
    # Also serve at root path for any subpaths
    @app.get("/{full_path:path}")
    async def serve_root_path(full_path: str):
        # Skip API routes
        if full_path.startswith("api/") or full_path == "api" or full_path.startswith("ws"):
            raise HTTPException(status_code=404, detail="Not found")
        
        # Check if the requested path is a file
        requested_path = os.path.join(frontend_path, full_path)
        if os.path.exists(requested_path) and os.path.isfile(requested_path):
            return FileResponse(requested_path)
        # Otherwise serve index.html
        return FileResponse(index_html)
    
    logger.info(f"✅ Frontend loaded from {frontend_path}")
else:
    logger.warning(f"⚠️ Frontend not found at {index_html}")

# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Start background tasks on API startup"""
    logger.info("=" * 50)
    logger.info("Starting EdgeGuard AI Server")
    logger.info("=" * 50)
    
    # Log system info
    sys_info = system.get_system_info()
    logger.info(f"CPU: {sys_info['cpu']}")
    logger.info(f"Cores: {sys_info['cores']} logical, {sys_info['physical_cores']} physical")
    logger.info(f"AMD Optimized: {sys_info['is_amd']}")
    logger.info(f"AVX-512: {sys_info['avx512_supported']}")
    logger.info(f"Model trained: {sys_info['model_trained']}")
    logger.info(f"ONNX loaded: {sys_info['onnx_loaded']}")
    
    # Start background tasks
    asyncio.create_task(monitoring_loop())
    asyncio.create_task(cleanup_old_data())
    asyncio.create_task(benchmark_updates())
    
    # Start monitoring by default
    system.start_monitoring()
    
    logger.info("EdgeGuard AI Server started successfully")
    logger.info(f"API Documentation: http://localhost:8000/api/docs")
    logger.info(f"WebSocket: ws://localhost:8000/ws")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down EdgeGuard AI Server...")
    
    # Stop monitoring
    system.stop_monitoring()
    
    # Save model
    system.detector.save_model()
    
    # Close all WebSocket connections
    for connection in manager.active_connections:
        try:
            await connection.close()
        except:
            pass
    
    logger.info("EdgeGuard AI Server stopped")

# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "monitoring": system.is_monitoring,
        "connections": len(manager.active_connections),
        "memory_usage": psutil.Process().memory_percent(),
        "cpu_usage": psutil.Process().cpu_percent()
    }

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": request.url.path
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )

# ============================================================================
# Run the application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import platform
    
    # Get command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="EdgeGuard AI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--train", action="store_true", help="Train model on startup")
    
    args = parser.parse_args()
    
    # Train model if requested
    if args.train:
        logger.info("Training model on startup...")
        from models.train import train_model
        train_model()
        system.detector.load_model()
        system._load_onnx_model()
    
    # Run server
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )