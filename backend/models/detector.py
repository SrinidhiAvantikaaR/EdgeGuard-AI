"""
Ransomware Detector using Isolation Forest
With ONNX export for AMD optimization
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import cpuinfo

logger = logging.getLogger(__name__)

class RansomwareDetector:
    """Real-time ransomware detection using Isolation Forest"""
    
    def __init__(self, model_path="data/models/isolation_forest.pkl", 
                 onnx_path="data/models/model.onnx"):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.threshold = 0.5  # Anomaly threshold (0-1)
        self.feature_names = [
            'cpu_percent', 
            'memory_percent', 
            'file_writes_rate',
            'entropy', 
            'num_threads', 
            'connections', 
            'cpu_burst',
            'sudden_cpu_change', 
            'file_type_changes'
        ]
        self.model_path = model_path
        self.onnx_path = onnx_path
        self.last_inference_time = 0
        self.session = None
        self.whitelist = set()  # Store whitelisted PIDs
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Load model if exists
        self.load_model()
        
        # Try to load ONNX model
        self._load_onnx_session()
        self.session = None
    
    def _load_onnx_session(self):
        """Load ONNX Runtime session if model exists"""
        if os.path.exists(self.onnx_path):
            try:
                providers = ['CPUExecutionProvider']
                if self._is_amd_processor():
                    try:
                        # Try to use OpenVINO for AMD optimization
                        providers.insert(0, 'OpenVINOExecutionProvider')
                        logger.info("OpenVINO Execution Provider enabled for AMD")
                    except:
                        pass
                
                self.session = ort.InferenceSession(
                    self.onnx_path, 
                    providers=providers
                )
                logger.info("ONNX model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load ONNX model: {e}")
                self.session = None
    
    def _is_amd_processor(self) -> bool:
        """Check if processor is AMD"""
        try:
            cpu_brand = cpuinfo.get_cpu_info()['brand_raw']
            return 'AMD' in cpu_brand
        except:
            return False
    
    def train(self, training_data: List[Dict]) -> float:
        """Train the Isolation Forest model"""
        logger.info(f"Training model with {len(training_data)} samples")
        
        try:
            # Convert to DataFrame
            
            df = pd.DataFrame(training_data)
            
            # CRITICAL: Only train on normal samples if label is available
            if 'label' in df.columns:
                normal_df = df[df['label'] == 0]
            else:
                normal_df = df  # fallback
            
            X = normal_df[self.feature_names].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = IsolationForest(
                contamination=0.05,  # Expect ~5% anomalies in real traffic
                random_state=42,
                n_estimators=200,    # More trees = better boundaries
                max_samples=256,     # Standard recommendation
                n_jobs=-1
            )
            self.model.fit(X_scaled)
            self.is_trained = True
            
            # Calculate training accuracy (on training data)
            predictions = self.model.predict(X_scaled)
            accuracy = np.mean(predictions == 1)  # 1 for normal, -1 for anomaly
            
            # Save model
            self.save_model()
            
            # Export to ONNX for AMD optimization
            self.export_to_onnx()
            
            logger.info(f"Model trained with accuracy: {accuracy:.2f}")
            return float(accuracy)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def predict(self, features: Any) -> float:
        """Predict anomaly score (0-1, higher = more anomalous)"""
        if not self.is_trained or self.model is None:
            return 0.0
        
        start_time = time.time()
        
        try:
            X = self._prepare_features(features) 
            # Prepare features
            if self.session:  # ← ONNX path — THIS is where the fix goes
                X_scaled = self.scaler.transform(X).astype(np.float32)
                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: X_scaled})
                
                if outputs and len(outputs) > 0:
                    output = outputs[0]
                    if isinstance(output, np.ndarray):
                        if output.size == 1:
                            raw = float(output.item())
                        else:
                            raw = float(output[0][0]) if output.ndim > 1 else float(output[0])
                    else:
                        raw = float(output)
                    print("ONNX raw output: ", raw)
                    
                    # ← REPLACE the old anomaly_score line with this:
                    anomaly_score = float(1 / (1 + np.exp(raw * 2)))
                else:
                    anomaly_score = 0.5
            
            else:  # ← sklearn fallback path — fix goes here too
                X_scaled = self.scaler.transform(X)
                decision = self.model.decision_function(X_scaled)
                # ← REPLACE old line (had a minus sign) with this:
                anomaly_score = float(1 / (1 + np.exp(decision[0])))  # no negation
            
            self.last_inference_time = (time.time() - start_time) * 1000
            return float(anomaly_score)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _prepare_features(self, features: Any) -> np.ndarray:
        """Convert input to numpy array"""
        try:
            if isinstance(features, dict):
                # Dictionary input
                values = []
                for name in self.feature_names:
                    val = features.get(name, 0)
                    if isinstance(val, (int, float, np.number)):
                        values.append(float(val))
                    elif isinstance(val, (list, np.ndarray)):
                        values.append(float(val[0]) if len(val) > 0 else 0)
                    else:
                        values.append(0.0)
                return np.array([values], dtype=np.float32)
                
            elif isinstance(features, (list, tuple)):
                # List input
                arr = np.array(features, dtype=np.float32)
                if arr.ndim == 1:
                    return arr.reshape(1, -1)
                return arr
                
            elif isinstance(features, np.ndarray):
                # NumPy array input
                if features.ndim == 1:
                    return features.reshape(1, -1).astype(np.float32)
                return features.astype(np.float32)
                
            else:
                logger.warning(f"Unexpected features type: {type(features)}")
                return np.zeros((1, len(self.feature_names)), dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.zeros((1, len(self.feature_names)), dtype=np.float32)
    
    def _predict_sklearn(self, X: np.ndarray) -> float:
        """Predict using scikit-learn model"""
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest returns -1 for anomalies, 1 for normal
        # Convert to anomaly score (0-1) using decision function
        decision = self.model.decision_function(X_scaled)
        
        # Sigmoid to map to 0-1 range (higher = more anomalous)
        anomaly_score = 1 / (1 + np.exp(-decision[0]))
        
        return anomaly_score
    
    def _predict_onnx(self, X: np.ndarray) -> float:
        """Predict using ONNX Runtime"""
        # Scale features
        X_scaled = self.scaler.transform(X).astype(np.float32)
        
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        outputs = self.session.run(None, {input_name: X_scaled})
        
        # Handle different output formats
        if outputs and len(outputs) > 0:
            if isinstance(outputs[0], np.ndarray):
                if outputs[0].size > 0:
                    # Convert output to anomaly score
                    raw_output = outputs[0][0]
                    # If output is decision function, convert to 0-1
                    if isinstance(raw_output, (np.floating, float)):
                        return 1 / (1 + np.exp(-float(raw_output)))
                    return float(raw_output)
        
        return 0.5
    
    def explain(self, features: Any) -> Dict:
        """Generate explanation for prediction"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        try:
            # Prepare features
            X = self._prepare_features(features)
            X_scaled = self.scaler.transform(X)[0]
            
            # Create feature dictionary for explanation
            if isinstance(features, dict):
                feature_dict = features
            else:
                feature_dict = {}
                for i, name in enumerate(self.feature_names):
                    if i < X.shape[1]:
                        feature_dict[name] = float(X[0][i])
                    else:
                        feature_dict[name] = 0.0
            
            # Get feature importances
            importances = self._get_feature_importances()
            
            # Calculate feature contributions
            feature_contributions = []
            for i, name in enumerate(self.feature_names):
                if i < len(importances):
                    contribution = float(importances[i] * abs(X_scaled[i]))
                else:
                    contribution = 0.0
                
                feature_contributions.append({
                    "name": name,
                    "value": float(feature_dict.get(name, X[0][i] if i < X.shape[1] else 0)),
                    "contribution": contribution,
                    "threshold": self._get_threshold(name)
                })
            
            # Sort by contribution
            feature_contributions.sort(key=lambda x: x['contribution'], reverse=True)
            
            # Get decision path
            decision_path = self._get_decision_path(feature_dict, X_scaled)
            
            # Get prediction
            anomaly_score = self.predict(X)
            
            return {
                "top_features": feature_contributions[:5],
                "all_features": feature_contributions,
                "decision_path": decision_path,
                "anomaly_score": float(anomaly_score),
                "threat_level": self._get_threat_level(anomaly_score)
            }
            
        except Exception as e:
            logger.error(f"Explain error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _get_feature_importances(self) -> np.ndarray:
        """Get feature importances from the model"""
        if hasattr(self.model, 'estimators_'):
            # Average feature importance across trees
            importances = []
            for tree in self.model.estimators_:
                if hasattr(tree, 'tree_'):
                    importances.append(tree.tree_.compute_feature_importances())
            
            if importances:
                return np.mean(importances, axis=0)
        
        # Fallback: equal importance
        return np.ones(len(self.feature_names)) / len(self.feature_names)
    
    def _get_threshold(self, feature_name: str) -> float:
        """Get typical threshold for feature"""
        thresholds = {
            'cpu_percent': 80.0,
            'file_writes_rate': 200.0,
            'entropy': 0.7,
            'sudden_cpu_change': 30.0,
            'file_type_changes': 10.0,
            'connections': 20.0,
            'num_threads': 50.0
        }
        return thresholds.get(feature_name, 0.0)
    
    def _get_threat_level(self, score: float) -> str:
        """Convert score to threat level"""
        if score < 0.3:
            return 'LOW'
        elif score < 0.6:
            return 'MEDIUM'
        elif score < 0.8:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _get_decision_path(self, feature_dict: Dict, scaled_features: np.ndarray) -> List[str]:
        """Generate human-readable decision path"""
        path = []
        
        # Check critical features by name
        file_writes = feature_dict.get('file_writes_rate', 0)
        if file_writes > 200:
            path.append(f"⚠️ High file write rate: {file_writes:.0f} writes/sec (normal: <200)")
        
        entropy = feature_dict.get('entropy', 0)
        if entropy > 0.7:
            path.append(f"🔐 High entropy: {entropy:.2f} - possible encryption (normal: <0.7)")
        
        cpu = feature_dict.get('cpu_percent', 0)
        if cpu > 80:
            path.append(f"⚡ CPU spike: {cpu:.0f}% (normal: <80%)")
        
        cpu_delta = feature_dict.get('sudden_cpu_change', 0)
        if abs(cpu_delta) > 30:
            path.append(f"📈 Sudden CPU change: {cpu_delta:.0f}%")
        
        file_changes = feature_dict.get('file_type_changes', 0)
        if file_changes > 10:
            path.append(f"📁 Multiple file type changes: {file_changes}")
        
        connections = feature_dict.get('connections', 0)
        if connections > 20:
            path.append(f"🌐 High network activity: {connections} connections")
        
        if not path:
            path.append("✅ Normal behavior pattern")
        
        return path
    
    def export_to_onnx(self):
        """Export model to ONNX format for AMD optimization"""
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained, cannot export to ONNX")
            return False
        
        try:
            # Create output directory
            os.makedirs(os.path.dirname(self.onnx_path), exist_ok=True)
            
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # Define input type
            initial_type = [('float_input', FloatTensorType([None, len(self.feature_names)]))]
            
            # CRITICAL FIX: Specify both domains with correct versions
            target_opset = {
                '': 12,              # Default domain
                'ai.onnx.ml': 3      # ML domain - set to version 3 as error suggests
            }
            
            # Convert to ONNX with domain-specific opset
            onnx_model = convert_sklearn(
                self.model, 
                initial_types=initial_type,
                target_opset=target_opset,  # Use dictionary format
                options={id(self.model): {'score_samples': True}}
            )
            
            # Save ONNX model
            with open(self.onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            logger.info(f"✅ Model exported to ONNX: {self.onnx_path}")
            
            # Test the ONNX model
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(
                    self.onnx_path, 
                    providers=['CPUExecutionProvider']
                )
                logger.info("✅ ONNX model verified and loaded")
                return True
            except Exception as e:
                logger.warning(f"ONNX verification failed: {e}")
                self.session = None
                return False
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            logger.info("Continuing with scikit-learn model")
            self.session = None
            return False
    
    def save_model(self):
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'threshold': self.threshold
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self):
        """Load model from disk"""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data.get('feature_names', self.feature_names)
                self.is_trained = model_data.get('is_trained', True)
                self.threshold = model_data.get('threshold', 0.6)
                logger.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.is_trained = False
        else:
            logger.info("No existing model found")
    
    def add_to_whitelist(self, pid: int):
        """Add process to whitelist"""
        self.whitelist.add(pid)
        logger.info(f"Added PID {pid} to whitelist")
    
    def remove_from_whitelist(self, pid: int):
        """Remove process from whitelist"""
        self.whitelist.discard(pid)
        logger.info(f"Removed PID {pid} from whitelist")
    
    def is_whitelisted(self, pid: int) -> bool:
        """Check if process is whitelisted"""
        return pid in self.whitelist
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'threshold': self.threshold,
            'onnx_loaded': self.session is not None,
            'whitelist_size': len(self.whitelist),
            'model_path': self.model_path,
            'onnx_path': self.onnx_path if os.path.exists(self.onnx_path) else None
        }


# Helper function to create detector instance
def create_detector(model_path: str = "data/models/isolation_forest.pkl") -> RansomwareDetector:
    """Create and configure ransomware detector"""
    detector = RansomwareDetector(model_path)
    
    # Log AMD optimization status
    if detector._is_amd_processor():
        logger.info("AMD processor detected - optimizations enabled")
        if detector.session:
            logger.info("ONNX acceleration active")
        else:
            logger.info("Using scikit-learn fallback")
    
    return detector


# Example usage
if __name__ == "__main__":
    # Test the detector
    detector = create_detector()
    
    # Generate some test data
    from models.train import generate_training_data
    
    print("Generating test data...")
    data = generate_training_data(1000)
    
    print("Training model...")
    accuracy = detector.train(data)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Test prediction
    test_features = {
        'cpu_percent': 85.0,
        'memory_percent': 70.0,
        'file_writes_rate': 300.0,
        'entropy': 0.85,
        'num_threads': 15,
        'connections': 10,
        'cpu_burst': 1,
        'sudden_cpu_change': 40.0,
        'file_type_changes': 50
    }
    
    score = detector.predict(test_features)
    print(f"Threat score: {score:.2f}")
    print(f"Threat level: {detector._get_threat_level(score)}")
    
    # Get explanation
    explanation = detector.explain(test_features)
    print("\nExplanation:")
    for feature in explanation.get('top_features', []):
        print(f"  {feature['name']}: {feature['value']} (contribution: {feature['contribution']:.2f})")

def calibrate_threshold(self, normal_processes: List[Dict]):
    """Calibrate the threshold based on normal system processes"""
    scores = []
    
    for proc in normal_processes:
        try:
            # Extract features for normal process
            features = {
                'cpu_percent': proc.get('cpu', 0),
                'memory_percent': proc.get('memory_percent', 0),
                'file_writes_rate': proc.get('file_writes', 0),
                'entropy': proc.get('entropy', 0.1),  # Normal processes have low entropy
                'num_threads': proc.get('num_threads', 1),
                'connections': proc.get('connections', 0),
                'cpu_burst': 0,  # Normal processes don't burst
                'sudden_cpu_change': 0,
                'file_type_changes': proc.get('file_type_changes', 0)
            }
            
            score = self.predict(features)
            scores.append(score)
            
        except Exception as e:
            logger.error(f"Calibration error: {e}")
    
    if scores:
        # Set threshold to 3 standard deviations above mean of normal scores
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        self.threshold = min(0.8, mean_score + 3 * std_score)
        logger.info(f"Calibrated threshold to: {self.threshold:.2f}")
    
    return self.threshold