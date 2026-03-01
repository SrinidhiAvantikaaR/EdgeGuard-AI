#  EdgeGuard AI

### AI-Powered Adaptive Ransomware Detection Optimized for AMD Edge Devices

---

##  Problem Statement

Ransomware encrypts files at high speed, often bypassing traditional signature-based antivirus systems. By the time a signature is identified, significant data damage has already occurred.

There is a need for a **real-time, signature-less, lightweight ransomware detection system** that operates efficiently on edge devices and leverages modern CPU capabilities.

---

##  Solution Overview

**EdgeGuard AI** is a real-time behavioral ransomware detection system that:

* Monitors system-level activity
* Profiles CPU and file behavior
* Detects anomalies using machine learning
* Runs efficiently on AMD edge processors
* Provides explainable AI insights

Instead of relying on known signatures, EdgeGuard AI uses **behavioral anomaly detection** to identify ransomware activity at early stages.

---

##  Key Features

###  Behavioral Monitoring

* File modification rate tracking
* File entropy change detection
* CPU usage spikes
* Suspicious process creation patterns

###  Machine Learning Engine

* Isolation Forest
* Optional lightweight LSTM for sequence modeling
* Trained only on normal system activity
* Real-time anomaly scoring

### AMD Optimization

* Optimized inference using ONNX Runtime
* Multi-threaded execution leveraging AMD Ryzen cores
* Low-latency inference benchmarking
* Edge deployment capability

### Dashboard

* Live anomaly score visualization
* Suspicious process flagging
* Hardware utilization monitoring
* Explainable AI feature contribution breakdown
* Attack simulation mode

---

## System Architecture

```
System Activity Monitor
        ↓
Feature Extraction Engine
        ↓
ML Anomaly Detection Model (ONNX Runtime)
        ↓
Threat Evaluation Layer
        ↓
Real-Time Dashboard + Alert System
```

---

## AI Approach

We use unsupervised anomaly detection:

* Model trained on normal system behavior
* Computes anomaly score in real-time
* Triggers alert if score exceeds threshold

Optional:

* Sequence modeling using lightweight LSTM
* Feature importance scoring for explainability

---

##  Dataset Strategy

We use:

* Synthetic ransomware-like behavior generation
* Custom feature extraction pipeline

---

##  Performance & Benchmarking

EdgeGuard AI is optimized for AMD edge devices:

* Low inference latency
* Efficient multi-core utilization
* Reduced energy footprint
* Real-time responsiveness under system load

Benchmark metrics include:

* Inference latency (ms)
* CPU utilization (%)
* Threads used
* Anomaly detection accuracy

---

##  Why EdgeGuard AI?

✔ Signature-less detection
✔ Early ransomware interception
✔ Edge deployment ready
✔ Hardware-aware optimization
✔ Lightweight ML model
✔ Explainable AI insights

---

##  Future Enhancements

* Auto-process termination
* Adaptive threshold tuning
* Federated learning for edge collaboration
* GPU acceleration support
* Enterprise deployment mode

---

##  Tech Stack

* Python
* Scikit-learn
* ONNX Runtime
* psutil
* Streamlit / React Dashboard
* AMD Ryzen Multi-thread Optimization

---

##  Use Cases

* Personal systems
* Enterprise endpoints
* Edge computing devices
* IoT security gateways
* Industrial monitoring systems

---
