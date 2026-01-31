<p align="center">
  <h1 align="center">🚀 Fleet Reliability Predictive Maintenance</h1>
  <h3 align="center">Production-grade ML system for predictive engine maintenance</h3>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker">
  <img src="https://img.shields.io/badge/AWS-App%20Runner-orange?logo=amazonaws">
  <img src="https://img.shields.io/badge/AWS-S3-yellow?logo=amazonaws">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit">
  <img src="https://img.shields.io/badge/ML-Predictive%20Maintenance-green">
</p>

---

##  Live Demo

 Deploying on AWS App Runner…  
(Live URL will appear here after activation)

---

##  Executive Summary

This project is a production-style predictive maintenance system that forecasts engine failure risk and transforms raw telemetry into actionable maintenance decisions.

It demonstrates the full lifecycle of modern ML engineering:

data ingestion → feature engineering → model training → deployment → interactive decision dashboard → cloud architecture

The system is designed for fleet operators to shift from reactive repairs to predictive reliability.

---

##  Dashboard Preview

> Animated dashboard preview (GIF will be added here)

<p align="center">
  <img src="docs/dashboard-preview.gif" width="90%">
</p>

---

##  System Architecture

<p align="center">
  <img src="docs/architecture-diagram.png" width="90%">
</p>

```
Streamlit Dashboard (Docker Container)
        ↓
AWS App Runner
        ↓
IAM Role (Least Privilege)
        ↓
Amazon S3 (Model Artifacts)
```

This architecture mirrors real-world ML production deployments.

---

##  Project Capabilities

✔ Remaining Useful Life prediction  
✔ Engine risk classification  
✔ Maintenance urgency decisions  
✔ Fleet health monitoring  
✔ Interactive dashboard analytics  
✔ Cloud-native deployment  

---

##  Machine Learning Pipeline

### Dataset
NASA Turbofan Engine Degradation (FD001)

### Models
- Random Forest regression
- LSTM neural network
- Decision ensemble logic

### Features
- Sensor time-series
- Engine cycle trends
- Health degradation signals
- Feature normalization

---

##  Cloud Deployment

The dashboard is containerized and deployed using AWS:

- Docker containerization
- AWS App Runner hosting
- S3 artifact storage
- IAM least-privilege access
- No hardcoded credentials

This mirrors industry production ML workflows.

---

##  Tech Stack

Python • Streamlit • scikit-learn • TensorFlow  
Docker • AWS App Runner • AWS S3 • IAM  
Pandas • NumPy • ML Ops principles

---

##  Local Setup

Clone:

```bash
git clone https://github.com/<your-username>/Fleet-Reliability-Predictive-Maintenance.git
cd Fleet-Reliability-Predictive-Maintenance
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run dashboard:

```bash
streamlit run src/app/dashboard.py
```

---

## 📂 Project Structure

```
src/
  app/        dashboard UI + decision engine
  models/     ML pipelines
data/
  raw/        NASA dataset
artifacts/
  models/     trained models (stored in S3)
```

---

##  Skills Demonstrated

- Predictive maintenance analytics
- Time-series modeling
- ML deployment architecture
- Docker containerization
- Cloud ML infrastructure
- Secure IAM design
- Interactive dashboard engineering
- Production ML workflows

---

##  Future Improvements

- Real-time telemetry streaming
- REST API endpoints
- Automated retraining pipeline
- Alerting & monitoring
- Cloud observability integration
