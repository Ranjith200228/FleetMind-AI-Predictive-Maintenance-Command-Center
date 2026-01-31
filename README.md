# ⚡ Fleet Reliability Predictive Maintenance  
### Tesla-Style ML System for Industrial Equipment Failure Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange?logo=amazonaws)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)

---

## 🚀 Live Demo

👉 **[Portfolio Landing Page](https://ranjith200228.github.io/Fleet-Reliability-Predictive-Maintenance/)**  
👉 **Dashboard (AWS Deployment – coming soon)**

---

## 🧠 Recruiter Summary

This project simulates a **real-world predictive maintenance platform** used in autonomous fleets and industrial IoT systems.

The system predicts **Remaining Useful Life (RUL)** of equipment using machine learning and deep learning models, enabling proactive maintenance decisions that reduce downtime and operational cost.

This mirrors production ML pipelines used at companies like Tesla, Amazon Robotics, and industrial AI platforms.

---

## 🏗 Architecture Overview

```
Sensor Data → Feature Engineering → ML Models → Decision Engine → Dashboard
                    ↓
              AWS Cloud Storage
                    ↓
              Streamlit App (Docker)
```

📌 Architecture Diagram:  
![Architecture](portfolio/architecture.png)

---

## 📊 Core Capabilities

✅ Predict Remaining Useful Life (RUL)  
✅ Random Forest + LSTM hybrid modeling  
✅ Failure decision thresholds  
✅ Fleet-level maintenance planning  
✅ Interactive dashboard visualization  
✅ Cloud-ready container deployment  
✅ Scalable ML pipeline structure  

---

## 🧪 Machine Learning Stack

| Component | Technology |
|----------|-----------|
Feature Engineering | NumPy / Pandas
Classical ML | Random Forest
Deep Learning | LSTM (TensorFlow/Keras)
Model Fusion | Ensemble Meta Model
Visualization | Streamlit Dashboard
Deployment | Docker + AWS
Storage | S3 artifact pipeline

---

## 📂 Project Structure

```
Fleet-Reliability-Predictive-Maintenance/
│
├── data/raw/                 → Sensor datasets
├── src/                      → ML pipeline modules
├── phase1_test.py            → Data validation
├── phase2_test.py            → Model testing
├── phase3_train.py           → Training pipeline
├── phase4_decisions.py       → Maintenance logic
├── phase5_fleet_report.py    → Fleet analytics
├── train_seq_tf.py           → LSTM training
├── requirements.txt
└── README.md
```

---

## 📈 Dashboard Preview

![Dashboard](portfolio/dashboard.gif)

Interactive dashboard shows:

• Predicted failure timelines  
• Fleet health status  
• Risk classification  
• Maintenance priority scoring  

---

## ☁ AWS Deployment (Production Design)

```
Docker Container → AWS ECR → AWS App Runner / ECS
                       ↓
                   S3 Model Store
                       ↓
                 Public Dashboard URL
```

This architecture mirrors enterprise ML system deployment patterns.

---

## 🔧 How to Run Locally

```bash
git clone https://github.com/Ranjith200228/Fleet-Reliability-Predictive-Maintenance.git
cd Fleet-Reliability-Predictive-Maintenance

pip install -r requirements.txt
python phase3_train.py
streamlit run src/app/dashboard.py
```

---

## 🎯 Real-World Impact

This system models how large fleets:

• Prevent catastrophic equipment failure  
• Reduce operational downtime  
• Optimize maintenance scheduling  
• Save millions in logistics cost  
• Enable predictive AI infrastructure  

This is the same class of problem solved by:

Tesla • GE Aviation • Amazon Robotics • SpaceX • Industrial IoT platforms

---

## profile

**Ranjith Kumar Maddirala**  
Data Science & Machine Learning Engineer   

🔗 LinkedIn: https://linkedin.com/in/ranjith-kumar-maddirala-5426801bb  
🌐 Portfolio: https://ranjith-x-data-core.lovable.app


---

## 📌 Future Enhancements

• Real-time streaming sensor ingestion  
• Auto model retraining pipeline  
• Fleet anomaly detection  
• Kubernetes scaling  
• Edge deployment simulation  

---

> Built with production ML engineering mindset ⚡

