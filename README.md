# ⚡ Fleet Reliability & Predictive Maintenance Dashboard (RUL + Fleet Health)

A production-style **fleet reliability & predictive maintenance** dashboard built using **Streamlit + Plotly + ML**.
It predicts **Remaining Useful Life (RUL)**, generates a **Health Index**, flags **anomalies**, and supports **live inference** via uploaded turbofan files.

## 🚀 Live Demo
👉 <PASTE_STREAMLIT_URL_AFTER_DEPLOY>

## Key Capabilities
- **Fleet Snapshot:** SERVICE_NOW / MONITOR / OK decisioning
- **RUL Prediction:** engine-level RUL estimates across cycles
- **Explainability:** feature importance + clean “Top Sensors” summary
- **Anomaly Timeline:** sensor spike detection (rolling z-score)
- **Early Warning Metric:** lead time before SERVICE_NOW threshold
- **Live Inference:** upload FD001-format `.txt` → download predictions CSV

## Tech Stack
- Python, Pandas, NumPy
- Streamlit (UI)
- Plotly (charts)
- scikit-learn + joblib (RF model bundle)
- TensorFlow (LSTM optional)

## Run Locally
```bash
python -m venv .venv
# activate it
pip install -r requirements.txt
python -m streamlit run src/app/dashboard.py
