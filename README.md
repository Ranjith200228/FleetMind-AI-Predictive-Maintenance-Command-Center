# FleetMind — Intelligent Predictive Maintenance Platform
### AI • Machine Learning • Decision Intelligence • Operational Systems

---

## Executive Summary

FleetMind is a production-style machine learning platform designed to predict equipment failure **before it occurs**, prioritize operational risk across an entire fleet, and provide AI-assisted maintenance recommendations through an LLM-powered Copilot.

This system moves beyond traditional notebook-based ML by delivering a **decision-grade operational interface** capable of supporting real-world reliability engineering workflows.

FleetMind demonstrates the architecture, engineering discipline, and system-level thinking expected from modern Machine Learning Engineers.

---

## Why This Project Matters

Unplanned industrial downtime costs the global economy **hundreds of billions annually**.

Most ML projects stop at prediction.

Very few solve the harder problem:

> Turning predictions into operational decisions.

FleetMind was built to close that gap.

It transforms raw sensor telemetry into prioritized maintenance actions — enabling organizations to transition from reactive maintenance toward intelligence-driven operations.

---

## System Capabilities

### Predict Failure Before It Happens
- Remaining Useful Life (RUL) prediction  
- Health index scoring  
- Dynamic risk classification  

---

### Fleet Command Center
A real-time operational view designed for reliability teams.

Includes:

- Fleet-wide risk posture  
- Priority swimlanes  
- Maintenance queue  
- Decision indicators  
- Next-action recommendations  

Built with executive-level clarity.

---

### Ops Copilot (LLM Decision Engine)

FleetMind integrates a structured AI Copilot capable of:

- Diagnosing likely failure drivers  
- Explaining risk signals  
- Generating prioritized actions  
- Asking operational follow-ups  
- Producing schema-validated outputs  

Critically — the system is designed so malformed LLM responses **cannot crash the application.**

This reflects production-grade defensive engineering.

---

### Failure Scenario Simulator

A parameterized simulation engine allows operators to stress-test fleet behavior under realistic degradation patterns:

- Bearing instability  
- Thermal runaway  
- Compressor shock events  
- Sensor drift  
- Gradual mechanical wear  

This enables proactive planning rather than reactive firefighting.

---

### Projected Failure Countdown

Forecasts when an engine is likely to cross operational thresholds.

Provides:

- Risk countdown  
- Confidence bands  
- Planning visibility  

Supports maintenance scheduling decisions before escalation occurs.

---

### Automated Maintenance Queue

FleetMind converts model outputs into actionable operations:

SERVICE_NOW → Immediate intervention  
MONITOR → Scheduled inspection  
OK → Continue operation  

The system ranks engines automatically — optimizing operator attention where it matters most.

---

### One-Click Executive Reporting

Generates professional PDF reports containing:

- Fleet snapshot  
- Risk distribution  
- Copilot insights  
- Recommended actions  
- Engine trajectory  

Designed for leadership visibility.

---

## Architecture Overview

FleetMind was intentionally designed as a modular ML system rather than a monolithic dashboard.

```mermaid
flowchart TD

A[Sensor Telemetry] --> B[Feature Engineering]
B --> C[ML Failure Prediction Model]

C --> D[Risk Classification Engine]
D --> E[Priority Swimlanes]

C --> F[Ops Copilot Interface]
F --> G[LLM Reasoning Layer]

E --> H[Operational Command Center]
G --> H

H --> I[Maintenance Queue]
H --> J[Failure Simulator]
H --> K[Executive Reporting Engine]

Architectural Philosophy
FleetMind was built around principles commonly seen in production ML platforms.

Modular System Design
Each capability exists as an isolated panel:

src/app/panels/
Improves maintainability and extensibility.

LLM Abstraction Layer
src/copilot/
Separates AI reasoning from UI logic — enabling future provider swaps without architectural disruption.

Schema-Enforced AI Responses
Every Copilot response is validated against a strict schema before entering the system.

This prevents:

UI failures

malformed outputs

unpredictable behaviors

A critical production safeguard.

Safe Fallback Strategy
If the LLM is unavailable:

→ Mock Copilot activates
→ System remains operational

Reliability is preserved.

Stateful UX Architecture
Session-state persistence ensures:

Engine selections survive reruns

Simulation parameters remain stable

Copilot outputs persist

Creating a seamless operational experience.

Technology Stack
Machine Learning
Random Forest predictive modeling

Rolling statistical feature engineering

Health index generation

Threshold-based decisioning

AI Layer
OpenAI structured reasoning

JSON response enforcement

Deterministic temperature tuning

Application Layer
Streamlit production architecture

Panel-based modular UI

Resilient state management

Visualization
Plotly interactive analytics

Executive-grade dashboard styling

Premium dark theme optimized for readability

Example Operational Flow
Sensor data enters the system

Features are engineered

Model predicts RUL

Risk engine classifies priority

Copilot explains the risk

Queue auto-generates

Executive report exports

This mirrors real-world reliability pipelines.

Engineering Challenges Solved
Preventing LLM Schema Drift
Implemented strict validation to enforce response structure.

Designing Failure-Resilient UX
Fallback layers ensure the dashboard never crashes.

Bridging Prediction → Decision
Converted ML outputs into operational workflows.

Simulating Realistic Degradation
Built parameterized failure models to emulate industrial behavior.

Business Impact (Modeled)
Predictive maintenance strategies can reduce downtime by 30–50%.

FleetMind illustrates how such outcomes become operationally achievable.

Running Locally
git clone <repo>
cd fleetmind

pip install -r requirements.txt
streamlit run src/app/dashboard.py
Environment Setup
Create a .env file:

OPENAI_API_KEY=your_key
Without a key, FleetMind automatically switches to a mock Copilot — preserving functionality.

About the me;
Ranjith Kumar Maddirala

Focused on building intelligent systems that transform data into operational decisions.

Core Interests:

Machine Learning Systems

Applied AI

Predictive Analytics

Decision Platforms

Reliability Engineering
