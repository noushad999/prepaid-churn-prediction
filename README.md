
<div align="center">

# 🔮 Telecom Customer Churn Prediction System
### 🚀 The AI-Powered Retention Engine

**Identify Risks. Prevent Churn. Save Millions.**

<p>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://flask.palletsprojects.com/">
    <img src="https://img.shields.io/badge/Flask-API_Ready-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  </a>
  <a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  </a>
</p>

<p>
  <a href="#-business-impact">💰 Business Impact</a> •
  <a href="#-system-architecture">🏗️ Architecture</a> •
  <a href="#-quick-start-guide">🚀 Quick Start</a> •
  <a href="#-api-documentation">📡 API Docs</a> •
  <a href="#-author">👨‍💻 Author</a>
</p>

<img src="https://raw.githubusercontent.com/noushad999/telecom-churn-prediction/main/docs/demo_placeholder.png" alt="System Dashboard Preview" width="800" onerror="this.style.display='none'"/>

</div>

---

## 💰 Business Impact
Why does this project exist? To solve a **$4.2M** problem.

| 📉 Revenue at Risk | 🛡️ Potential Savings | 🎯 Prediction Accuracy | ⚡ Processing Speed |
|:---:|:---:|:---:|:---:|
| **$4.2M** | **$2.1M / year** | **75%** | **<100ms** |
| *Identified Exposure* | *Targeted Retention* | *Recall Score* | *Real-time API* |

> **Bottom Line:** By identifying high-risk customers early, we enable proactive retention campaigns that yield a **738% ROI**.

---

## 🏗️ System Architecture

A robust, end-to-end machine learning pipeline designed for production.

```mermaid
graph TD
    %% Define Styles
    classDef dark fill:#2d3436,stroke:#fff,stroke-width:2px,color:#fff;
    classDef ai fill:#6c5ce7,stroke:#fff,stroke-width:2px,color:#fff;
    classDef api fill:#00cec9,stroke:#fff,stroke-width:2px,color:#000;
    classDef ui fill:#fab1a0,stroke:#333,stroke-width:2px,color:#000;

    subgraph Data_Pipeline [📡 Data Pipeline]
        A[Customer Data] -->|Cleaning| B(Preprocessing)
        B -->|Feature Eng| C(ADASYN Balancing)
    end

    subgraph AI_Core [🧠 AI Core]
        C --> D[Deep Neural Network]
        D --> E[Prediction Engine]
    end

    subgraph Deployment [🚀 Deployment & UI]
        E --> F[REST API / Flask]
        F --> G[Web Interface]
        F --> H[Executive Reports]
    end

    %% Apply Styles
    class A,B,C dark;
    class D,E ai;
    class F api;
    class G,H ui;
````

-----

## 🚀 Quick Start Guide

Run the entire system in minutes using Docker or Python directly.

\<details open\>
\<summary\>\<b\>🐳 Option 1: Docker (Recommended)\</b\>\</summary\>
<br>

Perfect for instant setup without dependency hell.

```bash
# 1. Clone the repository
git clone [https://github.com/noushad999/telecom-churn-prediction.git](https://github.com/noushad999/telecom-churn-prediction.git)
cd telecom-churn-prediction

# 2. Build and Start
docker-compose up -d --build

# 3. Access
# UI: http://localhost:5001
```

\</details\>

\<details\>
\<summary\>\<b\>🐍 Option 2: Manual Python Setup\</b\>\</summary\>
<br>

If you want to modify the code or run locally.

```bash
# 1. Create Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install Deps
pip install -r requirements.txt

# 3. Initialize & Run
python download_data.py   # 1. Get Data
python main_simple.py     # 2. Train Model
python api.py             # 3. Start Server
```

\</details\>

-----

## 📡 API Documentation

Integrate the churn engine into your existing CRM.

### `POST /predict`

Get a real-time risk assessment for a single customer.

**Example Request:**

```json
{
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "tenure": 2,
  "MonthlyCharges": 95.50
}
```

**Example Response:**

```json
{
  "prediction": "Churn",
  "probability": 0.92,
  "risk_level": "Critical",
  "action": "Offer 20% discount immediately"
}
```

-----

## 📂 Project Structure

A clean, modular architecture separating concerns.

\<details\>
\<summary\>\<b\>📂 Click to view file tree\</b\>\</summary\>

```text
telecom-churn-prediction/
├── 📄 api.py                # 🌐 Flask REST API entry point
├── 🧠 main_simple.py        # 🏋️ Model training pipeline
├── 🐳 Dockerfile            # 📦 Container configuration
├── 📜 docker-compose.yml    # 🐙 Multi-container orchestration
├── 📋 requirements.txt      # 📦 Python dependencies
├── 📁 src/
│   ├── 🏗️ models/           # PyTorch neural network classes
│   ├── ⚙️ data_processing/  # Cleaning & feature engineering
│   └── 🧪 training/         # Training loops & validation
├── 📁 data/                 # 💾 Raw & processed datasets
└── 📁 checkpoints/          # 💾 Saved model weights (.pth)
```

\</details\>

-----

## 🔮 Roadmap & Future Improvements

  - [ ] **Database Integration:** Move from CSV to PostgreSQL for live data.
  - [ ] **Explainable AI (XAI):** Integrate SHAP values to explain *why* a customer is leaving.
  - [ ] **Frontend:** Build a React/Next.js dashboard for better visualization.

-----

## 👨‍💻 Author

\<div align="center"\>

**Md Noushad Jahan Ramim**
<br>
*AI Developer | ML Researcher | Innovator*

[](https://linkedin.com/in/yourprofile)
[](https://github.com/noushad999)
[](mailto:your.email@example.com)

<br>
&lt;i&gt;&quot;Turning Data into Decisions.&quot;&lt;/i&gt;
&lt;/div&gt;

-----

\<div align="center"\>
\<sub\>Built in Bangladesh\</sub\>
\</div\>

```
```
