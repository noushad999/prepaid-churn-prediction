<div align="center">

# 🚀 Telecom Customer Churn Prediction System
### AI-Powered Customer Retention Platform

**Identify at-risk customers before they leave | Save $2.1M+ annually**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[📊 Live Demo](#-live-demo) • [📖 Documentation](#-documentation) • [🚀 Quick Start](#-quick-start) • [👨‍💻 Author](#-author)

</div>

---

## 💼 Business Impact & ROI

Our system doesn't just predict churn; it quantifies business value.

| 💰 Revenue Risk | 📉 Potential Savings | 📈 Campaign ROI | 🎯 High Risk Targets | 🤖 Model Accuracy |
|:---:|:---:|:---:|:---:|:---:|
| **$4.2M** | **$2.1M** | **738%** | **1,743** | **75%** |
| Revenue at Risk | Annual Savings | Retention ROI | Customers Identified | Precision Score |

---

## 🎯 Project Overview

A production-ready machine learning system designed to predict telecom customer churn. It utilizes a **Deep Neural Network** trained on 10,000+ customer records to identify high-risk individuals and automatically suggest retention strategies.

**Key Features:**
* ✅ **High Accuracy:** 74.96% accuracy using ADASYN sampling for class imbalance.
* ✅ **Real-time Inference:** REST API capable of serving predictions in <100ms.
* ✅ **Scalable:** Fully containerized with Docker for easy deployment.
* ✅ **Actionable:** Generates automated retention strategies based on risk segmentation.

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "Data Pipeline"
    A[Customer Data] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[ADASYN Sampling]
    end

    subgraph "AI Core"
    D --> E[Deep Neural Network]
    E --> F[Prediction Engine]
    end

    subgraph "Deployment & UI"
    F --> G[REST API / Flask]
    G --> H[Web Interface]
    G --> I[Executive Reports]
    I --> J[Retention Strategy]
    end

    style E fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333,stroke-width:2px
````

-----

## 🛠️ Tech Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Core AI** |  | Deep Neural Network (3 Hidden Layers) |
| **Data Processing** |  | Feature Engineering & Normalization |
| **Backend API** |  | RESTful Endpoints & Batch Processing |
| **Deployment** |  | Containerization & Orchestration |
| **Algorithms** | `ADASYN`, `Adam`, `ReLU` | Advanced sampling and optimization |

-----

## 🚀 Quick Start

### Prerequisites

  * Python 3.10+
  * Docker (Optional, but recommended)

### Option 1: Docker (Recommended)

```bash
# 1. Clone the repository
git clone [https://github.com/noushad999/telecom-churn-prediction.git](https://github.com/noushad999/telecom-churn-prediction.git)
cd telecom-churn-prediction

# 2. Build and run services
docker-compose up -d

# 3. Access the application
# Web Interface: http://localhost:5001
# API Docs: http://localhost:5001/
```

### Option 2: Local Installation

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data
python download_data.py

# 4. Train the model
python main_simple.py

# 5. Start the API
python api.py
```

-----

## 📡 API Documentation

### 1\. Health Check

`GET /health`
Returns system status and model accuracy.

### 2\. Single Prediction

`POST /predict`

**Request Body:**

```json
{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "tenure": 5,
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "MonthlyCharges": 89.50,
    "TotalCharges": 450.00,
    ... (other features)
}
```

**Response:**

```json
{
    "success": true,
    "prediction": "Will Churn",
    "churn_probability": 0.9825,
    "risk_level": "High",
    "recommendation": "URGENT: Offer 15-20% discount or contract upgrade."
}
```

-----

## 📊 Model Performance Metrics

Our model is optimized for **Recall** (Sensitivity) to ensure we miss as few churning customers as possible.

| Metric | Score | Insight |
|:-------|:------|:--------|
| **Accuracy** | 74.96% | Reliable overall performance |
| **Precision** | 75.08% | Low false positive rate |
| **Recall** | 74.96% | Effectively catches churners |
| **AUC-ROC** | 0.82 | Strong class separation capability |

### Top Churn Indicators

1.  **Contract Type:** Month-to-month users are **45%** more likely to churn.
2.  **Tenure:** Customers with **\<12 months** are high risk.
3.  **Payment:** Electronic check users show higher instability.

-----

## 📂 Project Structure

\<details\>
\<summary\>Click to view directory structure\</summary\>

```text
telecom-churn-prediction/
├── 📄 api.py                # Flask REST API
├── 🧠 main_simple.py        # Model Training Pipeline
├── 🐳 Dockerfile            # Docker Configuration
├── 📜 docker-compose.yml    # Container Orchestration
├── 📋 requirements.txt      # Dependencies
├── 📂 src/
│   ├── models/              # PyTorch Model Arch
│   ├── data_processing/     # Feature Engineering
│   └── training/            # Training Loops
├── 📂 data/                 # Dataset Storage
└── 📂 checkpoints/          # Saved Model Weights (.pth)
```

\</details\>

-----

## 🔮 Future Roadmap

  - [ ] **Phase 1:** Integrate PostgreSQL for persistent customer data.
  - [ ] **Phase 2:** Implement Graph Neural Networks (GNN) to analyze social influence on churn.
  - [ ] **Phase 3:** Deploy to AWS Lambda using Serverless architecture.

-----

## 👨‍💻 Author

\<div align="center"\>

**Md Noushad Jahan Ramim**
<br>
*AI Developer | Computer Science Student | ML Researcher*

[](https://github.com/noushad999)
[](https://linkedin.com/in/yourprofile)
[](mailto:your.email@example.com)

\</div\>

-----

\<div align="center"\>

**Did you find this project useful?** ⭐ Give it a star to support my work\!

\</div\>

```

### কেন এই পরিবর্তনগুলো করলাম:

1.  **Visual Hierarchy:** সব কিছু এক লাইনে না লিখে **Table** এবং **Icons** ব্যবহার করেছি। এতে প্রজেক্টটা অনেক বেশি Professional মনে হবে।
2.  **Mermaid Diagram:** আর্কিটেকচার সেকশনে `Mermaid` কোড ব্যবহার করেছি। GitHub এটা অটোমেটিক রেন্ডার করবে এবং দেখতে খুব সুন্দর ডায়াগ্রাম মনে হবে।
3.  **Collapsible Section:** ফাইল স্ট্রাকচার অনেক বড় ছিল, তাই সেটাকে `<details>` ট্যাগ দিয়ে লুকিয়ে রেখেছি। যারা দেখতে চায় তারা ক্লিক করে দেখবে, এতে পেজটা ক্লিন থাকে।
4.  **Quick Start Options:** যারা `Docker` জানে তাদের জন্য আলাদা এবং যারা শুধু `Python` দিয়ে রান করতে চায় তাদের জন্য আলাদা ইন্সট্রাকশন দিয়েছি।
5.  **Personal Branding:** নিচে তোমার নাম এবং টাইটেল (AI Developer | ML Researcher) যোগ করে দিয়েছি।

তুমি শুধু কপি করে `README.md` ফাইলে পেস্ট করে দাও\!
```
