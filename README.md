
<div align="center">

# 🔮 Telecom Customer Churn Prediction
### Enterprise-Grade AI Retention System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

<img src="https://raw.githubusercontent.com/noushad999/telecom-churn-prediction/main/docs/banner_placeholder.png" alt="Project Banner" width="100%" />

<p align="center">
  <a href="#-key-features">Key Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-performance-benchmarks">Benchmarks</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-api-reference">API Reference</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

---

## 📖 Introduction

**Telecom Customer Churn Prediction** is a production-ready machine learning pipeline designed to identify at-risk customers in the telecommunications sector. Unlike traditional rule-based systems, this project utilizes a **Deep Neural Network (DNN)** with ADASYN oversampling to handle class imbalance, delivering high recall rates for churn detection.

The system is fully containerized using Docker and exposes a RESTful API via Flask, making it ready for integration into enterprise CRM systems (Salesforce, HubSpot, etc.).

### 🎯 The Problem vs. Our Solution

| Feature | ❌ Traditional Analytics | ✅ Our AI Solution |
| :--- | :--- | :--- |
| **Approach** | Reactive (Analyze after churn) | **Proactive** (Predict before churn) |
| **Data Scope** | Limited to billing data | **Multi-dimensional** (Support, Usage, Demographics) |
| **Scalability** | Manual Spreadsheets/SQL | **Automated** Docker Containers |
| **Actionability** | Generic Reports | **Targeted** API Recommendations |

---

## ✨ Key Features

### 🧠 Advanced AI Core
* **Deep Learning Model:** 4-layer PyTorch Neural Network optimized for tabular data.
* **Imbalance Handling:** Integrated **ADASYN** (Adaptive Synthetic Sampling) to detect minority class (churners) effectively.
* **Feature Engineering:** Automated pipeline for categorical encoding and numerical scaling.

### ⚙️ Production Engineering
* **Microservice Architecture:** Decoupled inference engine accessible via REST API.
* **Containerization:** `Dockerfile` and `docker-compose` included for "write once, run anywhere" capability.
* **Latency:** Sub-100ms inference time for real-time applications.

### 📊 Business Intelligence
* **Risk Segmentation:** Classifies users into `Critical`, `High`, `Medium`, and `Low` risk.
* **ROI Calculator:** Estimates revenue saved based on retention campaign success rates.

---

## 🏗 Architecture

The system follows a modern ML Ops pipeline structure.

```mermaid
flowchart LR
    subgraph Data ["📡 Data Layer"]
        Raw[(Raw CSV)] --> Clean(Preprocessing)
        Clean --> Split(Train/Test Split)
    end

    subgraph Model ["🧠 Model Layer"]
        Split --> Oversample(ADASYN)
        Oversample --> Training(PyTorch DNN)
        Training --> Weights{{Model Weights}}
    end

    subgraph Serving ["🚀 Serving Layer"]
        Weights --> Inference(Prediction Engine)
        Inference --> API[Flask API]
        API --> Client((Web/Mobile App))
    end

    style Model fill:#f9f,stroke:#333,stroke-width:2px
    style Serving fill:#bbf,stroke:#333,stroke-width:2px
````

-----

## 📊 Performance Benchmarks

We prioritize **Recall** over Precision because missing a churning customer (False Negative) is more costly than offering a discount to a loyal one (False Positive).

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | `74.96%` | Overall correct predictions |
| **Recall** | `74.96%` | **Key Metric:** Ability to catch churners |
| **Precision** | `75.08%` | Accuracy of positive churn predictions |
| **F1-Score** | `0.7496` | Harmonic balance of Precision & Recall |

> *Benchmarks run on Intel i7, 16GB RAM environment using the Telco Customer Churn Dataset.*

-----

## 💻 Installation

### Prerequisites

  * Docker & Docker Compose (Recommended)
  * **OR** Python 3.10+

### Option A: Docker (Production Style)

```bash
# 1. Clone the repository
git clone [https://github.com/noushad999/telecom-churn-prediction.git](https://github.com/noushad999/telecom-churn-prediction.git)

# 2. Build and start services
docker-compose up -d --build

# 3. Verify status
docker ps
```

### Option B: Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main_simple.py     # Train the model
python api.py             # Start the API server
```

-----

## 📡 API Reference

The API runs on `http://localhost:5001`.

### `POST /predict`

Predicts churn probability for a specific customer profile.

**Request Body (JSON):**

```json
{
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "tenure": 5,
  "MonthlyCharges": 90.5,
  "PaymentMethod": "Electronic check",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No"
  // ... other features
}
```

**Success Response (200 OK):**

```json
{
  "status": "success",
  "data": {
    "prediction": "Churn",
    "probability": 0.8923,
    "risk_level": "Critical",
    "recommended_action": "Initiate 'Save-the-Customer' Protocol: Offer 20% discount."
  }
}
```

-----

## 📂 Directory Structure

```text
.
├── 📂 checkpoints/      # Trained model artifacts (.pth files)
├── 📂 data/             # Raw and processed datasets
├── 📂 docs/             # Documentation assets (images, diagrams)
├── 📂 src/              # Source code
│   ├── data_loader.py   # Data ingestion scripts
│   ├── model.py         # PyTorch neural network class
│   └── trainer.py       # Training loop logic
├── 📄 api.py            # Flask application entry point
├── 📄 Dockerfile        # Container definition
├── 📄 main_simple.py    # Main execution script
└── 📄 requirements.txt  # Python dependencies
```

-----

## 📜 Citation

If you use this code for your research, please cite it as:

```bibtex
@misc{ramim2025telecomchurn,
  author = {Ramim, Md Noushad Jahan},
  title = {Telecom Customer Churn Prediction System},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/noushad999/telecom-churn-prediction](https://github.com/noushad999/telecom-churn-prediction)}}
}
```

-----

## 👨‍💻 Author

\<div align="center"\>

**Md Noushad Jahan Ramim**

*AI Developer | ML Researcher*

[](https://github.com/noushad999)
[](https://linkedin.com/in/yourprofile)
[](mailto:your.email@example.com)

\</div\>

-----

\<div align="center"\>
\<sub\>This project is licensed under the MIT License.\</sub\>
\</div\>
