<div align="center">

# 🚗 CarValue-AI
### Intelligent Used Vehicle Valuation & Depreciation Forecasting Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://carvalue-ai-rbhwm7wktdng3mw9syxuxf.streamlit.app/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-brightgreen?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web platform designed to forecast secondary automobile market prices with high precision, featuring dynamic multi-variable inference and real-time interactive depreciation curves.

[**🌐 Live Demo**](https://carvalue-ai-rbhwm7wktdng3mw9syxuxf.streamlit.app/) • [**📂 GitHub Repository**](https://github.com/sunnythakursunny650-cell/CarValue-AI)

</div>

---

## 📌 Executive Summary

Predicting used car values presents non-linear challenges due to brand positioning, localized market demand, vehicle aging, and performance specs. Traditional linear estimators often fail to capture complex categorical interactions (e.g., brand-tier premiums combined with high odometer readings).

**CarValue-AI** solves this by implementing an ensemble-based **Random Forest Regressor** encased within an automated, leakage-free Scikit-Learn data transformation pipeline, backed by dynamic category filtering, fintech tools, and an intuitive dashboard.

---

## 🏗️ Technical Architecture & Key Highlights

- **Leak-Free Transformation Pipeline:** Implements Scikit-learn's `ColumnTransformer` to package `StandardScaler` for continuous numerical features (`Mileage`, `EngineV`, `Year`) and `OneHotEncoder` for high-cardinality categorical variables (`Brand`, `Body`, `Model`, `Engine Type`, `Registration`).
- **Dynamic Dependent Dropdowns:** Resolves brand-model mismatch bugs by maintaining a metadata index (`artifacts/car_meta.json`) to dynamically populate legitimate models matching the selected manufacturer.
- **Log-Target Regularization:** Mitigates target right-skewness using a logarithmic transformation ($\log(1 + y)$), ensuring stable variance and superior loss convergence during training.
- **Explainable Depreciation Forecasting:** Employs dynamic Plotly visualization to plot depreciation velocity curves across vehicle age horizons for instantaneous decision support.
- **Fintech & Certificate Generation:** Features a dynamic Loan & EMI Customizer with adjustable interest/down payment and a 1-click downloadable Valuation Certificate (CSV).
- **In-App Technical Specifications:** Expandable architectural panel detailing pipeline parameters, real-time inference latency, and feature attributions.

---

## ℹ️ Model Specifications & Performance Summary

- **Core Model:** Random Forest Regressor Pipeline (Scikit-Learn)
- **Target Scale:** Log-transformed (`np.log1p` / `np.expm1`) to mitigate price skewness
- **Validation Score ($R^2$):** **0.7513** (Cross-validated test set)
- **Serving Latency:** **< 14 ms** real-time single-row inference runtime
- **Feature Attribution:** Weighted by Engine Displacement, Model Longevity decay (~6.2% annual), and Brand Equity Multipliers

---

## 📊 Model Evaluation & Benchmarking

Models were trained and evaluated on 85/15 train-test splits using 5-fold cross-validation. The **Random Forest Regressor** delivered the highest predictive accuracy and lowest residual dispersion.

| Model Architecture | R² Score | MAE (INR) | Pipeline Latency | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **0.7513** | **₹3,60,502** | **< 12ms** | **Production Deployed** |
| Ridge Regression ($\alpha = 1.0$) | 0.6320 | ₹4,45,120 | < 2ms | Baseline Benchmark |
| Linear Regression (OLS) | 0.6291 | ₹4,48,900 | < 2ms | Baseline Benchmark |
| Lasso Regression ($\alpha = 0.01$) | 0.6184 | ₹4,58,300 | < 2ms | Baseline Benchmark |

---

## 📁 Repository Structure

```text
CarValue-AI/
├── artifacts/
│   ├── car_meta.json              # Extracted categorical mappings & boundaries
│   └── car_price_pipeline.pkl     # Serialized end-to-end inference pipeline
├── data/
│   └── car_data.csv               # Vehicle attributes dataset
├── notebooks/
│   └── 01_EDA_and_Modeling.ipynb  # EDA, distributions & benchmark models
├── .gitignore                     # Git tracking exclusions
├── app.py                         # Production Streamlit web application
├── train_model.py                 # Pipeline training & artifact export
├── requirements.txt               # Application dependencies
└── README.md                      # Project documentation

---

## 🚀 Installation & Local Deployment

### 1. Clone the Repository
```bash
git clone [https://github.com/sunnythakursunny650-cell/CarValue-AI.git](https://github.com/sunnythakursunny650-cell/CarValue-AI.git)
cd CarValue-AI
```

### 2. Set Up Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute Pipeline Training
```bash
python train_model.py
```

### 5. Launch Dashboard
```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Sunny Thakur**  
- **Role:** Machine Learning Engineer & Educator  
- **GitHub:** [@sunnythakursunny650-cell](https://github.com/sunnythakursunny650-cell)  
- **LinkedIn:** [Sunny Thakur](https://www.linkedin.com/in/sunny-thakur-4a56103b9/)

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).