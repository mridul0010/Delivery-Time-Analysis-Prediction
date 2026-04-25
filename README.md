<div align="center">

# 🚚 Delivery-Time-Analysis-Prediction

### Machine Learning & Analytics Approach

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-blue)](https://xgboost.readthedocs.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Predict delivery times with an end-to-end ML pipeline — from data preprocessing and exploratory analysis to model comparison, evaluation, and interactive deployment.**

[Live Demo](https://delivery-time-analysis-prediction.streamlit.app/) . [Report Bug](https://github.com/issues) · [Request Feature](https://github.com/issues)

</div>

---

## 📑 Table of Contents

- [About the Project](#-about-the-project)
- [Problem Statement](#-problem-statement)
- [Project Structure](#-project-structure)
- [Model Development Workflow](#-model-development-workflow)
- [Analytics Dashboard](#-analytics-dashboard)
- [Model Evaluation Metrics](#-model-evaluation-metrics)
- [Screenshots](#-screenshots)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Key Takeaways](#-key-takeaways)
- [Future Improvements](#-future-improvements)

---

## 📌 About the Project

This repository implements a **delivery time prediction system** using a comprehensive machine learning approach. Multiple models were trained and evaluated to predict accurate delivery times from logistics data. **XGBoost was selected as the final production model** based on performance, robustness, and suitability for the prediction task.

A **Streamlit web application** is provided for interactive use, enabling users to:
- **Input delivery parameters** and receive real-time delivery time predictions
- **Explore analytics** through an interactive dashboard with visualizations
- **View model performance** metrics and evaluation results
- **Understand delivery patterns** through exploratory data analysis

> **Philosophy:** This project emphasizes _data-driven decision making, model comparison, comprehensive evaluation, and deployment-ready solutions_ — ensuring predictions are both accurate and interpretable.

---

## 🧠 Problem Statement

Accurate delivery time prediction is critical for logistics optimization, customer satisfaction, and operational efficiency. The objective is to **predict delivery times based on order and delivery characteristics**, enabling:

| Challenge | Description |
|---|---|
| **Feature Complexity** | Multiple interdependent features affecting delivery time |
| **Non-linear Relationships** | Complex interactions between delivery factors |
| **Real-world Deployment** | Model must provide quick, reliable predictions in production |

---

## 📂 Project Structure

```
Delivery-Time-Prediction/
│
├── notebooks/                          # Jupyter notebooks for analysis
│   ├── 01_Data_Preprocessing.ipynb      # Data cleaning and preparation
│   ├── 02_EDA.ipynb                     # Exploratory Data Analysis
│   ├── 03_Model_Selection.ipynb         # Model comparison & evaluation
│   └── 04_Model_Training.ipynb          # Final model training
│
├── app/                                 # Streamlit application
│   ├── 01_analytics.py                  # Analytics & insights dashboard
│   ├── 02_prediction.py                 # Prediction interface
│   └── 03_main_app.py                   # Main app entry point
│
├── data/                                # Datasets
│   ├── Delivery Dataset.csv             # Original dataset
│   └── Cleaned Delivery Dataset.csv     # Preprocessed dataset
│
├── models/                              # Saved models & pipelines
│
├── requirements.txt                     # Project dependencies
├── exporting_to_sql.ipynb              # Database export notebook
├── hello.py                             # Utility script
└── README.md                            # Project documentation
```

---

## 🧩 Model Development Workflow

### 1. Data Preprocessing

- **Data Loading & Exploration:** Loaded and analyzed the delivery dataset
- **Missing Value Handling:** Implemented strategies for incomplete data
- **Feature Encoding:** Converted categorical variables to numerical format
- **Feature Scaling:** Normalized numerical features for model compatibility
- **Data Splitting:** Applied stratified train-test split for unbiased evaluation
- **Pipeline Creation:** Built preprocessing + model pipeline for consistent inference

**Notebook:** [`01_Data_Preprocessing.ipynb`](notebooks/01_Data_Preprocessing.ipynb)

### 2. Exploratory Data Analysis (EDA)

- Statistical summary of delivery features
- Distribution analysis of delivery times
- Correlation analysis between features and target variable
- Geographical and temporal delivery patterns
- Visualization of key business insights

**Notebook:** [`02_EDA.ipynb`](notebooks/02_EDA.ipynb)

### 3. Model Training & Selection

All experiments and comparisons are documented in [`03_Model_Selection.ipynb`](notebooks/03_Model_Selection.ipynb).

| Model | Type | Performance | Notes |
|---|---|---|---|
| **XGBoost** ✅ | Gradient Boosting | Highest R² & Lowest MAE | Final production model |
| Random Forest | Ensemble | Strong baseline | Good interpretability |
| Linear Regression | Regression | Baseline | Quick reference |

**Model Selection Criteria:**
- Prediction accuracy (R² Score, MAE, RMSE)
- Generalization to unseen data
- Robustness and stability
- Interpretability and explainability

### 4. Final Model Choice — XGBoost

XGBoost was selected because it:

- ✅ Excels on regression tasks with mixed feature types
- ✅ Captures non-linear relationships and feature interactions efficiently
- ✅ Provides strong feature importance scores for business insights
- ✅ Offers superior generalization compared to simpler models
- ✅ Handles the complex nature of delivery data effectively

**Notebook:** [`04_Model_Training.ipynb`](notebooks/04_Model_Training.ipynb)

---

## 📊 Analytics Dashboard

The Streamlit application includes a **Data Analytics** page that helps explore delivery patterns before making predictions.

### Key Analytics Features

- **Delivery Statistics:** Summary of average delivery times, distance, and costs
- **Distribution Analysis:** Histograms and density plots for key variables
- **Correlation Heatmap:** Feature relationships and dependencies
- **Temporal Patterns:** Delivery trends over time periods
- **Geographical Analysis:** Regional delivery performance metrics
- **Feature Importance:** ML model's perspective on key drivers

### Interactive Visualizations

- Filter data by various dimensions
- Drill-down into specific delivery patterns
- Export insights for business reporting

---

## 📊 Model Evaluation Metrics

The final **XGBoost model** demonstrates strong performance across multiple regression metrics.

### Performance Summary (Test Set)

| Metric | Score | Description |
|---|---|---|
| **R² Score** | 0.83 | Proportion of variance explained |
| **Adjusted R² Score** | 0.82 | Adjusted for number of features (penalizes unnecessary variables) |
| **Mean Absolute Error (MAE)** | 3.02 minutes | Average prediction error |
| **Root Mean Squared Error (RMSE)** | 3.79 minutes | Penalizes larger errors |

### Model Performance Dashboard (In-App)

The **Model Performance** page visualizes evaluation outputs:

- Actual vs Predicted delivery times scatter plot
- Residual plots to assess prediction errors
- Model performance metrics and statistics
- Feature importance rankings

---
## 📸 Screenshots

<details>
<summary>Click to expand screenshots</summary>

<img width="1909" height="1077" alt="image" src="https://github.com/user-attachments/assets/c8a0493d-f54a-480d-93bb-f1be9110ded7" />
<img width="1614" height="876" alt="image" src="https://github.com/user-attachments/assets/48954c0a-9ed6-454f-810c-83d8508930eb" />
<img width="1595" height="1026" alt="image" src="https://github.com/user-attachments/assets/4f22bc70-b2fc-41a3-b6ad-64cbe6085ced" />
<img width="1619" height="1038" alt="image" src="https://github.com/user-attachments/assets/752f7ed2-7212-43cd-96de-69acc7f5350f" />
<img width="1919" height="948" alt="image" src="https://github.com/user-attachments/assets/36ce6920-450a-451f-ae86-3d722fbd0d88" />
<img width="1833" height="1025" alt="image" src="https://github.com/user-attachments/assets/6ebf66d5-e638-4075-b3b4-20b3fbbca40a" />
<img width="1846" height="949" alt="image" src="https://github.com/user-attachments/assets/73b22d53-fcfa-4e9a-a036-cd6b8b1a0242" />








</details>

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.10+ |
| **Machine Learning** | XGBoost, Random Forest, Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Deployment** | Streamlit |
| **Development** | Jupyter Notebook |


---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Delivery-Time-Prediction.git
   cd Delivery-Time-Prediction
   ```

2. **Create a virtual environment** (optional but recommended)

   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate      # On macOS/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app/03_main_app.py
   ```

5. **Access the application**

   Open your browser and navigate to `http://localhost:8501`

---

## 💡 Usage

### Using the Web Application

1. **Navigate to Prediction Tab**
   - Input delivery parameters (distance, cost, product weight, etc.)
   - Click **"🚚 Predict Delivery Time"**

2. **View Prediction Results**
   - Get estimated delivery time
   - See confidence metrics
   - Receive actionable insights

3. **Explore Analytics Dashboard**
   - View delivery patterns and trends
   - Analyze feature correlations
   - Understand key delivery drivers

4. **Check Model Performance**
   - Review model accuracy metrics
   - Visualize actual vs predicted values
   - Understand prediction reliability

### Running Notebooks

For detailed analysis and model training:

```bash
jupyter notebook
```

Then open the desired notebook from the `notebooks/` folder.

---

## 🎯 Key Takeaways

- Demonstrates **end-to-end ML pipeline** from raw data to production deployment
- Shows **comparative model evaluation** with data-driven model selection
- Emphasizes **real-world considerations** including data preprocessing, validation, and evaluation
- Provides **interactive analytics** for stakeholder insights
- Combines **ML accuracy** with **business interpretability**
- Showcases **best practices** in ML project structure and documentation

---

## 🔮 Future Improvements

- [ ] Add real-time model monitoring and performance tracking
- [ ] Implement automated retraining pipeline for model updates
- [ ] Integrate external data sources (weather, traffic, etc.)
- [ ] Deploy to cloud platforms (AWS, GCP, Azure)
- [ ] Add confidence intervals for predictions
- [ ] Implement A/B testing framework for model versions
- [ ] Create API endpoint for production integration
- [ ] Add batch prediction capabilities

---

**Mridul Lata**

📍 Jaipur, India · 💼 Aspiring Data Scientist / ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mridullata)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/mridul0010)

</td>
</tr>
</table>

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

📌 _This project focuses on real-world ML decision-making and production-ready solutions._

⭐ **If you found this helpful, please give the repository a star and share your feedback!**

</div>
