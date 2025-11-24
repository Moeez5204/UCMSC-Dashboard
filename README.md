# 🎯 UC-MSC Therapy for Type 2 Diabetes - Interactive Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Dash](https://img.shields.io/badge/Dash-Plotly-orange)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📊 Project Overview

An interactive data science dashboard analyzing **Umbilical Cord Mesenchymal Stem Cell (UC-MSC)** therapy effectiveness for **Type 2 Diabetes** patients. This project demonstrates end-to-end data science skills from data analysis to machine learning and interactive visualization.

**Live Demo**: Run `python app.py` and open http://127.0.0.1:8050

## 🎯 Key Findings

- **📈 40.0% responder rate** with UC-MSC vs **11.2%** with placebo (+28.8% improvement)
- **🩺 17.5% good control** (<8% HbA1c) vs 12.5% with placebo
- **📊 Statistical significance** at 48 weeks (p=0.0268)
- **🌍 Population impact**: 4+ million additional responders globally at 140M patients

## 📚 Research Basis

This project is based on clinical research from:

**Cai, J., Wu, Z., Xu, X. et al. (2022). *Umbilical cord mesenchymal stromal cell with autologous bone marrow cell transplantation in established type 1 diabetes: a pilot randomized controlled open-label clinical study*. Stem Cell Research & Therapy, 13, 199.**

**DOI:** [10.1186/s13287-022-02875-3](https://pmc.ncbi.nlm.nih.gov/articles/PMC9066971/)


## 📸 Dashboard Visualizations

### 🏠 Main Dashboard & Metrics
<img width="929" height="1064" alt="Screenshot 2025-11-03 at 8 35 46 PM" src="https://github.com/user-attachments/assets/3f9a235e-9508-4b8c-b6f5-cf4684b2e6f8" />
*Interactive dashboard showing key performance metrics and treatment comparisons*

### 📈 Clinical Analysis
<img width="929" height="444" alt="Screenshot 2025-11-03 at 8 35 58 PM" src="https://github.com/user-attachments/assets/b6d79cfd-34ea-4a77-bbcd-b1ed47c4e18b" />
*HbA1c distribution by treatment group with clinical targets*

### 🤖 Machine Learning Insights
<img width="929" height="693" alt="Screenshot 2025-11-03 at 8 36 13 PM" src="https://github.com/user-attachments/assets/813268c5-4985-45d0-bc2d-21a4e0f67e27" />
*Machine learning feature importance showing key predictors of treatment response*

### 📊 Advanced Statistics
 <img width="929" height="650" alt="Screenshot 2025-11-03 at 8 36 32 PM" src="https://github.com/user-attachments/assets/b80b0176-9b19-4c90-9fe4-d7baa367a1eb" />
*Statistical power analysis with sample size calculations*

### 🌐 3D Visualizations
<img width="929" height="964" alt="Screenshot 2025-11-03 at 8 37 54 PM" src="https://github.com/user-attachments/assets/0fb4340d-e8a5-441c-8c78-f3d51035a6fc" />
*3D network graph showing biomarker relationships and treatment effects*

<img width="929" height="939" alt="Screenshot 2025-11-03 at 8 38 04 PM" src="https://github.com/user-attachments/assets/7a10e223-5dfa-4fd9-9391-cfd2d88ae37a" />
*Interactive 3D scatter plot of HbA1c vs BMI vs Age by treatment group*

## 🛠️ Technical Architecture

### Tech Stack
- **Frontend**: Dash, Plotly, HTML/CSS
- **Backend**: Python, Pandas, NumPy
- **Machine Learning**: Scikit-learn, Feature Importance
- **Statistics**: SciPy, Hypothesis Testing, Power Analysis
- **Visualization**: 3D Plotly, Network Graphs, Interactive Charts


## 🎓 Mathematical & Statistical Methods

This project demonstrates practical application of:

- **Statistical Hypothesis Testing**: T-tests, Chi-square tests, p-values
- **Effect Size Analysis**: Cohen's d, risk differences
- **Power Analysis**: Sample size calculations and statistical power
- **Machine Learning**: Feature importance, model performance metrics
- **Probability Distributions**: Clinical outcome distributions
- **3D Spatial Analysis**: Network graphs and multidimensional visualization

## 🚀 Getting Started

### Installation
```bash
# Clone repository
git clone https://github.com/Moeez5204/UCMSC-Dashboard.git
cd UCMSC-Dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python app.py
