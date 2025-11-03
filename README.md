# Create the updated README
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
![Main Dashboard](screenshots/Screenshot%202025-11-03%20at%208.35.46 PM.png)
*Interactive dashboard showing key performance metrics and treatment comparisons*

### 📈 Clinical Analysis
![HbA1c Distribution](screenshots/Screenshot%202025-11-03%20at%208.35.58 PM.png)
*HbA1c distribution by treatment group with clinical targets*

### 🤖 Machine Learning Insights
![Feature Importance](screenshots/Screenshot%202025-11-03%20at%208.36.13 PM.png)
*Machine learning feature importance showing key predictors of treatment response*

### 📊 Advanced Statistics
![Power Analysis](screenshots/Screenshot%202025-11-03%20at%208.36.35 PM.png)
*Statistical power analysis with sample size calculations*

### 🌐 3D Visualizations
![3D Network](screenshots/Screenshot%202025-11-03%20at%208.37.54 PM.png)
*3D network graph showing biomarker relationships and treatment effects*

![3D Scatter Plot](screenshots/Screenshot%202025-11-03%20at%208.38.04 PM.png)
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