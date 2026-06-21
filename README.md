# English Premier League Match Outcome Prediction

## Project Overview
This study presents a time-aware machine learning pipeline designed to predict home team victories in the English Premier League[cite: 6]. Due to the highly stochastic nature of football, the problem was framed as a binary classification task (Home Win vs. Not Win) to reduce noise and improve model reliability[cite: 6].

## Tech Stack & Methodology
* **Language:** R[cite: 7]
* **Libraries:** `tidyverse`, `caret`, `e1071` (SVM), `randomForest`, `xgboost`, `nnet`, `pROC`, `SHAPforxgboost`[cite: 7]
* **Cross-Validation:** TimeSeriesSplit (Expanding Window) was utilized instead of standard k-fold splitting to strictly prevent temporal data leakage and simulate real-world deployment conditions[cite: 6, 7].
* **Feature Engineering:** Venue-aware 5-match rolling window averages (with a shift operation to prevent in-game leakage) resulting in 26 pre-match features[cite: 6, 7].

## Dataset
* **Source:** Football-Data.co.uk (2000/01 to 2024/25 EPL seasons)[cite: 6, 7]
* **Sample Size:** 9,125 viable matches after the rolling window warmup period[cite: 6].
* **Key Features:** Shot Difference, Win Rate Difference, and Shots on Target Difference between home and away teams[cite: 6].

## Model Performance & Calibration
Six algorithms were trained and evaluated on a chronologically split 20% test set[cite: 6, 7]. 

| Model | Calibration | ROC-AUC | F1-Score |
| :--- | :--- | :---: | :---: |
| **SVM (Best Model)** | **Platt Scaling** | **0.6864** | **0.5424** |
| Naive Bayes | Raw | 0.6849 | 0.5226 |
| XGBoost | Platt Scaling | 0.6836 | 0.5300 |
| Random Forest | Raw | 0.6703 | 0.5063 |

*Note: Threshold analysis indicated that dropping the decision threshold from 0.50 to 0.35 maximized the F1-Score (0.6413) for the SVM model by capturing more actual home wins[cite: 6].*

## Key Insights & Ethical Considerations
* **The Variance-Bias Tradeoff:** A simple probabilistic classifier like Naive Bayes outperformed heavy ensemble algorithms (Random Forest, XGBoost)[cite: 6]. This demonstrates that in highly noisy and high-variance environments like football, simpler models with lower variance often generalize better[cite: 6].
* **Home Advantage Bias:** A fairness evaluation utilizing the False Positive to False Negative (FP/FN) ratio (0.613) revealed that the model does not suffer from a systematic home bias; instead, it exhibits a conservative approach, placing significant weight on the away team's resilience[cite: 6].
* **Interpretability:** SHAP analysis confirmed the model learned underlying tactical dominance (e.g., shot and win-rate differences) rather than merely memorizing scoreboard results[cite: 6].
