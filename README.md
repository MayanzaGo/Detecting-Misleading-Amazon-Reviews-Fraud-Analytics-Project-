# Detecting Misleading Amazon Reviews (Fraud Analytics Project)

## Executive Summary
This project addresses the **audit risk** of misleading Amazon product reviews, which can distort consumer decision-making and compromise marketplace trust.  
By applying **NLP techniques, sentiment analysis, and machine learning**, the framework detects **fraudulent or inconsistent reviews** that may indicate manipulation.

- **Audit Risk:** Misleading reviews undermine consumer trust and can signal weak fraud controls.  
- **Audit Approach:** Test control by checking **consistency between review title, text, and rating sentiment** using VADER and NLP preprocessing.  
- **Key Findings:** Random Forest achieved the **highest fraud detection sensitivity**, while Naïve Bayes had the lowest false positives but poor fraud detection.  
- **Recommendation:** Adopt **ensemble-based models (Random Forest/Gradient Boosting)** with periodic retraining to support continuous fraud monitoring.  

---

## Objectives
- Build an **audit-oriented NLP pipeline** to identify untrustworthy product reviews.  
- Translate **textual and rating inconsistencies** into quantifiable fraud risk indicators.  
- Provide **auditors and risk managers** with a framework for anomaly detection in online platforms.  

---

## Methodology (Audit Approach)
1. **Data Selection**  
   - Extract reviews from the top 5 Amazon departments.  
   - Include title, text, rating, and department metadata.  

2. **Control Testing**  
   - Use **VADER Sentiment Analysis** to assign sentiment labels (Positive, Neutral, Negative).  
   - Compare **sentiment of title vs. text** (consistency check).  
   - Compare **sentiment vs. rating** (fraud control test).  

3. **Data Preprocessing**  
   - Lowercasing, tokenization, stopword removal.  
   - Feature engineering with **Binary, Frequency, and TF-IDF representations**.  

4. **Model Evaluation**  
   - Classifiers tested:  
     - Naïve Bayes  
     - Logistic Regression  
     - Random Forest  
     - Support Vector Machine (SVM)  
     - Gradient Boosting  
   - Metrics used (audit framing):  
     - **Control Exception Rate (1 – Accuracy)**  
     - **Fraud Detection Sensitivity (Recall)**  
     - **False Positive Risk (1 – Precision)**  
     - **F1 Score**  

---

## Key Audit Findings

| Model               | Control Exception Rate (1-Accuracy) | Fraud Detection Sensitivity (Recall) | False Positive Risk (1-Precision) | F1 Score |
|----------------------|--------------------------------------|--------------------------------------|-----------------------------------|----------|
| Naïve Bayes         | 0.193                                | 0.002                                | 0.000                             | 0.004    |
| Logistic Regression | 0.196                                | 0.017                                | 0.634                             | 0.033    |
| Random Forest       | 0.197                                | 0.078                                | 0.546                             | 0.134    |
| SVM                 | 0.202                                | 0.034                                | 0.697                             | 0.061    |
| Gradient Boosting   | 0.193                                | 0.004                                | 0.091                             | 0.007    |

**Interpretation:**  
- **Random Forest**: Highest fraud detection sensitivity (best fraud risk coverage).  
- **Naïve Bayes**: Very low false positives but fails to detect fraud.  
- **SVM & Logistic Regression**: Moderate fraud detection but at the cost of high false positives.  
- **Gradient Boosting**: Balanced error rate but extremely low sensitivity.  

---

## Recommendations
- **Random Forest** should be adopted for fraud analytics in audit engagements.  
- **Periodic recalibration** is recommended to maintain accuracy as new review data emerges.  
- **Hybrid ensemble approaches** may further reduce false positives without sacrificing fraud detection.  

---

## Tech Stack
- **Languages/Frameworks:** Python, Scikit-learn  
- **NLP Tools:** VADER, NLTK, TF-IDF  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn  
- **Methodology:** Audit Analytics, Fraud Detection, Machine Learning, NLP  

---
## Data Access
The full dataset is too large to host on GitHub.  
- Download full dataset from https://www.kaggle.com/datasets/yeshmesh/inconsistent-and-consistent-amazon-reviews


