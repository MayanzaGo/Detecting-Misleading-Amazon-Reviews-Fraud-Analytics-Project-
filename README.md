Executive Summary
Online marketplaces face a fraud risk from misleading reviews that misstate product quality, undermining trust in reported customer experience.
This project applies an Audit Analytics approach to test review reliability by analyzing sentiment–rating consistency and classifying reviews as trustworthy or untrustworthy.
•	Risk Identified: Misrepresentation of product sentiment in Amazon reviews.
•	Audit Approach: Exception testing (sentiment vs. rating), classification modeling, and fraud sensitivity evaluation.
•	Key Findings: Random Forest achieved the highest fraud sensitivity (recall), but at the cost of higher false positive risk.
•	Recommendation: Deploy ensemble-based fraud monitoring with periodic recalibration as a preventive control to maintain trust.
________________________________________
Project Objectives
•	Identify inconsistent or misleading reviews.
•	Apply sentiment concordance tests to flag control exceptions.
•	Benchmark multiple classifiers for fraud detection assurance.
•	Deliver an audit-oriented framework for fraud/anomaly detection.
________________________________________
Methodology (Audit Approach)
1.	Risk Assessment
o	Fraud risk: misleading reviews distort audit evidence of product quality.
o	Control under test: consistency between review sentiment and rating.
2.	Audit Procedures
o	Data preprocessing & NLP pipeline.
o	Sentiment analysis using VADER.
o	TF-IDF feature extraction.
o	Classification of reviews → Trustworthy vs. Untrustworthy.
3.	Control Evaluation
o	Metrics reframed in audit terms:
	Control Exception Rate (1 - Accuracy)
	Fraud Detection Sensitivity (Recall)
	False Positive Risk (1 - Precision)
	F1 Score (balanced audit measure)
4.	Reporting
o	Tabular findings.
o	Fraud risk dashboard (heatmap).
o	Audit opinion.

Key Audit Findings
Model	Control Exception Rate (1-Accuracy)	Fraud Detection Sensitivity (Recall)	False Positive Risk (1-Precision)	F1 Score
Naïve Bayes	0.193	0.002	0	0.004
Logistic Regression	0.196	0.017	0.634	0.033
Random Forest	0.197	0.078	0.546	0.134
SVM	0.202	0.034	0.697	0.061
Gradient Boosting	0.193	0.004	0.091	0.007


Audit Analysis
Naïve Bayes → Lowest false positive risk, but essentially blind to fraud (recall = 0.2%).
Logistic Regression → Detects some fraud, but unacceptably high false positive risk (63%).
Random Forest → Strongest fraud sensitivity (7.8%), but moderate exception rate and false positive risk (55%).
SVM → Balanced, but higher exception rate (20%).
Gradient Boosting → Very low fraud detection (0.4%) despite stable exception rate.

Audit Insight:
Random Forest provides the highest fraud detection assurance, but model outputs must be complemented with manual review or layered controls to mitigate false positives.

 Fraud Risk Dashboard
(Insert fraud heatmap screenshot here after running notebook — showing Exception Rate vs. Fraud Sensitivity vs. False Positive Risk with traffic-light coding)

 Green: Acceptable audit risk
Yellow: Requires monitoring
 Red: High risk / ineffective control

Tech Stack
Python (Pandas, NumPy, Matplotlib, Scikit-learn)
NLP: VADER Sentiment Analysis
Models: Naïve Bayes, Logistic Regression, Random Forest, SVM, Gradient Boosting
Visualization: Matplotlib (Fraud Risk Dashboard)


 Audit Opinion
In our opinion, the Random Forest classifier provides the highest reasonable assurance in detecting misleading Amazon reviews. However, residual fraud risk remains due to false positives. We recommend combining automated anomaly detection with periodic manual review as a layered control framework.
