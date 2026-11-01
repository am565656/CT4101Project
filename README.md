# CT4101Project

Classification question: How do age, blood pressure, blood sugar, body temperature, and heart rate contribute to maternal mortality risk in rural Bangladesh using machine learning prediction?

Dataset source: Maternal Health Risk, UCI Machine Learning Repository
M. Ahmed. "Maternal Health Risk," UCI Machine Learning Repository, 2020. [Online]. Available: https://doi.org/10.24432/C5DP5D.

Folder structure: README.md explains the project introduction

Setup
Create and activate a virtual environment:

python -m venv .venv
.venv\Scripts\Activate.ps1

Preprocessing
Missing value analysis
Duplicate detection and removal
Outlier analysis using the IQR method
Target label encoding

Models
Baseline:
Stratified Dummy Classifier
Simple Learned Baseline:
Decision Tree (max_depth=3)
Validation:
5-Fold Stratified Cross Validation

Assistance Declaration
Generative AI tools were used for methodological guidance, debugging support and drafting documentation. All code, analysis and conclusions were reviewed and validated by the author.

Repository link:
https://github.com/am565656/CT4101Project
