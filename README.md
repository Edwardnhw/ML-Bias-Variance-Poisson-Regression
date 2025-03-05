# Bike Sharing Demand Analysis Using Regression Models

**Author**: Hon Wa Ng\
**Date**: October 2024  

## Overview

This repository implements regression-based techniques to analyze and predict bike-sharing demand. The project explores Ordinary Least Squares (OLS) regression, Locally Weighted Regression (LWR), and Poisson regression to model rental counts based on weather, time, and seasonal features.

The dataset used in this analysis is included in the repository under the data/ directory.

## Objectives

- Perform exploratory data analysis (EDA) on bike rental data.
- Apply OLS regression to model rental demand.
- Implement Locally Weighted Regression (LWR) for non-linear modeling.
- Develop a Poisson regression model to predict count-based outcomes.
- Compare model performance using R² and D² scores.

## Repository Structure
```bash
Bike-Sharing-Regression/
│── data/                            # Dataset storage
│   ├── bike+sharing+dataset          # Original dataset 
│   ├── hour.csv                      # Hourly bike rental dataset
│
│── docs/                            # Documentation files
│   ├── assignment_questions.pdf       # Problem statement
│   ├── project_writeup.pdf            # Detailed project analysis
│
│── src/                             # Source code
│   ├── main.py                        # Main execution script
│
│── LICENSE                          # MIT License
│── requirements.txt                  # Dependencies for running the project
│── README.md                         # Project documentation


```

---

## Installation & Usage

### 1. Clone the Repository
```
git clone https://github.com/Edwardnhw/Bike-Sharing-Regression.git
cd Bike-Sharing-Regression

```

### 2. Install Dependencies
Ensure you have Python installed (>=3.7), then run:
```
pip install -r requirements.txt

```

---
## How to Run the Project
Execute the main script to run regression models:

```
python src/main.py

```
The script will:

- Load the dataset (hour.csv).
- Perform exploratory data analysis (EDA).
- Train and evaluate OLS, LWR, and Poisson regression models.
- Output R² and D² scores for performance evaluation.

---
## Methods Used

1. Exploratory Data Analysis (EDA)
- Summary statistics and missing value analysis.
- Feature correlation heatmap.
- Visualizing rental demand patterns.
2. Feature Engineering & Selection
- Removing redundant columns (instant, atemp, registered, casual, dteday).
- One-hot encoding categorical variables (season, month, hour, weekday, weather).
3. Regression Models Implemented
- Ordinary Least Squares (OLS) Regression
  - Closed-form solution using matrix inversion.
  - Feature selection using one-hot encoding.
- Locally Weighted Regression (LWR)
  - Assigns weights to data points based on proximity.
  - Adaptive predictions using bandwidth parameter τ.
  - Poisson Regression
- Gradient descent implementation.
  - Uses Tweedie deviance (D²) for performance evaluation.

---

## Results & Analysis

- Temperature and hour of the day are the most significant predictors of bike rentals.
- Poisson regression outperforms OLS for count-based predictions.
- Feature engineering improves model accuracy.
- LWR captures non-linear patterns but is computationally expensive.

For detailed performance evaluation, refer to project_writeup.pdf in the docs/ folder.
---
## License
This project is licensed under the MIT License.



