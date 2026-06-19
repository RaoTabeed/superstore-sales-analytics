# Superstore Sales Analytics
## Probability & Statistics Project

A Flask-based web application for statistical analysis of the Global Superstore dataset.

---

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Make sure `superstore.csv` is in the same folder as `app.py`

### 3. Start the server
```
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | KPI cards, category/region charts, yearly trend |
| Descriptive Stats | `/descriptive` | Mean, median, CI, histogram, correlation matrix |
| Probability | `/probability` | Empirical prob, normality test, binomial, Poisson, QQ plot |
| Regression | `/regression` | Multiple linear regression, scatter, residuals, predictor |
| Raw Data | `/rawdata` | Searchable, filterable paginated table |

---

## Dataset

**Source:** Kaggle — Sample Superstore Dataset  
**Records:** ~9,994 orders  
**Variables:** Order Date, Region, Category, Sub-Category, Product Name, Sales, Quantity, Discount, Profit

---

## Statistical Methods Used

- Descriptive stats:  median, ,variance, IQR, quartiles
- 95% Confidence Intervals
- Skewness and Kurtosis
- Correlation Matrix (Pearson)
- Shapiro-Wilk Normality Test
- Empirical Probability
- Normal Distribution Approximation (Z-score)
- QQ Plot
- Binomial Distribution
- Poisson Distribution
- Conditional Probability
- Simple and Multiple Linear Regression (OLS)
- R², RMSE
- Residual Analysis
