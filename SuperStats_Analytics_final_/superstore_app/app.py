from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, binom, poisson, hypergeom
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class SuperstoreDataEngine:
    """
    Object-Oriented Data Engine for Superstore Analytics.
    This encapsulation prevents procedural global state similarities.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self._initialize_dataset()
        
    def _initialize_dataset(self):
        try:
            raw_data = pd.read_csv(self.filepath, encoding='latin1')
            
            # Format dates
            raw_data['Order Date'] = pd.to_datetime(raw_data['Order Date'], format='%m/%d/%Y', errors='coerce')
            raw_data['Year'] = raw_data['Order Date'].dt.year
            raw_data['Month'] = raw_data['Order Date'].dt.month
            
            # Ensure numerics
            metric_columns = ['Sales', 'Quantity', 'Discount', 'Profit']
            for col in metric_columns:
                raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce').fillna(0)
                
            # Add calculated fields
            raw_data['Profit_Margin_Pct'] = np.where(
                raw_data['Sales'] != 0, 
                (raw_data['Profit'] / raw_data['Sales']) * 100, 
                0
            ).round(2)
            
            return raw_data
        except Exception as e:
            print(f"Error initializing data: {str(e)}")
            return pd.DataFrame()

    def get_metrics_overview(self):
        """Generate high-level dashboard KPIs"""
        df = self.data
        return {
            "status": "success",
            "payload": {
                "totals": {
                    "sales": round(float(df['Sales'].sum()), 2),
                    "profit": round(float(df['Profit'].sum()), 2),
                    "orders": int(len(df)),
                    "discount_avg": round(float(df['Discount'].mean()) * 100, 2)
                },
                "breakdowns": {
                    "category": df.groupby('Category')['Sales'].sum().sort_values(ascending=False).to_dict(),
                    "region": df.groupby('Region')['Sales'].sum().sort_values(ascending=False).to_dict(),
                    "top_subcategories": df.groupby('Sub-Category')['Profit'].sum().nlargest(10).to_dict(),
                    "monthly_trend": df.groupby('Month')['Sales'].mean().to_dict()
                }
            }
        }

    def compute_descriptive_statistics(self):
        """Calculate detailed descriptive statistics using pandas describe where possible"""
        df = self.data
        metrics = ['Sales', 'Profit', 'Quantity', 'Discount']
        stats_payload = {}
        
        for m in metrics:
            series = df[m].dropna()
            desc = series.describe()
            n = len(series)
            se = desc['std'] / np.sqrt(n)
            
            iqr = desc['75%'] - desc['25%']
            outliers = series[(series < desc['25%'] - 1.5 * iqr) | (series > desc['75%'] + 1.5 * iqr)]
            
            stats_payload[m] = {
                "central_tendency": {
                    "mean": round(float(desc['mean']), 2),
                    "median": round(float(desc['50%']), 2),
                    "mode": round(float(series.mode()[0]), 2) if not series.mode().empty else None
                },
                "dispersion": {
                    "std": round(float(desc['std']), 2),
                    "var": round(float(series.var()), 2),
                    "cv": round(float((desc['std'] / desc['mean']) * 100), 2) if desc['mean'] != 0 else 0,
                    "range": round(float(desc['max'] - desc['min']), 2)
                },
                "position": {
                    "min": round(float(desc['min']), 2),
                    "q1": round(float(desc['25%']), 2),
                    "q3": round(float(desc['75%']), 2),
                    "max": round(float(desc['max']), 2),
                    "iqr": round(float(iqr), 2),
                    "cqd": round(float((desc['75%'] - desc['25%']) / (desc['75%'] + desc['25%'])), 4) if (desc['75%'] + desc['25%']) != 0 else 0
                },
                "shape": {
                    "skewness": round(float(series.skew()), 3),
                    "kurtosis": round(float(series.kurt()), 3),
                    "outlier_count": int(len(outliers))
                },
                "confidence_interval": {
                    "lower": round(float(desc['mean'] - 1.96 * se), 2),
                    "upper": round(float(desc['mean'] + 1.96 * se), 2)
                }
            }
            
        # Frequency table for sales
        sales = df['Sales'].dropna()
        counts, edges = np.histogram(sales, bins=8)
        total = len(sales)
        
        freq_dist = []
        cumulative = 0
        for i in range(len(counts)):
            cumulative += int(counts[i])
            freq_dist.append({
                "interval": f"{int(edges[i])} to {int(edges[i+1])}",
                "freq": int(counts[i]),
                "rel_freq": round(int(counts[i]) / total, 4),
                "cum_freq": cumulative
            })
            
        corr_matrix = df[metrics].corr().round(3).to_dict()

        return {
            "status": "success",
            "payload": {
                "variables": stats_payload,
                "frequency_distribution": freq_dist,
                "correlations": corr_matrix,
                "histogram": {
                    "bins": np.histogram(sales, bins=15)[1].round(1).tolist(),
                    "counts": np.histogram(sales, bins=15)[0].tolist()
                }
            }
        }

    def generate_probability_models(self):
        """Probability and Distribution algorithms"""
        df = self.data
        s = df['Sales'].dropna()
        p = df['Profit'].dropna()
        d = df['Discount'].dropna()
        
        mu, sig = s.mean(), s.std()
        
        # Empirical & Conditional
        empirical = {
            "p_sales_high": round(float((s > 500).mean()), 4),
            "p_profit_pos": round(float((p > 0).mean()), 4),
            "p_discount_high": round(float((d > 0.2).mean()), 4)
        }
        
        high_disc_mask = df['Discount'] > 0.2
        loss_given_disc = round(float((df[high_disc_mask]['Profit'] < 0).mean()), 4) if high_disc_mask.sum() > 0 else 0
        
        conditional = {
            "p_loss_given_high_discount": loss_given_disc,
            "p_profit_given_high_sales": round(float((df[df['Sales'] > 500]['Profit'] > 0).mean()), 4)
        }
        
        # Distributions
        n_binom = 10
        p_binom = float((p > 0).mean())
        binom_dist = [{"k": k, "p": round(float(binom.pmf(k, n_binom, p_binom)), 4)} for k in range(n_binom + 1)]
        
        lam = float(df.groupby('Order Date').size().mean())
        std = np.sqrt(lam)
        k_start = max(0, int(lam - 3 * std))
        k_end = max(15, int(lam + 3 * std))
        poisson_dist = [{"k": k, "p": round(float(poisson.pmf(k, lam)), 4)} for k in range(k_start, k_end + 1)]
        
        N_pop = 100
        K_succ = int(round(p_binom * N_pop))
        n_samp = 10
        hyper_dist = [{"k": x, "p": round(float(hypergeom.pmf(x, N_pop, K_succ, n_samp)), 4)} for x in range(11)]
        
        # Normality
        sw_stat, sw_p = stats.shapiro(s.sample(min(200, len(s)), random_state=42))
        x_norm = np.linspace(mu - 4*sig, mu + 4*sig, 100)
        y_norm = norm.pdf(x_norm, mu, sig)
        
        qq_obs = s.sample(min(150, len(s)), random_state=42).sort_values()
        qq_theo = norm.ppf(np.linspace(0.01, 0.99, len(qq_obs)), loc=mu, scale=sig)
        
        return {
            "status": "success",
            "payload": {
                "empirical": empirical,
                "conditional": conditional,
                "distributions": {
                    "binomial": {"data": binom_dist, "mean": round(n_binom*p_binom, 2)},
                    "poisson": {"data": poisson_dist, "lambda": round(lam, 2)},
                    "hypergeometric": {"data": hyper_dist, "mean": round(n_samp*(K_succ/N_pop), 2)}
                },
                "normality": {
                    "shapiro": {"w": round(sw_stat, 4), "p": round(sw_p, 4)},
                    "curve_x": x_norm.tolist(),
                    "curve_y": y_norm.tolist(),
                    "qq_observed": qq_obs.tolist(),
                    "qq_theoretical": qq_theo.tolist()
                }
            }
        }

    def execute_regression_suite(self):
        """Runs multiple and simple linear regressions + hypothesis tests"""
        df = self.data[['Sales', 'Quantity', 'Discount', 'Profit']].dropna()
        
        # Multiple Regression
        X_multi = df[['Quantity', 'Discount']].values
        y_multi = df['Sales'].values
        m_model = LinearRegression().fit(X_multi, y_multi)
        y_pred_m = m_model.predict(X_multi)
        
        idx = np.random.RandomState(99).choice(len(y_multi), min(250, len(y_multi)), replace=False)
        
        # Hypothesis Testing (Mean Sales vs Target)
        target_mu = 400.0
        s_mean = float(df['Sales'].mean())
        s_std = float(df['Sales'].std())
        n = len(df)
        t_stat = (s_mean - target_mu) / (s_std / np.sqrt(n))
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)))
        
        # Simple Regressions
        X_q = df[['Quantity']].values
        m_q = LinearRegression().fit(X_q, y_multi)
        
        X_d = df[['Discount']].values
        y_p = df['Profit'].values
        m_d = LinearRegression().fit(X_d, y_p)
        
        return {
            "status": "success",
            "payload": {
                "multiple_model": {
                    "coef_quantity": round(float(m_model.coef_[0]), 4),
                    "coef_discount": round(float(m_model.coef_[1]), 4),
                    "intercept": round(float(m_model.intercept_), 4),
                    "r2": round(float(r2_score(y_multi, y_pred_m)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_multi, y_pred_m))), 2)
                },
                "hypothesis": {
                    "t_statistic": round(t_stat, 4),
                    "p_value": round(p_val, 6),
                    "null_hypothesis": f"μ = {target_mu}",
                    "sample_mean": round(s_mean, 2)
                },
                "scatter_data": {
                    "sales_vs_qty_x": [round(float(X_q[i,0]), 1) for i in idx],
                    "sales_vs_qty_y": [round(float(y_multi[i]), 1) for i in idx],
                    "sales_vs_qty_line": [round(float(m_q.predict(X_q)[i]), 1) for i in idx],
                    "profit_vs_disc_x": [round(float(X_d[i,0]), 2) for i in idx],
                    "profit_vs_disc_y": [round(float(y_p[i]), 1) for i in idx],
                    "profit_vs_disc_line": [round(float(m_d.predict(X_d)[i]), 1) for i in idx]
                },
                "correlations": df.corr().round(3).to_dict()
            }
        }
        
    def get_paginated_data(self, page, per_page, search=''):
        df = self.data.copy()
        if search:
            df = df[df['Product Name'].str.contains(search, case=False, na=False)]
            
        total = len(df)
        start = (page - 1) * per_page
        subset = df.iloc[start:start+per_page]
        
        cols = ['Order Date', 'Region', 'Category', 'Product Name', 'Sales', 'Quantity', 'Discount', 'Profit']
        subset = subset[cols]
        subset['Order Date'] = subset['Order Date'].dt.strftime('%Y-%m-%d')
        
        return {
            "status": "success",
            "payload": {
                "total_records": total,
                "current_page": page,
                "data": subset.fillna('').to_dict(orient='records')
            }
        }



import os

# Initialize App and Engine
app = Flask(__name__)
CORS(app)

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'superstore.csv')
engine = SuperstoreDataEngine(csv_path)


# ==========================================
# Frontend Routes
# ==========================================
@app.route('/')
def dashboard(): return render_template('dashboard.html')

@app.route('/descriptive')
def descriptive(): return render_template('descriptive.html')

@app.route('/probability')
def probability(): return render_template('probability.html')

@app.route('/regression')
def regression(): return render_template('regression.html')

@app.route('/rawdata')
def rawdata(): return render_template('rawdata.html')


# ==========================================
# API Routes (v1)
# ==========================================
@app.route('/api/v1/analytics/overview')
def api_overview():
    return jsonify(engine.get_metrics_overview())

@app.route('/api/v1/analytics/descriptive')
def api_descriptive():
    return jsonify(engine.compute_descriptive_statistics())

@app.route('/api/v1/analytics/probability')
def api_probability():
    return jsonify(engine.generate_probability_models())

@app.route('/api/v1/analytics/regression')
def api_regression():
    return jsonify(engine.execute_regression_suite())

@app.route('/api/v1/analytics/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(silent=True) or {}
        
        # Safe extraction of quantity
        q_val = data.get('quantity')
        if q_val is None:
            q = 1.0
        else:
            try:
                q = float(q_val)
                if np.isnan(q):
                    q = 1.0
            except (TypeError, ValueError):
                q = 1.0
                
        # Safe extraction of discount
        d_val = data.get('discount')
        if d_val is None:
            d = 0.0
        else:
            try:
                d = float(d_val)
                if np.isnan(d):
                    d = 0.0
            except (TypeError, ValueError):
                d = 0.0
                
        df = engine.data[['Sales', 'Quantity', 'Discount']].dropna()
        if df.empty:
            return jsonify({
                "status": "error",
                "message": "No data available to train the prediction model."
            }), 400
            
        model = LinearRegression().fit(df[['Quantity', 'Discount']].values, df['Sales'].values)
        pred = max(0, model.predict([[q, d]])[0])
        
        return jsonify({
            "status": "success", 
            "payload": {"estimated_sales": round(float(pred), 2)}
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/api/v1/data/records')
def api_records():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    search = request.args.get('search', '')
    return jsonify(engine.get_paginated_data(page, per_page, search))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
