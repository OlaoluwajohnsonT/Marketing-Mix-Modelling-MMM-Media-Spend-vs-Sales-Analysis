# Marketing Mix Modelling (MMM) – Media Spend vs Sales Analysis

## Project Description
This project applies **Marketing Mix Modelling (MMM)** techniques to estimate the impact of different media channels on sales.  
Using regression and machine learning models, we decompose sales contributions, estimate **ROI by channel**, and provide actionable **budget allocation recommendations**.

---

## Project Objectives
- Analyse the effect of **TV, Radio, Digital, Print, and OOH** channels on sales revenue.  
- Identify **high ROI channels** and those with diminishing returns.  
- Build **predictive models** to forecast sales based on spend patterns.  
- Provide **strategic recommendations** for marketing budget allocation.  

---

## Dataset
- **Source:** Synthetic dataset (inspired by Kaggle).  
- **Size:** 200 rows × 14+ features.  
- **Features:**
  - Media spends: *TV Sponsorships, TV Cricket, Radio, Search Ads, Display Ads, Social Media, etc.*  
  - Target: *Sales Revenue*  
  - Engineered: *Lagged Sales, Total Marketing Spend, Cumulative Spend, Month, Weekday*  

---

## Methodology

1. **Data Preparation**  
   - Handle outliers and missing values  
   - Feature engineering (lag variables, cumulative spend, seasonality)  

2. **Exploratory Data Analysis (EDA)**  
   - Sales trends and seasonality  
   - Correlation heatmaps  
   - Media spend vs sales scatterplots  

3. **Modelling Approaches**  
   - **Baseline:** Multiple Linear Regression  
   - **Regularised Models:** Ridge, Lasso, Elastic Net  
   - **Tree-Based Models:** Random Forest, Gradient Boosting  
   - **Validation:** Train-Test Split & Cross-Validation  

4. **Channel Contribution & ROI**  
   - Feature importance analysis  
   - ROI estimation per channel  
   - Detect diminishing/increasing returns  

5. **Final Recommendations**  
   - Prioritise **Radio and TV** (top drivers of sales)  
   - Optimise **Digital channels** (Search, Display, Social)  
   - Reduce/reallocate **Print & Native Ads** (low ROI)  

---

## Tech Stack
- **Python 3.9+**
- **Libraries:**  
  - Data: `pandas`, `numpy`  
  - Visualisation: `matplotlib`, `seaborn`  
  - Modelling: `scikit-learn`, `statsmodels`  

---

## Key Results
- **Gradient Boosting** delivered the best performance (**R² ≈ 0.91, RMSE ≈ 1340**).  
- **Radio + TV** consistently emerged as the strongest sales drivers.  
- **Native Ads & Print** had negligible or no contribution.  
- Clear **ROI guidance** was derived for budget optimisation.  

---

## Deliverables
- **Jupyter Notebook**: End-to-end MMM workflow  
- **Visualisations**: Trends, feature importance, ROI by channel  
- **Report**: Executive-ready insights and recommendations  

---

## Conclusion
This MMM analysis demonstrates how **data-driven insights** can optimise marketing budgets by quantifying the contribution and ROI of each channel.  
The results provide a strong foundation for **evidence-based media planning** and **strategic decision-making**.  

---
