# Junior Data Scientist - TakeHome Assignment
# Delivery Time Prediction — ML Workflow

https://github.com/m1nnx/DS-Assessment-Delivery-Time-Prediction/blob/a317352d6b1745304994da3ee617f4e4be11a1a7/notebooks/starter_notebook.ipynb

Prior to the assesment requirement, I have developed an explainable machine learning and statistical modelling workflow to understand and predict delivery time. The focus is not only on prediction accuracy, but also on **interpretability, statistical validity, and practical insights**.

The workflow follows a structured progression: from data understanding and cleaning, to feature engineering, model development, and interpretation.

---

## 1. Data Ingestion & Understanding

The dataset contains delivery records with operational and contextual attributes such as:
- Delivery distance
- Weather condition
- Traffic level
- Package weight
- Driver rating
- Timestamp
- Actual delivery time (target)

Initial exploration was performed to understand:
- Data types
- Distribution of numerical variables
- Possible categories for categorical features

---

## 2. Data Cleaning & Quality Control

Several data quality issues were identified and addressed:

### Invalid Values
- Negative values were found in `package_weight`, which is not physically meaningful.
- These rows represented ~2% of the dataset and were removed as data anomalies.

### Missing Values
- `driver_rating` contained missing values.
- Missing entries were filled using the column median to preserve distribution robustness.

After cleaning:
- Dataset size reduced from 1000 to 980 rows
- No missing or invalid numerical values remained

---

## 3. Feature Engineering

The `timestamp` column was converted into datetime format and used to derive temporal features:
- Hour
- Day of week
- Day of month
- Week of month
- Week of year
- Weekend indicator

These features were introduced to capture potential behavioural and operational patterns related to delivery timing.

---

## 4. Encoding Strategy

To balance interpretability and model efficiency:

- `traffic_level` was converted into an **ordinal variable**  
  (`Low → Medium → High`) to preserve order while avoiding unnecessary dimensionality.
- `weather_condition` and selected temporal variables were one-hot encoded when treated as categorical features.

---

## 5. Statistical Modelling & Feature Selection (OLS)

Ordinary Least Squares (OLS) regression was used extensively to:
- Check linearity assumptions
- Evaluate statistical significance (p-values)
- Compare feature contributions across model iterations


### Iterative Pre-Modelling Rounds Study to get better understanding on Features Significance (100% Dataset as train & test Set)
Multiple model rounds were tested:
1. Original features only  
2. Original + all engineered time features  
3. Original + statistically significant time features  
4. Original + statistically significant time features, categorical representation of DayofWeek features  

Across these rounds:
- Model performance remained stable without train-test split (R2 ≈ 0.88)
- Many engineered time features were found to be statistically insignificant
- `DayOfWeek` and `IsWeekend` consistently showed meaningful contribution

Non-significant variables (package weight, driver rating, some time features) were removed to improve clarity without sacrificing performance.

---

## 6. Final Model Performance

Using a compact and interpretable feature set:
- **R2** ≈ 0.88  
- **MAE** ≈ 3.9 minutes  
- **MAPE** ≈ 10%

This indicates that the selected features explain a large portion of delivery time variability while remaining stable and interpretable.

---

## 7. Key Insights

- **Distance** is one of the strongest driver of delivery time.
- **Traffic level** has a clear and meaningful impact.
- **Weather conditions** significantly affect delivery duration:
  - Stormy and rainy conditions introduce the largest delays.
- Temporal effects exist, but only specific aspects (day of week, weekend) add meaningful value.

---

## 8. Partial Regression & Isolated Effects

To better understand one-to-one relationships while holding other factors constant, partial regression (Frisch-Waugh-Lovell theorem) was applied.

This allowed isolation of:
- Distance → delivery time effect
- Traffic level → delivery time effect

Findings show that:
- When other factors are controlled, each additional kilometer adds ~2 minutes to delivery time.
- Holding other factors constant, each additional unit or a level increase in Traffic level adds roughly 7.3 minutes to delivery time, showing a clear and interpretable marginal effect without ignoring operational complexity.
- This confirms a clean, interpretable relationship beyond multivariate correlations.

---

## 9. Model Training & Deployment Pipeline

Model training is implemented using an end-to-end pipeline to ensure consistency, reproducibility, and prevention of data leakage.

The dataset is first cleaned and enriched with key time-based features. Input variables are grouped by type:
- Numeric features are scaled
- Ordinal features (traffic level) are encoded with explicit ordering
- Categorical features are one-hot encoded

### Model Structure

The final model consists of:
- A preprocessing stage
- Polynomial feature expansion (degree = 2)
- Linear regression

Polynomial features allow the model to capture interaction and mild non-linear effects while remaining explainable and analytically tractable.

### Cross-Validation

Model performance is evaluated using 5-fold cross-validation:
- Each fold trains the model on a subset of data and evaluates it on unseen samples.
- Performance is assessed using R² and Mean Absolute Error (MAE).
- Consistent results across folds indicate model stability and robustness.

### Final Training & Model Persistence

After validation, the model is retrained using the full cleaned dataset to maximize learning:
- This final model is intended for deployment or further analysis.
- The trained pipeline (including preprocessing and regression) is saved as a single artifact for reproducibility.
---
## Future Improvements

1. Aplication of PCA to reduce data dimension.
2. Utilize Date & Time to introduce a time series forecasting model for better ahead planning. 
   - Rolling Means, Rolling Min Max, Lags Values and etc can be feature engineered
3. Addition of seasonal factors such as holidays, festive seasons and etc to improve model accuracy,
4. Include more data & longer timeframe.

---

## Extra - Model Deployment (Prediction App)

### Delivery Time Predictor

A simple web app to predict delivery times.

1. Install Python Packages
```bash
pip install flask pandas scikit-learn joblib
```

2. Update Model Path

Open `app.py` and change this line to where your model file is located:
```python
model = joblib.load("path/to/your/delivery_time_poly_model_with_pipeline.pkl")
```

3. Run the App
```bash
python app.py
```

4. Open in Browser

Go to: **http://localhost:5000**

![Alt text](https://github.com/m1nnx/DS-Assessment-Delivery-Time-Prediction/blob/bac5e762a155436ebd39537c2d1ad8defcc2d1c4/src/Delivery%20Prediction%20App.JPG)

---


## Answers to Questions

1. You might have noticed rows with negative package weights. If you found that 25% of the dataset had negative weights, would you drop them? If not, what would you do instead?

- Yes, I would remove them as these could be considered data anomalies. I believe in the GIGO concept: garbage in, garbage out. There is no existence of negative values in weight. Including them in model training will bias the learned relationships, and potentially distort the fitted curve (especially in polynomial regression). The best use case is to set a range of weight >0 during deployment as a preventive measure against invalid prediction inputs.

2. Imagine the traffic_level data comes from a paid API that costs us money every time we call it. How would you determine if this feature is 'worth' the cost?

- Yes, I would purchase it. traffic_level is statistically proven to be significant when fitting into OLS together with other factors. Even after holding other Betas constant, it is proven that one level increase in traffic_level (ordinal encoded) incurs 7.37 minutes to delivery time. This indicates that traffic_level has a material and interpretable effect on the target variable.

![Alt text](https://github.com/m1nnx/DS-Assignment-Delivery-Time-Prediction/blob/9db039c9450045c8b1447df6f209b031f2c58cee/src/Traffic%20Level%20Corr%20-%20Other%20X%20Controled.png)

