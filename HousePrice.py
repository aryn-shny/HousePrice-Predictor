from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline 
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["price"] = data.target

X = df.drop(columns="price")
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 2 - Parameter grid (just a dictionary)
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__max_depth": [3, 5, 7]
}
# '__' Since the GradientBoostingRegressor is wrapped inside a Pipeline under the name "model", 
# you have to tell GridSearchCV where to find it. The syntax is:
# "stepname__parametername"

# Step 3 - Pipeline
gb_pipeline = Pipeline(steps=[
    ("scaling", StandardScaler()),
    ("model", GradientBoostingRegressor(random_state=42)) # Removed n_estimator and learning_rate cuz GCV takes care of that
])

# Step 4 - GridSearchCV setup
grid_search = GridSearchCV(
    estimator=gb_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=2
)

# Step 5 - Fit (this will take a minute)
grid_search.fit(X_train, y_train)

# Step 6 - Best settings
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score (RMSE):", round(-grid_search.best_score_, 4))

# Step 7 - Evaluate on test data
best_predictions = grid_search.predict(X_test)
best_rmse = np.sqrt(mean_squared_error(y_test, best_predictions))
best_r2 = r2_score(y_test, best_predictions)

print("\n--- Tuned Gradient Boosting ---")
print("RMSE:", round(best_rmse, 4))
print("R2 Score:", round(best_r2, 4))


# Phase 6 — Feature Importance

# Step 1: Extract the trained model from inside the Pipeline
# Your pipeline has two steps: "scaling" and "model"
# grid_search.best_estimator_ is the full pipeline, so we reach inside with named_steps
gb_model = grid_search.best_estimator_.named_steps['model']

# Step 2: Get feature importances
# .feature_importances_ is an attribute that exists on tree-based models
# It returns a numpy array — one number per feature
# All values add up to 1.0 (they're proportions)
importances = gb_model.feature_importances_

# Step 3: Get feature names
# X is your dataframe, so X.columns gives the column names
feature_names = X.columns

# Step 4: Sort features from most to least important
# np.argsort returns the indices that would sort the array
# [::-1] reverses it so highest comes first
sorted_indices = np.argsort(importances)[::-1]

# Step 5: Print a ranked list
print("Feature Importance Ranking:")
print("-" * 35)
for rank, idx in enumerate(sorted_indices):
    print(f"{rank+1}. {feature_names[idx]:<12} {importances[idx]:.4f}")

# Step 6: Visualise as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_indices])
plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=45)
plt.title("Feature Importance — Gradient Boosting (Tuned)")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()