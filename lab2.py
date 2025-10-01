import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import matplotlib.pyplot as plt

# Optional: for RandomizedSearchCV distributions
try:
    from scipy.stats import loguniform
    HAVE_SCI_PY = True
except Exception:
    HAVE_SCI_PY = False


# ==============================
# 1) Load data
# ==============================
#housing = pd.read_csv(r"C:\Users\peter\handson-ml2\datasets\housing\housing.csv")
housing  = pd.read_csv("housing.csv")

# ==============================
# 2) Stratified split by income category (book/lab practice)
# ==============================
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3., 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_idx].drop(columns=["income_cat"])
    strat_test_set  = housing.loc[test_idx].drop(columns=["income_cat"])

X_train = strat_train_set.drop("median_house_value", axis=1)
y_train = strat_train_set["median_house_value"].copy()
X_test  = strat_test_set.drop("median_house_value", axis=1)
y_test  = strat_test_set["median_house_value"].copy()


# ==============================
# 3) Custom feature engineering transformer
#    (rooms_per_household, population_per_household, bedrooms_per_room)
# ==============================
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def _init_(self, rooms_ix, bedrooms_ix, population_ix, households_ix, add_bedrooms_per_room=True):
        self.rooms_ix = rooms_ix
        self.bedrooms_ix = bedrooms_ix
        self.population_ix = population_ix
        self.households_ix = households_ix
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# ==============================
# 4) Identify columns & build full pipeline
# ==============================
num_attribs = list(X_train.drop("ocean_proximity", axis=1).columns)
cat_attribs = ["ocean_proximity"]

# indices for feature engineering (based on the order in num_attribs)
rooms_ix      = num_attribs.index("total_rooms")
bedrooms_ix   = num_attribs.index("total_bedrooms")
population_ix = num_attribs.index("population")
households_ix = num_attribs.index("households")

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder(
        #rooms_ix=rooms_ix,
        #bedrooms_ix=bedrooms_ix,
        #population_ix=population_ix,
        #households_ix=households_ix,
        #add_bedrooms_per_room=True
    )),
    ("scaler", StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
])

# Fit on train only, transform train and test
X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared  = full_pipeline.transform(X_test)

print("Prepared shapes -> train:", X_train_prepared.shape, " test:", X_test_prepared.shape)


# ==============================
# 5) Helper: CV RMSE
# ==============================
def rmse_cv(model, X, y, cv=10):
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
    rmse = np.sqrt(-scores)
    return rmse.mean(), rmse.std(), rmse  # return full vector too if needed


# ==============================
# 6) Train baseline models + 10-fold CV
# ==============================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

cv_summary = {}  # name -> (mean, std)

for name, mdl in models.items():
    mdl.fit(X_train_prepared, y_train)
    mean_rmse, std_rmse, _ = rmse_cv(mdl, X_train_prepared, y_train, cv=10)
    cv_summary[name] = (mean_rmse, std_rmse)
    print(f"{name:>16}: CV RMSE mean={mean_rmse:,.0f}, std={std_rmse:,.0f}")


# ==============================
# 7) SVR experiments (GridSearchCV and RandomizedSearchCV)
# ==============================
# GridSearch over a reasonable grid
svr = SVR()
param_grid = {
    "kernel": ["rbf"],
    "C": [3, 10, 30, 100],
    "gamma": ["scale", 0.1, 0.03, 0.01],
    "epsilon": [0.1, 0.2, 0.3]
}
grid_svr = GridSearchCV(svr, param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=0)
grid_svr.fit(X_train_prepared, y_train)
best_svr_grid = grid_svr.best_estimator_
grid_mean, grid_std, _ = rmse_cv(best_svr_grid, X_train_prepared, y_train, cv=10)
cv_summary["SVR (GridSearch)"] = (grid_mean, grid_std)
print(f"\nSVR (GridSearch) best params: {grid_svr.best_params_}")
print(f"SVR (GridSearch): CV RMSE mean={grid_mean:,.0f}, std={grid_std:,.0f}")

# RandomizedSearch over broader ranges (if SciPy available)
if HAVE_SCI_PY:
    rand_params = {
        "kernel": ["rbf"],
        "C": loguniform(1, 300),
        "gamma": loguniform(1e-3, 1),
        "epsilon": [0.05, 0.1, 0.2, 0.3]
    }
    rand_svr = RandomizedSearchCV(
        SVR(), rand_params, n_iter=25,
        scoring="neg_mean_squared_error", cv=5, n_jobs=-1,
        random_state=42, verbose=0
    )
    rand_svr.fit(X_train_prepared, y_train)
    best_svr_rand = rand_svr.best_estimator_
    rand_mean, rand_std, _ = rmse_cv(best_svr_rand, X_train_prepared, y_train, cv=10)
    cv_summary["SVR (RandomizedSearch)"] = (rand_mean, rand_std)
    print(f"\nSVR (RandomizedSearch) best params: {rand_svr.best_params_}")
    print(f"SVR (RandomizedSearch): CV RMSE mean={rand_mean:,.0f}, std={rand_std:,.0f}")
else:
    best_svr_rand = None
    print("\n(SciPy not available: skipping RandomizedSearchCV for SVR.)")


# ==============================
# 8) Pick the overall best-by-CV model and evaluate on test
# ==============================
best_name = min(cv_summary, key=lambda k: cv_summary[k][0])
best_cv_mean = cv_summary[best_name][0]

# ensure we have a fitted model instance
name_to_model = {
    "Linear Regression": models["Linear Regression"],
    "Decision Tree": models["Decision Tree"],
    "Random Forest": models["Random Forest"],
    "SVR (GridSearch)": best_svr_grid,
}
if best_svr_rand is not None:
    name_to_model["SVR (RandomizedSearch)"] = best_svr_rand

best_model = name_to_model[best_name]

# Evaluate on the held-out test set (no refit; just predict)
test_preds = best_model.predict(X_test_prepared)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
print(f"\nBest by CV: {best_name} (CV mean RMSE={best_cv_mean:,.0f})")
print("TEST RMSE:", f"{test_rmse:,.0f}")


# ==============================
# 9) Figure 1 — CV RMSE bar chart (mean ± std), saved as cv_rmse_bar.png
# ==============================
def plot_cv_rmse_bar(cv_results: dict, filename="cv_rmse_bar.png"):
    """
    cv_results: dict {model_name: (mean_rmse, std_rmse)}
    Saves a simple bar chart with error bars. No styles or colors specified.
    """
    labels = list(cv_results.keys())
    means = np.array([cv_results[k][0] for k in labels], dtype=float)
    stds  = np.array([cv_results[k][1] for k in labels], dtype=float)

    x = np.arange(len(labels))
    plt.figure(figsize=(7.5, 4.5))
    bars = plt.bar(x, means, yerr=stds, capsize=5)

    # Print exact values above bars
    for i, (m, s) in enumerate(zip(means, stds)):
        offset = s if np.isfinite(s) and s > 0 else 1000.0
        plt.text(i, m + offset, f"{m:,.0f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("RMSE")
    plt.title("Cross-Validation RMSE by Model (10-fold)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved Figure 1 to: {filename}")

plot_cv_rmse_bar(cv_summary, filename="cv_rmse_bar.png")