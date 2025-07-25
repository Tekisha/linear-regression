import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# Parametri
PRICE_THRESHOLD = 50000
CURRENT_YEAR = 2025
N_SPLITS = 5
ALPHAS = np.logspace(-4, 2, 100)

# Učitavanje podataka
df = pd.read_csv("train.tsv", sep="\t").drop_duplicates()
df = df[df["Cena"] <= PRICE_THRESHOLD].copy()
df["Starost"] = CURRENT_YEAR - df["Godina proizvodnje"]

X = df.drop(columns=["Cena"])
y = df["Cena"]

# Numeričke i kategoričke kolone
num_feats = ["Starost", "Zapremina motora", "Kilometraza", "Konjske snage"]
cat_feats = ["Marka", "Grad", "Karoserija", "Gorivo", "Menjac"]

# Preprocesiranje
num_pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_feats),
    ("cat", cat_pipe, cat_feats)
])

# RMSE scorer
rmse_scorer = make_scorer(mean_squared_error, squared=False)
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Modeli
models = {
    "RidgeCV": RidgeCV(alphas=ALPHAS, cv=5),
    "ElasticNetCV": ElasticNetCV(alphas=ALPHAS, l1_ratio=[0.1, 0.5, 0.9], cv=5, n_jobs=-1, max_iter=10000)
}

# Evaluacija
for name, model in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("reg", model)
    ])

    ttr = TransformedTargetRegressor(
        regressor=pipe,
        transformer=FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
    )

    scores = cross_val_score(ttr, X, y, scoring=rmse_scorer, cv=cv, n_jobs=-1)
    print(f"{name} → RMSE (log-target): {scores.mean():.2f} ± {scores.std():.2f} EUR")
