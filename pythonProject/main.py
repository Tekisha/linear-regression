import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LassoCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer

def main(q=1.00, alphas=None, n_splits=10):
    # 1. Učitavanje i drop_duplicates
    df0 = pd.read_csv("train.tsv", sep="\t").drop_duplicates()

    # 2. Filtriranje po kvantilu
    if q < 1.0:
        th = df0["Cena"].quantile(q)
        df = df0[df0["Cena"] <= th].reset_index(drop=True)
        print(f"Kvantил {q:.2f} (prag={th:.0f} EUR) → {len(df)} uzoraka")
    else:
        df = df0
        print(f"Koristimo ceo skup: {len(df)} uzoraka")

    CURRENT_YEAR = 2025
    df["Starost"] = CURRENT_YEAR - df["Godina proizvodnje"]

    X = df.drop(columns=["Cena"])
    y = df["Cena"]

    # 3. Pipeline definicija
    num_feats = ["Starost", "Zapremina motora", "Kilometraza", "Konjske snage"]
    cat_feats = ["Marka", "Grad", "Karoserija", "Gorivo", "Menjac"]

    num_pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_feats),
        ("cat", cat_pipe, cat_feats),
    ])

    # 4. LassoCV unutar pipeline
    if alphas is None:
        alphas = np.logspace(-3, 3, 50)
    base_pipeline = Pipeline([
        ("prep", preprocessor),
        ("lasso", LassoCV(alphas=alphas, cv=5, random_state=42, n_jobs=-1, max_iter=5000))
    ])

    ttr = TransformedTargetRegressor(
        regressor=base_pipeline,
        transformer=FunctionTransformer(func=np.log1p,
                                        inverse_func=np.expm1)
    )

    # 5. Definišemo scorer za RMSE
    rmse_scorer = make_scorer(mean_squared_error, squared=False)

    # 6. K-fold cross-validation
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # cross_val_score će pozvati fit/predict za svaki fold; LassoCV će unutar toga optimizovati alpha
    scores = cross_val_score(ttr, X, y,
                             scoring=rmse_scorer,
                             cv=cv,
                             n_jobs=-1)

    print(f"{n_splits}-fold CV RMSE: {scores.mean():.2f} ± {scores.std():.2f} EUR")

    # 7. Finalno fitovanje na celom skupu i best alpha
    ttr.fit(X, y)
    best_alpha = ttr.regressor_.named_steps["lasso"].alpha_
    print(f"Best α (na celom skupu): {best_alpha:.5f}")

    return {
        "cv_rmse_mean": scores.mean(),
        "cv_rmse_std": scores.std(),
        "best_alpha": best_alpha,
        "model": ttr
    }


if __name__ == "__main__":
    stats = main(q=0.90, alphas=np.logspace(-4, 2, 100), n_splits=5)
