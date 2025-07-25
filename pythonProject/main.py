import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LassoCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import RidgeCV
import sys

Q = 0.90
PRICE_THRESHOLD = 50000
ALPHAS = np.logspace(-4, 2, 100)
N_SPLITS = 5
CURRENT_YEAR = 2025


def run_training(df_train: pd.DataFrame):
    df0 = df_train.drop_duplicates()
    df0 = df0[df0["Cena"] <= PRICE_THRESHOLD].reset_index(drop=True)
    df0["Starost"] = CURRENT_YEAR - df0["Godina proizvodnje"]

    X = df0.drop(columns=["Cena"])
    y = df0["Cena"]

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

    base_pipeline = Pipeline([
        ("prep", preprocessor),
        ("ridge", RidgeCV(alphas=ALPHAS, cv=5))
    ])

    ttr = TransformedTargetRegressor(
        regressor=base_pipeline,
        transformer=FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
    )

    rmse_scorer = make_scorer(mean_squared_error, squared=False)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    scores = cross_val_score(ttr, X, y, scoring=rmse_scorer, cv=cv, n_jobs=-1)
    ttr.fit(X, y)
    print(f"RidgeCV 5-fold RMSE: {scores.mean():.2f} ± {scores.std():.2f} EUR")
    print(f"Izabrani alpha: {ttr.regressor_.named_steps['ridge'].alpha_:.5f}")

    return ttr


def run_prediction(model_test, df_test: pd.DataFrame):
    df_test = df_test.copy()
    df_test["Starost"] = CURRENT_YEAR - df_test["Godina proizvodnje"]
    X_test = df_test.drop(columns=["Cena"], errors="ignore")

    preds = model_test.predict(X_test)
    rmse = mean_squared_error(df_test["Cena"], preds, squared=False)
    print(rmse)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        train_df = pd.read_csv(sys.argv[1], sep="\t")
        test_df = pd.read_csv(sys.argv[2], sep="\t")
        model = run_training(train_df)
        run_prediction(model, test_df)

    elif len(sys.argv) == 2:
        df = pd.read_csv(sys.argv[1], sep="\t").drop_duplicates()
        df = df[df["Cena"] <= PRICE_THRESHOLD].reset_index(drop=True)

        df['price_bin'] = pd.qcut(df['Cena'], q=5, labels=False, duplicates='drop')

        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['price_bin']
        )
        train_df = train_df.drop(columns=['price_bin'])
        test_df = test_df.drop(columns=['price_bin'])

        #print(f"Укупно {len(df)} узорака → Train: {len(train_df)}, Test: {len(test_df)}")

        model = run_training(train_df)
        run_prediction(model, test_df)
