import pandas as pd
import numpy as np

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

def load_data(path: str):
    """
    Učitaj TSV fajl, ukloni duplikate i podeli na X i y.
    """
    df = pd.read_csv(path, sep="\t").drop_duplicates()
    X = df.drop(columns=["Cena"])
    y = df["Cena"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Podeli podatke na train i test set.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_preprocessor(numeric_features, categorical_features):
    """
    Napravi ColumnTransformer koji:
      - standardizuje numeričke
      - one-hot enkoduje kategorijske
    """
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_features),
        ("cat", cat_pipeline, categorical_features)
    ])
    return preprocessor

def build_model(preprocessor):
    """
    Napravi pipeline sa preprocesorom i Lasso regresijom.
    Parametre alpha okrećemo kroz GridSearchCV.
    """
    pipe = Pipeline([
        ("prep", preprocessor),
        ("lasso", Lasso(max_iter=10_000, random_state=42))
    ])
    # Pretraga najboljeg alpha
    param_grid = {
        "lasso__alpha": np.logspace(-3, 2, 50)
    }
    search = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    return search

def evaluate(model, X_train, y_train, X_test, y_test):
    """
    Izračunaj kros-validacioni RMSE na train-setu i konačni RMSE na testu.
    """
    # Cross-validation score (neg RMSE), pretvorimo u pozitivni RMSE
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    cv_rmse = -cv_scores.mean()
    print(f"CV RMSE (5-fold): {cv_rmse:.2f} EUR")

    # Fit na celom train setu i evaluacija na testu
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Test RMSE: {test_rmse:.2f} EUR")
    print(f"Odabrano alpha: {model.best_params_['lasso__alpha']:.5f}")

def main():
    # 1. Učitaj podatke
    X, y = load_data("train.tsv")

    # 2. Definiši feature-e
    numeric_feats = [
        "Godina proizvodnje",
        "Zapremina motora",
        "Kilometraza",
        "Konjske snage"
    ]
    categorical_feats = [
        "Marka",
        "Grad",
        "Karoserija",
        "Gorivo",
        "Menjac"
    ]

    # 3. Podela na train/test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Preprocesor
    preprocessor = build_preprocessor(numeric_feats, categorical_feats)

    # 5. Model + GridSearch za Lasso
    model_search = build_model(preprocessor)

    # 6. Evaluacija
    evaluate(model_search, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
