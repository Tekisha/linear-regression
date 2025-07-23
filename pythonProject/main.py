import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor

def main():
    q = 1.00
    df0 = pd.read_csv("train.tsv", sep="\t").drop_duplicates()
    X_full = df0.drop(columns=["Cena"])
    y_full = df0["Cena"]

    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    th = y_train_full.quantile(q)
    mask = y_train_full <= th
    X_train_filt = X_train_full[mask]
    y_train_filt = y_train_full[mask]
    print(f"Filter q={q:.2f} → treniraš na {len(y_train_filt)} od {len(y_train_full)} primera (th={th:.0f} EUR)")

    num_feats = ["Godina proizvodnje", "Zapremina motora", "Kilometraza", "Konjske snage"]
    cat_feats = ["Marka", "Grad", "Karoserija", "Gorivo", "Menjac"]

    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", num_pipe, num_feats),
                                      ("cat", cat_pipe, cat_feats)])

    # 1) Definiši bazni Ridge regressor
    lasso_cv = LassoCV(
        alphas=np.logspace(-3, 3, 50),  # testira alfe od 1e-3 do 1e3
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    # --- 4. Obavij Lasso log-transformom cilja ---
    model = TransformedTargetRegressor(
        regressor=Pipeline([
            ("prep", preprocessor),
            ("lasso", lasso_cv)
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )

    model.fit(X_train_filt, y_train_filt)
    print("Odabrano alpha za Lasso:", model.regressor_.named_steps["lasso"].alpha_)

    # 2) Uzmi sam pipeline iz TransformedTargetRegressor
    pipe: Pipeline = model.regressor_

    # 3) Izvuci ColumnTransformer
    ct: ColumnTransformer = pipe.named_steps["prep"]

    # 4) Izvuci num_pipe pa StandardScaler
    scaler: StandardScaler = ct.named_transformers_["num"].named_steps["scaler"]

    # 5) Sada možeš pogledati mean_ i scale_
    print("Means learned:", scaler.mean_)
    print("Stds learned: ", scaler.scale_)

    y_pred = model.predict(X_test_final)
    rmse = mean_squared_error(y_test_final, y_pred, squared=False)
    print(f"Test RMSE nakon log-transformacije i LassoCV: {rmse:.2f} EUR")

    # print(df.shape)  # koliko redova/kolona
    # print(df.info())  # tipovi, prazne vrednosti
    # print(df.describe())  # osnovne statistike za numeričke kolone
    #
    # num_cols = ['Cena', 'Godina proizvodnje',
    #             'Zapremina motora', 'Kilometraza', 'Konjske snage']
    # df_num = df[num_cols]
    #
    # corr = df_num.corr(method='pearson')
    #
    # print(corr)
    #
    # plt.figure(figsize=(6,5))
    # sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
    #             vmin=-1, vmax=1, linewidths=0.5)
    # plt.title("Matrica korelacija (numeričke kolone)")
    # plt.tight_layout()
    # plt.show()

    # # Histogram cene
    # plt.figure(figsize=(8, 4))
    # plt.hist(df['Cena'], bins=100)
    # plt.ticklabel_format(useOffset=False, style='plain', axis='x')
    # plt.title("Distribucija cena (bez naučne notacije)")
    # plt.xlabel("Cena (EUR)")
    # plt.ylabel("Broj automobila")
    # plt.tight_layout()
    # plt.show()
    #
    # # Scatter kilometraža vs cena
    # plt.scatter(df0['Grad'], df0['Cena'], alpha=0.3)
    # plt.title("Grad vs Cena")
    # plt.xlabel("Grad")
    # plt.ylabel("Cena (EUR)")
    # plt.show()

if __name__ == '__main__':
    main()
