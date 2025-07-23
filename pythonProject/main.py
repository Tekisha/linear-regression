import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main():
    df0 = pd.read_csv("train.tsv", sep="\t").drop_duplicates()
    num_feats = ["Godina proizvodnje", "Zapremina motora", "Kilometraza", "Konjske snage"]
    cat_feats = ["Marka", "Grad", "Karoserija", "Gorivo", "Menjac"]

    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore") )])
    preprocessor = ColumnTransformer([("num", num_pipe, num_feats),
                                      ("cat", cat_pipe, cat_feats)])
    base_model = Pipeline([("prep", preprocessor),
                            ("reg", Ridge(alpha=1.0, random_state=42))])

    quantiles = [0.80, 0.85, 0.90, 0.95, 1.00]

    results = []
    for q in quantiles:
        # 1. napravi threshold i filtriraj
        th = df0["Cena"].quantile(q)
        df = df0[df0["Cena"] <= th]

        # 2. split
        X = df.drop(columns=["Cena"])
        y = df["Cena"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 3. cross-val RMSE na treningu
        neg_mse = cross_val_score(base_model, X_train, y_train,
                                  scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
        cv_rmse = -neg_mse.mean()

        # 4. fit i test RMSE
        base_model.fit(X_train, y_train)
        y_pred = base_model.predict(X_test)
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)

        results.append((q, cv_rmse, test_rmse))
        print(f"Quantile {q:.2f} → CV RMSE: {cv_rmse:.0f}, Test RMSE: {test_rmse:.0f}")

    # 5. pronađi najbolji (po CV ili po testu)
    best = min(results, key=lambda x: x[1])  # po CV
    print(f"\nNajbolji kvantil po CV RMSE: {best[0]:.2f} → {best[1]:.0f}")

    # Ako hoćeš po test:
    best_test = min(results, key=lambda x: x[2])
    print(f"Najbolji kvantil po Test RMSE: {best_test[0]:.2f} → {best_test[2]:.0f}")

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
    # plt.scatter(df['Kilometraza'], df['Cena'], alpha=0.3)
    # plt.title("Kilometraža vs Cena")
    # plt.xlabel("Kilometraža (km)")
    # plt.ylabel("Cena (EUR)")
    # plt.show()

if __name__ == '__main__':
    main()
