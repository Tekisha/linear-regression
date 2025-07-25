import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Učitavanje podataka
df = pd.read_csv("train.tsv", sep="\t")

# Osnovne informacije
print("Dimenzije skupa:", df.shape)
print("\nBroj nedostajućih vrednosti po kolonama:")
print(df.isna().sum())

# Uklanjanje duplikata
df = df.drop_duplicates()

# Statistika numeričkih kolona
print("\nOpis numeričkih kolona:")
print(df.describe().T)

numericke_kolone = ["Cena", "Godina proizvodnje", "Zapremina motora", "Kilometraza", "Konjske snage"]
medijane = df[numericke_kolone].median()
quater = df[numericke_kolone].quantile(0.25)

print("\nMedijane po kolonama:")
print(medijane)

print("\n25% po kolonama:")
print(quater)

# Dodavanje kolone 'Starost'
df["Starost"] = 2025 - df["Godina proizvodnje"]

# Korelacija
numericke = ["Cena", "Starost", "Zapremina motora", "Kilometraza", "Konjske snage"]
corr_matrix = df[numericke].corr()
print("\nKorelaciona matrica:")
print(corr_matrix)

# Broj unikatnih vrednosti po kategorijama
kategoricke = ["Marka", "Grad", "Karoserija", "Gorivo", "Menjac"]
print("\nBroj unikatnih vrednosti u kategoričkim kolonama:")
for col in kategoricke:
    print(f"{col}: {df[col].nunique()}")

# Vizualizacije:
plt.figure(figsize=(10, 5))
sns.histplot(df["Cena"], kde=True, bins=50)
plt.title("Distribucija cena")
plt.xlabel("Cena (EUR)")
plt.tight_layout()
plt.savefig("hist_cena.png")

plt.figure(figsize=(8, 6))
sns.boxplot(x=df["Cena"])
plt.title("Boxplot cena")
plt.tight_layout()
plt.savefig("box_cena.png")

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Korelaciona matrica")
plt.tight_layout()
plt.savefig("corr_matrix.png")

df = pd.read_csv("train.tsv", sep="\t")
df = df.drop_duplicates()
df["Starost"] = 2025 - df["Godina proizvodnje"]

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Starost", y="Cena", alpha=0.5)
plt.title("Cena u odnosu na starost automobila")
plt.xlabel("Starost vozila (godine)")
plt.ylabel("Cena [EUR]")
plt.tight_layout()
plt.savefig("scatter_starost_vs_cena.png")