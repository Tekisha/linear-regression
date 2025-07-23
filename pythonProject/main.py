import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


class CarPricePredictor:
    def __init__(self):
        self.pipeline = None
        self.best_alpha = None
        self.scaler = StandardScaler()
        self.preprocessor = None

    def load_and_explore_data(self, data_path):
        """Učitava i analizira podatke"""
        print("=== UČITAVANJE I ANALIZA PODATAKA ===")

        # Učitavanje TSV podataka
        df = pd.read_csv(data_path, sep='\t')
        print(f"Dimenzije dataseta: {df.shape}")
        print(f"\nKolone u datasetu: {list(df.columns)}")
        print(f"\nPrvih 5 redova:")
        print(df.head())

        # Osnovne informacije
        print(f"\nOsnovne informacije o podacima:")
        print(df.info())

        # Nedostajuće vrednosti
        print(f"\nNedostajuće vrednosti:")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("Nema nedostajućih vrednosti")

        # Osnovne statistike za numeričke kolone
        print(f"\nOsnovne statistike:")
        print(df.describe())

        # Pregled kategoričkih kolona
        print(f"\nKategoričke kolone i broj jedinstvenih vrednosti:")
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_count = df[col].nunique()
                print(f"  {col}: {unique_count} jedinstvenih vrednosti")
                if unique_count <= 10:
                    print(f"    Vrednosti: {list(df[col].unique())}")

        return df

    def detect_and_handle_outliers_train_only(self, X_train, y_train, numerical_columns):
        """Detektuje i uklanja outliere SAMO iz train skupa"""
        print("\n=== DETEKCIJA I TRETIRANJE OUTLIER-A (SAMO TRAIN SKUP) ===")

        # Kombinujemo X_train i y_train za lakše rukovanje
        train_df = X_train.copy()
        train_df['target'] = y_train

        outliers_removed = 0
        original_size = len(train_df)

        # Dodajemo target u listu numeričkih kolona
        all_numerical = numerical_columns + ['target']

        for col in all_numerical:
            if col in train_df.columns:
                try:
                    # Konvertujemo u numeričke vrednosti
                    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

                    # Uklanjamo NaN vrednosti
                    before_nan = len(train_df)
                    train_df = train_df.dropna(subset=[col])
                    after_nan = len(train_df)

                    if before_nan != after_nan:
                        print(f"{col}: uklonjeno {before_nan - after_nan} ne-numeričkih vrednosti")

                    # IQR metoda za outliere
                    Q1 = train_df[col].quantile(0.25)
                    Q3 = train_df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers_before = len(train_df)
                    train_df = train_df[(train_df[col] >= lower_bound) & (train_df[col] <= upper_bound)]
                    outliers_after = len(train_df)

                    removed = outliers_before - outliers_after
                    outliers_removed += removed

                    if removed > 0:
                        print(f"{col}: uklonjeno {removed} outlier-a (bounds: {lower_bound:.0f} - {upper_bound:.0f})")

                except Exception as e:
                    print(f"Greška pri obradi kolone {col}: {e}")

        print(f"Originalna veličina train skupa: {original_size}")
        print(f"Ukupno uklonjeno outlier-a: {outliers_removed}")
        print(f"Nova veličina train skupa: {len(train_df)}")
        print(f"Procenat zadržanih podataka: {len(train_df) / original_size * 100:.2f}%")

        # Vraćamo X_train_clean i y_train_clean
        y_train_clean = train_df['target']
        X_train_clean = train_df.drop(columns=['target'])

        return X_train_clean, y_train_clean

    def prepare_features(self, df):
        """Priprema feature-e za model"""
        print("\n=== PRIPREMA FEATURE-A ===")

        # Definišemo kategoričke i numeričke kolone
        categorical_features = ['Marka', 'Grad', 'Karoserija', 'Gorivo', 'Menjac']
        numerical_features = ['Godina proizvodnje', 'Zapremina motora', 'Kilometraza', 'Konjske snage']

        # Proveravamo da li sve kolone postoje
        available_categorical = [col for col in categorical_features if col in df.columns]
        available_numerical = [col for col in numerical_features if col in df.columns]

        print(f"Kategoričke kolone: {available_categorical}")
        print(f"Numeričke kolone: {available_numerical}")

        # Kreiranje preprocessor-a sa poboljšanim parametrima
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), available_numerical),
                ('cat', OneHotEncoder(
                    drop='first',
                    sparse_output=False,
                    handle_unknown='ignore',
                    min_frequency=2,  # Ignoriše kategorije koje se pojavljuju manje od 2 puta
                    max_categories=50  # Ograničava broj kategorija po koloni
                ), available_categorical)
            ],
            remainder='drop'  # Dropuje kolone koje nisu specificane
        )

        return available_categorical, available_numerical

    def split_data(self, df, target_column='Cena'):
        """Deli podatke na train i test skup"""
        print("\n=== PODELA PODATAKA ===")

        # Separacija features i target varijable
        X = df.drop(columns=[target_column])
        y = df[target_column]

        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")
        print(f"Target kolona: {target_column}")
        print(f"Opis target varijable:")
        print(f"  - Mean: {y.mean():.2f}")
        print(f"  - Std: {y.std():.2f}")
        print(f"  - Min: {y.min():.2f}")
        print(f"  - Max: {y.max():.2f}")

        # Stratified split nije moguć za regresiju, koristimo random_state za reproducibilnost
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        print(f"Train set: {X_train.shape[0]} uzoraka")
        print(f"Test set: {X_test.shape[0]} uzoraka")
        print(f"Train/Test ratio: {X_train.shape[0] / X_test.shape[0]:.2f}")

        return X_train, X_test, y_train, y_test

    def find_best_alpha_cv(self, X_train, y_train, cv_folds=5):
        """Pronalazi najbolju alpha vrednost koristeći cross-validation"""
        print("\n=== PRONALAŽENJE OPTIMALNE ALPHA VREDNOSTI ===")

        # Definišemo opseg alpha vrednosti za pretragu
        alphas = np.logspace(-4, 2, 50)  # Od 0.0001 do 100

        # Kreiramo pipeline sa preprocessor-om i Lasso modelom (povećan max_iter)
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', Lasso(max_iter=5000, tol=1e-4))  # Povećali iteracije i toleranciju
        ])

        # Grid search sa cross-validation
        param_grid = {'regressor__alpha': alphas}

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',  # Negative MSE jer sklearn maximizuje score
            n_jobs=-1,
            verbose=1
        )

        print(f"Pokretanje GridSearchCV sa {len(alphas)} alpha vrednosti i {cv_folds}-fold CV...")
        grid_search.fit(X_train, y_train)

        self.best_alpha = grid_search.best_params_['regressor__alpha']
        best_score = np.sqrt(-grid_search.best_score_)  # Konvertujemo u RMSE

        print(f"Najbolja alpha vrednost: {self.best_alpha:.6f}")
        print(f"Najbolji CV RMSE: {best_score:.2f}")

        return grid_search

    def train_final_model(self, X_train, y_train):
        """Trenira finalni model sa najboljom alpha vrednošću"""
        print("\n=== TRENIRANJE FINALNOG MODELA ===")

        # Kreiramo finalni pipeline sa povećanim brojem iteracija
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', Lasso(alpha=self.best_alpha, max_iter=5000, tol=1e-4))
        ])

        # Treniramo model
        self.pipeline.fit(X_train, y_train)

        # Dobijamo koeficijente (nakon fit-a)
        feature_names = (
                list(self.preprocessor.named_transformers_['num'].get_feature_names_out()) +
                list(self.preprocessor.named_transformers_['cat'].get_feature_names_out())
        )

        coefficients = self.pipeline.named_steps['regressor'].coef_

        # Prikazujemo najvažnije feature-e
        feature_importance = list(zip(feature_names, coefficients))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 10 najvažnijih feature-a:")
        for feature, coef in feature_importance[:10]:
            if abs(coef) > 1e-10:  # Prikazujemo samo nenulte koeficijente
                print(f"{feature}: {coef:.4f}")

    def evaluate_model(self, X_test, y_test):
        """Evaluira model na test skupu"""
        print("\n=== EVALUACIJA MODELA ===")

        # Predikcije na test skupu
        y_pred = self.pipeline.predict(X_test)

        # Računanje RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Test RMSE: {rmse:.2f}")
        print(f"Target RMSE (< 6500): {'✓ USPEH' if rmse < 6500 else '✗ NEUSPEH'}")

        # Dodatne metrike
        mae = np.mean(np.abs(y_test - y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        print(f"Test MAE: {mae:.2f}")
        print(f"Test MAPE: {mape:.2f}%")

        # Statistike predikcija
        print(f"\nStatistike predikcija:")
        print(f"Srednja vrednost realnih cena: {y_test.mean():.2f}")
        print(f"Srednja vrednost predviđenih cena: {y_pred.mean():.2f}")
        print(f"Std realnih cena: {y_test.std():.2f}")
        print(f"Std predviđenih cena: {y_pred.std():.2f}")

        return rmse, mae, mape

    def cross_validate_final_model(self, X_train, y_train, cv_folds=5):
        """Dodatna cross-validation evaluacija finalnog modela"""
        print("\n=== CROSS-VALIDATION FINALNOG MODELA ===")

        cv_scores = cross_val_score(
            self.pipeline,
            X_train,
            y_train,
            cv=cv_folds,
            scoring='neg_mean_squared_error'
        )

        cv_rmse_scores = np.sqrt(-cv_scores)

        print(f"CV RMSE scores: {cv_rmse_scores}")
        print(f"Mean CV RMSE: {cv_rmse_scores.mean():.2f} (+/- {cv_rmse_scores.std() * 2:.2f})")

        return cv_rmse_scores

    def run_complete_pipeline(self, data_path, target_column='Cena'):
        """Pokreće kompletan machine learning pipeline"""
        print("🚀 POKRETANJE KOMPLETNOG ML PIPELINE-A ZA PREDIKCIJU CENA AUTOMOBILA")
        print("=" * 80)

        # 1. Učitavanje i eksploracija podataka
        df = self.load_and_explore_data(data_path)

        # 2. Osnovna priprema podataka (konverzija tipova, basic cleaning)
        print("\n=== OSNOVNA PRIPREMA PODATAKA ===")

        # Konverzija numeričkih kolona
        numerical_cols = ['Godina proizvodnje', 'Zapremina motora', 'Kilometraza', 'Konjske snage']
        for col in numerical_cols + [target_column]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Uklanjanje redova sa NaN vrednostima u ključnim kolonama
        original_size = len(df)
        df = df.dropna(subset=[target_column])
        print(f"Uklonjeno {original_size - len(df)} redova sa nedostajućom target vrednošću")

        # 3. PRVO: Podela podataka (BEZ outlier removal)
        categorical_features, numerical_features = self.prepare_features(df)
        X_train, X_test, y_train, y_test = self.split_data(df, target_column)

        # 4. DRUGO: Outlier detection i removal SAMO na train skupu
        X_train_clean, y_train_clean = self.detect_and_handle_outliers_train_only(
            X_train, y_train, numerical_cols
        )

        print(f"\n📊 Finalne dimenzije:")
        print(f"Train skup (nakon outlier removal): {X_train_clean.shape[0]} uzoraka")
        print(f"Test skup (netaknut): {X_test.shape[0]} uzoraka")

        # 5. Pronalaženje optimalne alpha vrednosti
        grid_search = self.find_best_alpha_cv(X_train_clean, y_train_clean)

        # 6. Treniranje finalnog modela
        self.train_final_model(X_train_clean, y_train_clean)

        # 7. Cross-validation finalnog modela
        cv_scores = self.cross_validate_final_model(X_train_clean, y_train_clean)

        # 8. Evaluacija na ORIGINALNOM test skupu (netaknutom)
        rmse, mae, mape = self.evaluate_model(X_test, y_test)

        print("\n" + "=" * 80)
        print("🏁 ZAVRŠETAK PIPELINE-A")
        print(f"Finalni Test RMSE: {rmse:.2f}")
        print(f"Status: {'✅ ZADATAK USPEŠNO ZAVRŠEN' if rmse < 6500 else '❌ ZADATAK NIJE USPEŠNO ZAVRŠEN'}")
        print("\n🔒 VAŽNO: Test skup je ostao netaknut tokom outlier removal procesa!")

        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'best_alpha': self.best_alpha,
            'cv_scores': cv_scores,
            'success': rmse < 6500,
            'train_size_after_outliers': len(X_train_clean),
            'test_size': len(X_test)
        }


# Primer korišćenja:
if __name__ == "__main__":
    # Kreiranje instance predictora
    predictor = CarPricePredictor()

    # Pokretanje kompletnog pipeline-a sa TSV fajlom
    try:
        results = predictor.run_complete_pipeline('train.tsv')
        print(f"\n📊 Finalni rezultati: {results}")
    except FileNotFoundError:
        print("❗ Molimo postavite TSV fajl 'train.tsv' sa podacima o automobilima.")
        print("Fajl treba da sadrži kolone: Marka, Grad, Cena, Godina proizvodnje,")
        print("Karoserija, Gorivo, Zapremina motora, Kilometraza, Konjske snage, Menjac")