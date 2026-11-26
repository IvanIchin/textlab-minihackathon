import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from src.feature_extractor import TextFeatureExtractor

def build_classifier(clf_name: str, params: dict):
    """Instancia el clasificador con los hiperparámetros seleccionados."""
    if clf_name == "Regresión logística":
        return LogisticRegression(
            penalty=params.get("penalty", "l2"),
            C=params.get("C", 1.0),
            max_iter=1000,
            random_state=42
        )
    elif clf_name == "SVM":
        # probability=True permite mostrar barras de probabilidad calibrada en predicción (Fig. 3)
        return SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "linear"),
            probability=True,
            random_state=42
        )
    elif clf_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=42
        )
    elif clf_name == "Naive Bayes":
        return MultinomialNB(alpha=params.get("alpha", 1.0))
    raise ValueError(f"Clasificador no soportado: {clf_name}")

def run_experiment(df, text_col, label_col, feat_cfg, clf_name, clf_params, test_size=0.2, cv_folds=0):
    """Ejecuta el pipeline completo sin fuga de datos."""
    # Convertir explícitamente a arreglos puros de NumPy para evitar el error de PyArrow
    X = np.asarray(df[text_col].tolist(), dtype=object)
    y = np.asarray(df[label_col].tolist(), dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    extractor = TextFeatureExtractor(
        feature_type=feat_cfg["feature_type"],
        ngram_size=feat_cfg["ngram_size"],
        representation=feat_cfg["representation"]
    )
    classifier = build_classifier(clf_name, clf_params)

    pipeline = Pipeline([
        ("features", extractor),
        ("clf", classifier)
    ])

    # Validación cruzada estratificada (ajustando features en cada fold)
    cv_scores = None
    if cv_folds and cv_folds > 1:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")

    # Entrenamiento final en train y predicción en test
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "pipeline": pipeline,
        "classes": list(pipeline.classes_),
        "y_test": y_test,
        "y_pred": y_pred,
        "cv_scores": cv_scores
    }