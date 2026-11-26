import pandas as pd

def register_experiment(session_state, clf_name, feat_cfg, metrics_summary, pipeline):
    """Guarda el experimento en sesión con ID incremental EXP-01, EXP-02..."""
    if "experiments" not in session_state:
        session_state.experiments = []

    exp_id = f"EXP-{len(session_state.experiments) + 1:02d}"
    feat_label = (
        f"{feat_cfg['ngram_size']}-gramas"
        if feat_cfg["feature_type"] == "N-gramas de palabras"
        else feat_cfg["feature_type"]
    )

    record = {
        "Experimento": exp_id,
        "Clasificador": clf_name,
        "Features": feat_label,
        "Representación": feat_cfg["representation"],
        "Accuracy (%)": metrics_summary["accuracy"],
        "F1 macro (%)": metrics_summary["f1_macro"],
        "pipeline": pipeline
    }
    session_state.experiments.append(record)
    return exp_id

def get_comparison_table(experiments: list) -> pd.DataFrame:
    """Devuelve tabla limpia sin el objeto pipeline para renderizar."""
    if not experiments:
        return pd.DataFrame()
    cols = ["Experimento", "Clasificador", "Features", "Representación", "Accuracy (%)", "F1 macro (%)"]
    return pd.DataFrame([{k: exp[k] for k in cols} for exp in experiments])

def predict_single_text(pipeline, text: str) -> tuple[str, pd.DataFrame]:
    """Predice clase y probabilidades de un texto individual."""
    pred = pipeline.predict([text])[0]
    probs = pipeline.predict_proba([text])[0]
    prob_df = pd.DataFrame({
        "Clase": pipeline.classes_,
        "Probabilidad (%)": (probs * 100).round(1)
    }).sort_values(by="Probabilidad (%)", ascending=False)
    return pred, prob_df

def predict_batch_csv(pipeline, csv_file, text_col: str) -> pd.DataFrame:
    """Predice clases para un CSV sin etiquetas."""
    df = pd.read_csv(csv_file)
    texts = df[text_col].astype(str).tolist()
    df["Clase_Predicha"] = pipeline.predict(texts)
    return df