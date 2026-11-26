import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_test, y_pred, classes):
    """Calcula métricas globales macro y métricas individuales por clase."""
    acc = accuracy_score(y_test, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, average=None, zero_division=0
    )

    per_class_df = pd.DataFrame({
        "Clase": np.tile(classes, 3),
        "Métrica": ["Precision"] * len(classes) + ["Recall"] * len(classes) + ["F1-score"] * len(classes),
        "Valor": np.concatenate([prec_c, rec_c, f1_c]) * 100
    })

    summary = {
        "accuracy": round(acc * 100, 2),
        "precision_macro": round(prec_macro * 100, 2),
        "recall_macro": round(rec_macro * 100, 2),
        "f1_macro": round(f1_macro * 100, 2),
    }
    return summary, per_class_df

def plot_per_class_performance(per_class_df):
    fig = px.bar(
        per_class_df, x="Clase", y="Valor", color="Métrica",
        barmode="group", text=per_class_df["Valor"].round(1),
        color_discrete_sequence=["#4472C4", "#2E9E6B", "#D99B43"]
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 110], yaxis_title="%", xaxis_title="", margin=dict(t=20, b=20))
    return fig

def plot_confusion_matrix(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig = px.imshow(
        cm, x=classes, y=classes, text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Clase predicha", y="Clase real", color="Conteo")
    )
    fig.update_xaxes(side="top")
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=20))
    return fig

def plot_cv_scores(cv_scores):
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    vals = np.round(cv_scores * 100, 2)
    fig = go.Figure(data=[go.Bar(x=folds, y=vals, text=vals, textposition="outside", marker_color="#4472C4")])
    fig.update_layout(yaxis_range=[0, 110], yaxis_title="Accuracy (%)", margin=dict(t=20, b=20))
    return fig