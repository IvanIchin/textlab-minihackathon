import streamlit as st
import plotly.express as px
from src.data_manager import load_and_inspect_csv, clean_dataset, get_class_distribution
from src.model_trainer import run_experiment
from src.evaluator import compute_metrics, plot_per_class_performance, plot_confusion_matrix, plot_cv_scores
from src.experiment_hub import register_experiment, get_comparison_table, predict_single_text, predict_batch_csv

st.set_page_config(page_title="TextLab - ML Experiment Studio", layout="wide")

# Encabezado estilo TextLab
st.markdown("""
<div style="background-color:#162436; padding:18px 24px; border-radius:10px; margin-bottom:20px;">
    <h1 style="color:white; margin:0; font-size:28px;">TextLab</h1>
    <p style="color:#CBD5E1; margin:0; font-size:14px;">Machine Learning Experiment Studio</p>
</div>
""", unsafe_allow_html=True)

if "experiments" not in st.session_state:
    st.session_state.experiments = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

tab_exp, tab_comp = st.tabs(["Configuración y Resultados", "Comparación y Predicción"])

# ==========================================
# PESTAÑA 1: CONFIGURACIÓN Y RESULTADOS
# ==========================================
with tab_exp:
    st.subheader("1. Dataset")
    uploaded_file = st.file_uploader("Cargar dataset etiquetado (CSV)", type=["csv"])

    if uploaded_file:
        raw_df, report = load_and_inspect_csv(uploaded_file)
        c1, c2, c3, c4 = st.columns(4)
        text_col = c1.selectbox("Columna de texto", report["columns"], index=0)
        label_col = c2.selectbox("Columna de etiqueta", report["columns"], index=min(1, len(report["columns"])-1))
        drop_nulls = c3.checkbox(f"Eliminar nulos ({report['total_nulls']})", value=True)
        drop_dups = c4.checkbox(f"Eliminar duplicados ({report['duplicates']})", value=True)

        df = clean_dataset(raw_df, text_col, label_col, drop_nulls, drop_dups)
        dist_df = get_class_distribution(df, label_col)

        st.caption(f"**{uploaded_file.name}** · **{len(df):,} textos limpios** · **{len(dist_df)} clases detectadas**")
        fig_dist = px.bar(
            dist_df, x="Porcentaje", y=["Distribución"] * len(dist_df), color="Clase",
            orientation="h", text=dist_df.apply(lambda r: f"{r['Clase']}: {r['Porcentaje']}% ({r['Cantidad']})", axis=1)
        )
        fig_dist.update_layout(height=110, showlegend=False, xaxis_visible=False, yaxis_visible=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_dist, use_container_width=True)

        st.divider()
        col_feat, col_mod = st.columns(2)

        # 2. Extracción de características
        with col_feat:
            st.subheader("2. Extracción de características")
            feature_type = st.selectbox("Tipo de características", ["N-gramas de palabras", "Etiquetas POS", "Stopwords", "Content words"])
            ngram_size = st.radio("Tamaño del n-grama", [1, 2, 3], horizontal=True)
            representation = st.radio("Representación", ["Frecuencia", "TF-IDF", "Binaria"], index=1, horizontal=True)

        # 3. Configuración del modelo
        with col_mod:
            st.subheader("3. Configuración del modelo")
            clf_name = st.selectbox("Clasificador", ["Regresión logística", "SVM", "Random Forest", "Naive Bayes"])
            train_pct = st.slider("Proporción Entrenamiento (%)", 50, 90, 80, step=5)

            clf_params = {}
            p1, p2 = st.columns(2)
            if clf_name == "Regresión logística":
                clf_params["penalty"] = p1.selectbox("Regularización", ["l2", None])
                clf_params["C"] = p2.selectbox("Parámetro C", [0.1, 1.0, 5.0, 10.0], index=1)
            elif clf_name == "SVM":
                clf_params["kernel"] = p1.selectbox("Kernel", ["linear", "rbf"])
                clf_params["C"] = p2.selectbox("Parámetro C", [0.1, 1.0, 5.0, 10.0], index=1)
            elif clf_name == "Random Forest":
                clf_params["n_estimators"] = p1.selectbox("Árboles (n_estimators)", [50, 100, 200], index=1)
                clf_params["max_depth"] = p2.selectbox("Profundidad máxima", [None, 10, 20, 50])
            elif clf_name == "Naive Bayes":
                clf_params["alpha"] = p1.selectbox("Suavizado (alpha)", [0.1, 0.5, 1.0], index=2)

            cv_choice = st.radio("Validación cruzada", ["Sin CV", "3 folds", "5 folds", "10 folds"], index=2, horizontal=True)
            cv_folds = 0 if cv_choice == "Sin CV" else int(cv_choice.split()[0])

        if st.button("🚀 Entrenar modelo", type="primary", use_container_width=True):
            with st.spinner("Entrenando pipeline sin fuga de datos..."):
                feat_cfg = {"feature_type": feature_type, "ngram_size": ngram_size, "representation": representation}
                res = run_experiment(df, text_col, label_col, feat_cfg, clf_name, clf_params, test_size=(100 - train_pct) / 100.0, cv_folds=cv_folds)
                summary, per_class_df = compute_metrics(res["y_test"], res["y_pred"], res["classes"])
                exp_id = register_experiment(st.session_state, clf_name, feat_cfg, summary, res["pipeline"])
                st.session_state.last_result = {"exp_id": exp_id, "summary": summary, "per_class_df": per_class_df, "res": res}
                st.success(f"Experimento **{exp_id}** completado y guardado en historial.")

        # Resultados del experimento (Figura 2)
        if st.session_state.last_result:
            st.divider()
            lr = st.session_state.last_result
            st.subheader(f"Resultados del experimento ({lr['exp_id']})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{lr['summary']['accuracy']}%")
            m2.metric("Precision macro", f"{lr['summary']['precision_macro']}%")
            m3.metric("Recall macro", f"{lr['summary']['recall_macro']}%")
            m4.metric("F1-score macro", f"{lr['summary']['f1_macro']}%")

            st.markdown("**Desempeño por clase (Precision / Recall / F1)**")
            st.plotly_chart(plot_per_class_performance(lr["per_class_df"]), use_container_width=True)

            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Matriz de confusión**")
                st.plotly_chart(plot_confusion_matrix(lr["res"]["y_test"], lr["res"]["y_pred"], lr["res"]["classes"]), use_container_width=True)
            with g2:
                st.markdown("**Validación cruzada**")
                if lr["res"]["cv_scores"] is not None:
                    scores = lr["res"]["cv_scores"]
                    st.plotly_chart(plot_cv_scores(scores), use_container_width=True)
                    st.caption(f"Media: **{scores.mean()*100:.2f}%** · DE: **{scores.std()*100:.2f} pp**")
                else:
                    st.info("Validación cruzada desactivada para este experimento.")

# ==========================================
# PESTAÑA 2: COMPARACIÓN Y PREDICCIÓN
# ==========================================
with tab_comp:
    st.subheader("4. Historial de experimentos")
    if not st.session_state.experiments:
        st.info("Aún no hay experimentos entrenados en esta sesión.")
    else:
        comp_df = get_comparison_table(st.session_state.experiments)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("5. Predicción de nuevos textos")
        exp_options = {f"{e['Experimento']} · {e['Clasificador']} ({e['Features']})": e["pipeline"] for e in st.session_state.experiments}
        selected_label = st.selectbox("Modelo seleccionado", list(exp_options.keys()))
        active_pipeline = exp_options[selected_label]

        mode = st.radio("Modo de entrada", ["Texto individual", "Archivo CSV sin etiquetas"], horizontal=True)
        if mode == "Texto individual":
            sample_text = st.text_area("Escribe el texto a clasificar", placeholder="Ej. El producto llegó antes de lo esperado y su calidad es excelente.")
            if st.button("Predecir texto") and sample_text.strip():
                pred_class, prob_df = predict_single_text(active_pipeline, sample_text)
                st.success(f"### Clase predicha: {pred_class}")
                fig_prob = px.bar(prob_df, x="Probabilidad (%)", y="Clase", orientation="h", text="Probabilidad (%)")
                fig_prob.update_layout(height=200, xaxis_range=[0, 105], margin=dict(t=10, b=10))
                st.plotly_chart(fig_prob, use_container_width=True)
        else:
            unlabeled_csv = st.file_uploader("Subir CSV sin etiquetar", type=["csv"], key="pred_csv")
            if unlabeled_csv:
                import pandas as pd
                preview_df = pd.read_csv(unlabeled_csv)
                unlabeled_csv.seek(0)
                pred_col = st.selectbox("Columna con los textos", list(preview_df.columns))
                if st.button("Predecir lote"):
                    result_df = predict_batch_csv(active_pipeline, unlabeled_csv, pred_col)
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button("Descargar predicciones (CSV)", result_df.to_csv(index=False).encode("utf-8"), "predicciones_textlab.csv", "text/csv")