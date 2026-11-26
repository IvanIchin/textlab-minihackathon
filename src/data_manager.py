import pandas as pd

def load_and_inspect_csv(uploaded_file) -> tuple[pd.DataFrame, dict]:
    """Carga un CSV y devuelve el DataFrame junto con métricas de calidad."""
    df = pd.read_csv(uploaded_file)
    report = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "total_nulls": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum())
    }
    return df, report

def clean_dataset(df: pd.DataFrame, text_col: str, label_col: str, drop_nulls: bool = True, drop_dups: bool = True) -> pd.DataFrame:
    """Limpia nulos y duplicados según selección del usuario."""
    clean_df = df[[text_col, label_col]].copy()
    if drop_nulls:
        clean_df = clean_df.dropna(subset=[text_col, label_col])
    if drop_dups:
        clean_df = clean_df.drop_duplicates(subset=[text_col])
    
    clean_df[text_col] = clean_df[text_col].astype(str)
    clean_df[label_col] = clean_df[label_col].astype(str)
    return clean_df.reset_index(drop=True)

def get_class_distribution(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Calcula conteos y porcentajes por clase para la barra horizontal."""
    counts = df[label_col].value_counts().reset_index()
    counts.columns = ["Clase", "Cantidad"]
    counts["Porcentaje"] = (counts["Cantidad"] / len(df) * 100).round(2)
    return counts