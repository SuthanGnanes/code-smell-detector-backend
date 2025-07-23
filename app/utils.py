def preprocess_ck_metrics(class_metrics, method_metrics):
    """Preprocess CK metrics for model input"""
    # Handle missing values
    class_metrics = class_metrics.fillna(0)
    method_metrics = method_metrics.fillna(0)

    # Convert boolean columns to int
    for df in [class_metrics, method_metrics]:
        for col in df.select_dtypes(include=['bool']).columns:
            df[col] = df[col].astype(int)

    return class_metrics, method_metrics