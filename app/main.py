import os
import shutil
import subprocess
import tempfile
import zipfile
import numpy as np
import pandas as pd
import shap
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and encoders
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
class_models = joblib.load(os.path.join(MODELS_DIR, "class_smell_models.pkl"))
method_models = joblib.load(os.path.join(MODELS_DIR, "method_smell_models.pkl"))
class_label_encoder = joblib.load(os.path.join(MODELS_DIR, "class_label_encoder.pkl"))
method_label_encoder = joblib.load(os.path.join(MODELS_DIR, "method_label_encoder.pkl"))

# CK tool path
CK_JAR_PATH = os.path.join(os.path.dirname(__file__), "ck", "ck-0.7.1.jar")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "matrix_")
MATRIX_DIR = os.path.join(os.path.dirname(__file__), "output")


def run_ck_analysis(project_path, encoding="latin1"):
    """Run CK tool on the project and return metrics dataframes"""
    # output_dir = tempfile.mkdtemp()

    # Use absolute paths for Windows compatibility
    project_path = os.path.abspath(project_path)
    output_dir = os.path.abspath(OUTPUT_DIR)
    matrix_dir = os.path.join(MATRIX_DIR)

    # New CK command format that works with version 0.7.1
    command = [
        "java",
        "-Dfile.encoding=UTF-8",  # Add encoding parameter
        "-jar", CK_JAR_PATH,
        project_path,
        "true",
        "0",
        "false",
        output_dir
    ]

    print("Running CK command:", " ".join(command))

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    # Debugging output
    print(f"CK exit code: {result.returncode}")
    print(f"CK stdout: {result.stdout}")
    print(f"CK stderr: {result.stderr}")

    # Look for output files in the specified output directory
    class_csv = os.path.join(matrix_dir, "matrix_class.csv")
    method_csv = os.path.join(matrix_dir, "matrix_method.csv")

    def clean_csv(file_path):
        with open(file_path, 'rb') as f:
            content = f.read()
        # Remove null bytes
        content = content.replace(b'\x00', b'')
        with open(file_path, 'wb') as f:
            f.write(content)

    clean_csv(class_csv)
    clean_csv(method_csv)

    if os.path.exists(class_csv):
        class_metrics = pd.read_csv(class_csv, encoding=encoding, engine='python', on_bad_lines='skip')
    else:
        raise RuntimeError(
            f"Class CSV not generated at {class_csv}\n"
            f"Directory contents: {os.listdir(output_dir)}"
        )

    if os.path.exists(method_csv):
        method_metrics = pd.read_csv(method_csv, encoding=encoding, engine='python', on_bad_lines='skip')
    else:
        raise RuntimeError(
            f"Method CSV not generated at {method_csv}\n"
            f"Directory contents: {os.listdir(output_dir)}"
        )
    return class_metrics, method_metrics


def validate_java_project(project_dir):
    """Check if directory contains compilable Java files"""
    # Look for build configuration files
    build_files = [
        "pom.xml",
        "build.gradle",
        "build.xml",
        ".project",  # Eclipse project
        ".classpath"
    ]

    for file in build_files:
        if os.path.exists(os.path.join(project_dir, file)):
            return

    # If no build files, check for Java source structure
    java_src_dirs = [
        "src/main/java",
        "src",
        "source"
    ]

    for dir in java_src_dirs:
        if os.path.exists(os.path.join(project_dir, dir)):
            return

    # If no standard structure, look for any Java files
    java_files = []
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".java"):
                try:
                    with open(os.path.join(root, file), 'rb') as f:
                        f.read().decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        with open(os.path.join(root, file), 'rb') as f:
                            f.read().decode('latin1')
                    except:
                        print(f"Warning: File {file} has unsupported encoding")
                java_files.append(os.path.join(root, file))

    if not java_files:
        raise ValueError("No Java files found in the uploaded project")

    print(f"Found {len(java_files)} Java files in non-standard structure")


def safe_string(value):
    """Safely decode any string value"""
    if isinstance(value, str):
        return value
    try:
        if isinstance(value, bytes):
            return value.decode('utf-8', 'ignore')
        return str(value).encode('latin1', 'ignore').decode('latin1', 'ignore')
    except:
        return "unknown"


def safe_file_path(path):
    try:
        # Handle both str and bytes inputs
        if isinstance(path, bytes):
            return path.decode('utf-8', 'ignore')
        return str(path).encode('latin1', 'ignore').decode('latin1', 'ignore')
    except:
        return "invalid_file_path"


import pandas as pd
import shap

import math


def clean_json_float(value):
    """Convert NaN and infinite values to JSON-compatible numbers"""
    if isinstance(value, float):
        if math.isnan(value):
            return 0.0
        if math.isinf(value):
            return 1e10 if value > 0 else -1e10
    return value


import math


def clean_json_float(value):
    """Convert NaN and infinite values to JSON-compatible numbers"""
    if isinstance(value, float):
        if math.isnan(value):
            return 0.0
        if math.isinf(value):
            return 1e10 if value > 0 else -1e10
    return value


def detect_smells(class_metrics, method_metrics):
    """
    Run each trained smell‑detector on the CK metrics,
    returning report entries.
    """
    report = []

    # 1) sanitize file/class/method columns
    for df in (class_metrics, method_metrics):
        df['file'] = df['file'].apply(safe_string)
        for col in ('class', 'method'):
            if col in df.columns:
                df[col] = df[col].apply(safe_string)

    def process_models(metrics, models, level_type):
        for smell_type, pipe in models.items():
            # Get feature names from training
            scaler = pipe.steps[0][1]
            classifier = pipe.named_steps['clf']
            feat_names = scaler.feature_names_in_

            for idx, row in metrics.iterrows():
                try:
                    # Create DataFrame with proper feature names
                    feature_data = {}
                    for f in feat_names:
                        value = row.get(f, 0)  # Use 0 if feature missing
                        try:
                            # Ensure numeric value
                            num = float(value)
                        except (TypeError, ValueError):
                            num = 0.0
                        feature_data[f] = [clean_json_float(num)]

                    # Create DataFrame with one row and correct columns
                    X_raw = pd.DataFrame(feature_data, columns=feat_names)
                    X_pre = scaler.transform(X_raw)

                    # predict smell probability
                    proba = classifier.predict_proba(X_pre)[0][1]
                    if proba <= 0.5:
                        continue

                    # Generate SHAP explanation
                    shap_vals = None
                    try:
                        te = shap.TreeExplainer(classifier)
                        sv = te.shap_values(X_pre)
                        if isinstance(sv, list) and len(sv) == 2:
                            shap_vals = [clean_json_float(x) for x in sv[1][0]]
                        else:
                            shap_vals = [clean_json_float(x) for x in sv[0]]
                    except Exception:
                        def pred_fn(X: np.ndarray):
                            return classifier.predict_proba(scaler.transform(X))[:, 1]

                        me = shap.Explainer(pred_fn, X_raw)
                        sv = me(X_raw).values[0]
                        shap_vals = [clean_json_float(x) for x in sv]

                    # Prepare report entry
                    explanation = {
                        "features": feat_names.tolist(),
                        "values": [clean_json_float(x) for x in X_raw.iloc[0].values],
                        "shap_values": shap_vals
                    }
                    file_path = safe_string(row.get('file', 'unknown'))
                    entry = {
                        "file": file_path,
                        "smell_type": smell_type,
                        "explanation": explanation
                    }

                    if level_type == "class":
                        class_name = safe_string(row.get('class', ''))
                        entry.update({
                            "line": "-",
                            "description": f"Class '{class_name}' exhibits {smell_type}",
                            "potential_fix": "Refactor class to reduce complexity"
                        })
                    else:
                        method_name = safe_string(row.get('method', ''))
                        class_name = safe_string(row.get('class', ''))
                        line = str(row.get('startLine', '-'))
                        entry.update({
                            "line": line,
                            "description": f"Method '{method_name}' in class '{class_name}' exhibits {smell_type}",
                            "potential_fix": "Refactor method to improve quality"
                        })

                    report.append(entry)

                except Exception as e:
                    print(f"❌ Error processing {level_type}-level smell "
                          f"{smell_type} at row {idx}: {e}")
                    continue

    process_models(class_metrics, class_models, "class")
    process_models(method_metrics, method_models, "method")
    return report


@app.post("/analyze")
async def analyze_code(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Unzip project
        project_dir = os.path.join(temp_dir, "project")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(project_dir)

        # Validate project
        validate_java_project(project_dir)
        encoding = validate_encoding(project_dir)

        # Run CK analysis
        class_metrics, method_metrics = run_ck_analysis(project_dir, encoding)

        # Detect smells
        report = detect_smells(class_metrics, method_metrics)

        return JSONResponse(content={"report": report})

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def validate_encoding(project_dir):
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".java"):
                try:
                    with open(os.path.join(root, file), encoding='utf-8') as f:
                        f.read()
                except UnicodeDecodeError:
                    return "latin1"
    return "utf-8"


@app.post("/generate-pdf")
async def generate_pdf_report(report: dict):
    try:
        # This would generate a PDF using a library like ReportLab
        # For simplicity, we'll just return a dummy file
        pdf_path = "dummy_report.pdf"
        return FileResponse(pdf_path, media_type="application/pdf", filename="code_smell_report.pdf")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
