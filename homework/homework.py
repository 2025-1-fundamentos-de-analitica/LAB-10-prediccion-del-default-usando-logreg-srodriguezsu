import os
import json
import gzip
import pickle
import zipfile
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


def read_csv_from_zip(zip_path):
    """Extract and read a CSV file from a ZIP archive."""
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        csv_filename = zip_file.namelist()[0]
        with zip_file.open(csv_filename) as csv_file:
            return pd.read_csv(csv_file)


def preprocess_dataframe(data):
    """Clean and transform the raw data."""
    data.rename(columns={"default payment next month": "target"}, inplace=True)
    data.drop(columns=["ID"], inplace=True)
    data.dropna(inplace=True)

    data = data[(data["EDUCATION"] != 0) & (data["MARRIAGE"] != 0)]
    data["EDUCATION"] = data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return data


def build_feature_pipeline():
    """Create a preprocessing and classification pipeline."""
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric_cols = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
        "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
        "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    preprocessing = ColumnTransformer(transformers=[
        ("scale_numeric", MinMaxScaler(), numeric_cols),
        ("encode_categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    return pipeline


def optimize_model(pipeline, X, y):
    """Perform hyperparameter tuning using GridSearchCV."""
    search_space = {
        "feature_selection__k": range(1, 11),
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["liblinear"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=search_space,
        scoring="balanced_accuracy",
        cv=10,
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(X, y)
    return grid_search


def save_model_to_disk(model, path="../files/models/model.pkl.gz"):
    """Save trained model as a gzip-compressed pickle."""
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


def compute_metrics(model, X_train, y_train, X_test, y_test):
    """Calculate and return training and test set evaluation metrics."""
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    metrics_train = {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, preds_train)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, preds_train)),
        "recall": float(recall_score(y_train, preds_train)),
        "f1_score": float(f1_score(y_train, preds_train)),
    }

    metrics_test = {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, preds_test)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds_test)),
        "recall": float(recall_score(y_test, preds_test)),
        "f1_score": float(f1_score(y_test, preds_test)),
    }

    return metrics_train, metrics_test


def get_confusion_matrices(model, X_train, y_train, X_test, y_test):
    """Generate confusion matrices for train and test sets."""
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    cm_train = confusion_matrix(y_train, preds_train)
    cm_test = confusion_matrix(y_test, preds_test)

    return {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0][0]), "predicted_1": int(cm_train[0][1])},
        "true_1": {"predicted_0": int(cm_train[1][0]), "predicted_1": int(cm_train[1][1])},
    }, {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0][0]), "predicted_1": int(cm_test[0][1])},
        "true_1": {"predicted_0": int(cm_test[1][0]), "predicted_1": int(cm_test[1][1])},
    }


def save_json_lines(*records, output_path="../files/output/metrics.json"):
    """Save multiple records to a JSON Lines file."""
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def ensure_directories():
    os.makedirs("../files/models", exist_ok=True)
    os.makedirs("../files/output", exist_ok=True)


if __name__ == "__main__":
    ensure_directories()

    train_data = preprocess_dataframe(read_csv_from_zip("../files/input/train_data.csv.zip"))
    test_data = preprocess_dataframe(read_csv_from_zip("../files/input/test_data.csv.zip"))

    X_train, y_train = train_data.drop(columns=["target"]), train_data["target"]
    X_test, y_test = test_data.drop(columns=["target"]), test_data["target"]

    pipeline = build_feature_pipeline()
    tuned_model = optimize_model(pipeline, X_train, y_train)

    save_model_to_disk(tuned_model)

    metrics_train, metrics_test = compute_metrics(tuned_model, X_train, y_train, X_test, y_test)
    cm_train, cm_test = get_confusion_matrices(tuned_model, X_train, y_train, X_test, y_test)

    save_json_lines(metrics_train, metrics_test, cm_train, cm_test)
