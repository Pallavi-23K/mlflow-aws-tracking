import sys
import logging

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet


# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow Tracking Server URI
import os

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI")
)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2


if __name__ == "__main__":

    print("NEW CODE RUNNING")

    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )

    try:
        data = pd.read_csv(csv_url, sep=";")

    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV. Error: %s",
            e
        )
        raise

    # Split Dataset
    train, test = train_test_split(
        data,
        random_state=42
    )

    train_x = train.drop(
        ["quality"],
        axis=1
    )

    test_x = test.drop(
        ["quality"],
        axis=1
    )

    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Hyperparameters
    alpha = (
        float(sys.argv[1])
        if len(sys.argv) > 1
        else 0.5
    )

    l1_ratio = (
        float(sys.argv[2])
        if len(sys.argv) > 2
        else 0.5
    )

    # Start MLflow Run
    with mlflow.start_run():

        lr = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=42
        )

        lr.fit(
            train_x,
            train_y
        )

        predicted_qualities = lr.predict(
            test_x
        )

        rmse, mae, r2 = eval_metrics(
            test_y,
            predicted_qualities
        )

        print(
            f"ElasticNet Model (alpha={alpha}, l1_ratio={l1_ratio})"
        )

        print(f"RMSE: {rmse}")
        print(f"MAE : {mae}")
        print(f"R2  : {r2}")

        # Log Parameters
        mlflow.log_param(
            "alpha",
            alpha
        )

        mlflow.log_param(
            "l1_ratio",
            l1_ratio
        )

        # Log Metrics
        mlflow.log_metric(
            "rmse",
            rmse
        )

        mlflow.log_metric(
            "mae",
            mae
        )

        mlflow.log_metric(
            "r2",
            r2
        )

         # Log Model Artifact
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="model"
        )

        print(
            "\nRun Logged Successfully to MLflow Server!"
        )