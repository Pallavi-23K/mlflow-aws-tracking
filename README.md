<div align="center">

# 🚀 MLflow Tracking Server on AWS EC2

### End-to-End MLOps Project using MLflow, AWS EC2, Amazon S3, and Scikit-Learn

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?style=for-the-badge\&logo=mlflow)
![AWS](https://img.shields.io/badge/AWS-Cloud-yellow?style=for-the-badge\&logo=amazonaws)
![S3](https://img.shields.io/badge/Amazon-S3-red?style=for-the-badge\&logo=amazons3)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-f7931e?style=for-the-badge\&logo=scikitlearn)

</div>

---

## 📖 Project Overview

This project demonstrates the deployment of a **remote MLflow Tracking Server on AWS EC2** with **Amazon S3** as the artifact store. A Scikit-Learn ElasticNet model is trained and tracked remotely, enabling experiment management, model versioning, and artifact storage through MLflow.

### ✨ Key Features

* 📊 Experiment Tracking
* 📈 Metrics Logging
* ⚙️ Hyperparameter Logging
* 🤖 Model Versioning
* ☁️ AWS EC2 Deployment
* 🪣 Amazon S3 Artifact Storage
* 🔄 Reproducible ML Workflows

---

## 🏗️ System Architecture

```text
+-----------------------+
| Local Training Script |
+-----------+-----------+
            |
            | Logs Parameters,
            | Metrics & Models
            v
+-----------------------+
| MLflow Tracking Server|
|      AWS EC2          |
+-----------+-----------+
            |
            | Stores Artifacts
            v
+-----------------------+
|      Amazon S3        |
|   Artifact Storage    |
+-----------------------+
```

---

## 🛠️ Tech Stack

| Technology   | Purpose               |
| ------------ | --------------------- |
| Python       | Model Development     |
| Scikit-Learn | Machine Learning      |
| MLflow       | Experiment Tracking   |
| AWS EC2      | Hosting MLflow Server |
| Amazon S3    | Artifact Storage      |
| IAM          | Access Management     |
| Pandas       | Data Processing       |
| NumPy        | Numerical Computing   |

---

## 📂 Project Structure

```text
MLflow-AWS/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore

```

---

## 🚀 AWS Deployment Steps

### 1️⃣ Create IAM User

* Login to AWS Console
* Create IAM User
* Attach AdministratorAccess Policy
* Configure AWS CLI

```bash
aws configure
```

---

### 2️⃣ Create S3 Bucket

```text
mlflowtracking1
```

Used for storing:

* Models
* Artifacts
* Experiment Files

---

### 3️⃣ Launch EC2 Instance

Recommended Configuration:

| Setting       | Value               |
| ------------- | ------------------- |
| AMI           | Ubuntu              |
| Instance Type | t2.micro / t3.micro |
| Storage       | 8 GB                |
| Open Ports    | 22, 5000            |

---

## ⚙️ Install Dependencies on EC2

```bash
sudo apt update

sudo apt install python3-pip -y
sudo apt install pipenv -y
sudo apt install virtualenv -y

mkdir mlflow
cd mlflow

pipenv install mlflow
pipenv install awscli
pipenv install boto3

pipenv shell
```

---

## ▶️ Start MLflow Server

```bash
mlflow server \
-h 0.0.0.0 \
--default-artifact-root s3://mlflowtracking1
```

---

## 🌐 Open MLflow Dashboard

```text
http://<EC2-PUBLIC-IP>:5000
```

---

## 📊 Model Training

The project trains an ElasticNet Regression model on the Wine Quality dataset.

### Logged Parameters

```text
alpha
l1_ratio
```

### Logged Metrics

```text
RMSE
MAE
R² Score
```

### Logged Artifacts

```text
Trained Model
Model Metadata
Experiment Runs
```


