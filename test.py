import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


# 1. Load and prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 2. Set up MLflow experiment
mlflow.set_experiment("Iris_Classification_Experiment")

# 3. Train model with MLflow tracking
with mlflow.start_run(run_name="LogReg_v1"):
    C_value = 1.0  # Regularization strength
    model = LogisticRegression(max_iter=200, C=C_value)
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log parameters and metrics
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", C_value)
    mlflow.log_metric("accuracy", acc)

    # Log the model
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"✅ Run complete — accuracy: {acc:.4f}")

# 4. (Optional) Save model locally for DVC tracking
os.makedirs("models", exist_ok=True)
mlflow.sklearn.save_model(model, "models/iris_logreg_model")

print("\nModel saved to 'models/iris_logreg_model'")
print("You can now version it with DVC if you like:")
print("  $ dvc add models/iris_logreg_model")