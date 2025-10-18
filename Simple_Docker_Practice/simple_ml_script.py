from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200, C=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {acc:.4f}")

#Save Predictions to CSV
predictions_df = pd.DataFrame({"true": y_test, "predicted": y_pred})
predictions_df.to_csv("predictions.csv", index=False)
print("Predictions saved to 'predictions.csv'")

# Save the model
joblib.dump(model, "simple_logreg_model.joblib")
print("Model saved to 'simple_logreg_model.joblib'")
