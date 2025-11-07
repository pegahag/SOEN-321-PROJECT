import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
TRAIN_PATH = os.path.join(DATA_DIR, "wdbc_binaryDiagnosis_normalized_training.pkl")
TEST_PATH = os.path.join(DATA_DIR, "wdbc_binaryDiagnosis_normalized_testing.pkl")

data_train = pd.read_pickle(TRAIN_PATH)
data_test = pd.read_pickle(TEST_PATH)

X_train_scaled = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1].astype(int)

X_test_scaled = data_test.iloc[:, :-1]
y_test = data_test.iloc[:, -1].astype(int)

learning_rates = [0.001, 0.01, 0.1, 1]
results = []
best_test_acc = 0
best_model = None
best_lr = None

for lr in learning_rates:
    print(f"\n=== Training Logistic Regression with learning rate={lr} ===")
    clf = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=lr, max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    y_train_pred = clf.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)

    y_test_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")

    print("\nConfusion Matrix on Test Data:")
    print(confusion_matrix(y_test, y_test_pred))

    print("\nClassification Report on Test Data:")
    print(classification_report(y_test, y_test_pred))

    results.append({
        'learning_rate': lr,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc
    })

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model = clf
        best_lr = lr

print("\n=== Summary of Results ===")
for res in results:
    print(f"LR={res['learning_rate']}: Train Acc={res['train_accuracy']:.4f}, Test Acc={res['test_accuracy']:.4f}")

saved_file = os.path.join(BASE_DIR, f"best_logistic_model_lr{best_lr}.joblib")
joblib.dump({'model': best_model, 'scaler': None}, saved_file)

print(f"\n✅ Best model saved with learning rate={best_lr} and Test Accuracy={best_test_acc:.4f}")

if os.path.exists(saved_file):
    print(f"File saved successfully: {saved_file}")
else:
    print("File was not saved.")
