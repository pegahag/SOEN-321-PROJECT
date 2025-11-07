"""
evasion_attack.py

Run evasion-style perturbations on a saved classifier+scaler and measure
how accurate results are and the classification report changes
"""
import os
import itertools
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
TRAIN_PKL = os.path.join(DATA_DIR, "wdbc_binaryDiagnosis_normalized_training.pkl")
TEST_PKL = os.path.join(DATA_DIR, "wdbc_binaryDiagnosis_normalized_testing.pkl")

MODEL_PATH = os.path.join(BASE_DIR, "best_logistic_model_lr1.joblib")

OUT_DIR = os.path.join(BASE_DIR, "evasion_results_output")
os.makedirs(os.path.join(OUT_DIR, "evasion_details"), exist_ok=True)
SEED = 42

np.random.seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "evasion_details"), exist_ok=True)

data_train = pd.read_pickle(TRAIN_PKL)
data_test = pd.read_pickle(TEST_PKL)

X_train = data_train.iloc[:, :-1].reset_index(drop=True)
y_train = data_train.iloc[:, -1].astype(int).reset_index(drop=True)

X_test = data_test.iloc[:, :-1].reset_index(drop=True)
y_test = data_test.iloc[:, -1].astype(int).reset_index(drop=True)

n_features = X_test.shape[1]
feature_names = list(X_test.columns)

saved = joblib.load(MODEL_PATH)
if isinstance(saved, dict) and 'model' in saved:
    clf = saved['model']
    scaler = saved.get('scaler', None)
else:
    clf = saved
    scaler = None

have_scaler = scaler is not None and hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_')

if have_scaler:
    feature_std = np.array(scaler.scale_)
    feature_mean = np.array(scaler.mean_)
    X_test_orig = pd.DataFrame(scaler.inverse_transform(X_test), columns=feature_names)
    X_train_orig = pd.DataFrame(scaler.inverse_transform(X_train), columns=feature_names)
else:
    X_test_orig = X_test.copy()
    X_train_orig = X_train.copy()
    feature_std = np.array(X_test_orig.std(axis=0, ddof=0))
    feature_mean = np.array(X_test_orig.mean(axis=0))

magnitudes = {
    'minute': 0.1,
    'medium': 0.5,
    'large': 1.0
}

results = []
def evaluate_and_record(X_test_pert_scaled, scenario_name, details):
    """Predict using clf and compute metrics, save confusion and report."""
    y_pred = clf.predict(X_test_pert_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=False, zero_division=0)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    base = os.path.join(OUT_DIR, "evasion_details", f"{timestamp}_{scenario_name}")

    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(base + "_confusion_matrix.csv", index=False)

    with open(base + "_class_report.txt", "w") as f:
        f.write(cr)

    pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    }).to_csv(base + "_preds.csv", index=False)


    rec = {
        'scenario': scenario_name,
        'accuracy': float(acc),
        'n_features_affected': details.get('n_features_affected', None),
        'features_affected': ";".join(map(str, details.get('features_affected', []))),
        'direction': details.get('direction', None),
        'magnitude_label': details.get('magnitude_label', None),
        'magnitude_value': float(details.get('magnitude_value', np.nan)),
        'notes': details.get('notes', "")
    }
    results.append(rec)
    print(f"Scenario '{scenario_name}': acc={acc:.4f}  affected={rec['features_affected']} magnitude={rec['magnitude_label']} dir={rec['direction']}")


def scale_perturbed(X_orig_df):
    if have_scaler:
        X_scaled = scaler.transform(X_orig_df)
    else:
        X_scaled = X_orig_df.values

    return pd.DataFrame(X_scaled, columns=feature_names)


X_test_scaled = scale_perturbed(X_test_orig)
evaluate_and_record(X_test_scaled, "baseline_no_perturb", {
    'n_features_affected': 0,
    'features_affected': [],
    'direction': None,
    'magnitude_label': None,
    'magnitude_value': 0.0,
    'notes': 'Baseline (no perturbation)'
})


for feat_idx in range(n_features):
    feat_name = feature_names[feat_idx]
    std_i = feature_std[feat_idx] if feature_std[feat_idx] != 0 else 1.0
    for mag_label, mag_mul in magnitudes.items():
        mag_value = mag_mul * std_i
        for direction in ['pos', 'neg']:
            X_pert = X_test_orig.copy()
            if direction == 'pos':
                X_pert.iloc[:, feat_idx] = X_pert.iloc[:, feat_idx] + mag_value
            else:
                X_pert.iloc[:, feat_idx] = X_pert.iloc[:, feat_idx] - mag_value

            X_pert_scaled = scale_perturbed(X_pert)
            scenario_name = f"feature_{feat_idx}_{feat_name}_{mag_label}_{direction}"
            evaluate_and_record(X_pert_scaled, scenario_name, {
                'n_features_affected': 1,
                'features_affected': [feat_name],
                'direction': direction,
                'magnitude_label': mag_label,
                'magnitude_value': mag_value,
                'notes': f"Perturb single feature {feat_name} by {mag_value:.6g} ({mag_label}) in {direction} direction"
            })


n_2_3 = max(1, int(np.floor(2 * n_features / 3.0)))
all_feature_indices = list(range(n_features))


rng = np.random.RandomState(SEED)
selected_2_3_idx = rng.choice(all_feature_indices, size=n_2_3, replace=False)
selected_2_3_names = [feature_names[i] for i in selected_2_3_idx]


for mag_label, mag_mul in magnitudes.items():
    mag_vector = mag_mul * feature_std
    for direction in ['pos', 'neg']:
        X_pert = X_test_orig.copy()
        for idx in selected_2_3_idx:
            if direction == 'pos':
                X_pert.iloc[:, idx] = X_pert.iloc[:, idx] + mag_vector[idx]
            else:
                X_pert.iloc[:, idx] = X_pert.iloc[:, idx] - mag_vector[idx]

        X_pert_scaled = scale_perturbed(X_pert)
        scenario_name = f"multi_2of3_{mag_label}_{direction}"
        evaluate_and_record(X_pert_scaled, scenario_name, {
            'n_features_affected': len(selected_2_3_idx),
            'features_affected': selected_2_3_names,
            'direction': direction,
            'magnitude_label': mag_label,
            'magnitude_value': float(np.mean(mag_vector[selected_2_3_idx])),
            'notes': f"Perturb {len(selected_2_3_idx)} (2/3) features: {selected_2_3_names}"
        })

for mag_label, mag_mul in magnitudes.items():
    mag_vector = mag_mul * feature_std
    for direction in ['pos', 'neg']:
        X_pert = X_test_orig.copy()
        if direction == 'pos':
            X_pert = X_pert + mag_vector
        else:
            X_pert = X_pert - mag_vector

        X_pert_scaled = scale_perturbed(X_pert)
        scenario_name = f"multi_all_{mag_label}_{direction}"
        evaluate_and_record(X_pert_scaled, scenario_name, {
            'n_features_affected': n_features,
            'features_affected': feature_names,
            'direction': direction,
            'magnitude_label': mag_label,
            'magnitude_value': float(np.mean(mag_vector)),
            'notes': f"Perturb all features by {mag_label}"
        })

summary_df = pd.DataFrame(results)
summary_csv = os.path.join(OUT_DIR, "evasion_results_summary.csv")
summary_df.to_csv(summary_csv, index=False)

with open(os.path.join(OUT_DIR, "evasion_results_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Summary saved to:", summary_csv)
print("Detailed per-scenario outputs are in the evasion_details folder inside the output directory.")
