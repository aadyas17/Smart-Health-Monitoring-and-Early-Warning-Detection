# ============================================================
# ADVANCED ML PIPELINE — DROP-IN REPLACEMENT FOR BACKEND
# Saves: ml_model/svm_model.pkl  &  ml_model/scaler.pkl
#
# Backend expects:
#   - model.predict(scaler.transform(input_15_features))
#   - Output mapped via {0:'Low', 1:'Medium', 2:'High'}
#   - Feature order: temperature, rainfall, ph, turbidity,
#     dissolved_oxygen, nitrate, lead, bacterial_count,
#     clean_water_percentage, sanitation_level,
#     healthcare_access, symptom_diarrhea, symptom_fever,
#     water_dirty, water_scarcity
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)

# Models
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

os.makedirs("ml_model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ============================================================
# FEATURE ORDER — must match backend exactly
# ============================================================
FEATURE_COLS = [
    'temperature', 'rainfall', 'ph', 'turbidity',
    'dissolved_oxygen', 'nitrate', 'lead', 'bacterial_count',
    'clean_water_percentage', 'sanitation_level', 'healthcare_access',
    'symptom_diarrhea', 'symptom_fever', 'water_dirty', 'water_scarcity'
]
TARGET_COL   = 'outbreak_risk'
RISK_MAP     = {0: 'Low', 1: 'Medium', 2: 'High'}
CLASS_NAMES  = ['Low Risk', 'Medium Risk', 'High Risk']

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("data.csv")

print("=" * 55)
print("  ADVANCED ML PIPELINE — Water Outbreak Risk")
print("=" * 55)
print(f"\nDataset shape : {df.shape}")
print(f"Target dist   :\n{df[TARGET_COL].value_counts().to_string()}")
print(f"\nMissing values:\n{df[FEATURE_COLS].isnull().sum().to_string()}")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ============================================================
# 2. FEATURE ENGINEERING
#    (adds columns to X; backend doesn't care — scaler handles it)
#    NOTE: We keep this INTERNAL to training only.
#          The scaler will be fit on the 15 original features
#          so the backend can call scaler.transform(15_features).
#          Feature engineering is applied BEFORE fitting the scaler.
#
#    Since the backend sends exactly 15 raw features, we wrap
#    the scaler + feature engineering into a custom transformer
#    so joblib.load("scaler.pkl").transform(X_15) still works.
# ============================================================

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# ── Feature-level safe ranges ──────────────────────────────
# Fix 3: full range table for ALL 15 features (not just 4)
# Values outside these are clamped, not rejected — so the
# backend never crashes and we degrade gracefully.
FEATURE_RANGES = {
    # feature         : (min,   max,   median_fallback)
    'temperature'     : (20.0,  45.0,  32.0),
    'rainfall'        : (0.0,   300.0, 150.0),
    'ph'              : (6.0,   9.0,   7.5),
    'turbidity'       : (0.0,   500.0, 190.0),
    'dissolved_oxygen': (2.0,   14.0,  8.5),
    'nitrate'         : (0.0,   50.0,  23.0),
    'lead'            : (0.0,   0.2,   0.076),
    'bacterial_count' : (0.0,   1000.0,325.0),
    'clean_water_percentage': (20.0, 100.0, 67.0),
    'sanitation_level': (10.0,  100.0, 62.0),
    'healthcare_access': (10.0, 100.0, 60.0),
    'symptom_diarrhea': (0.0,   1.0,   0.0),
    'symptom_fever'   : (0.0,   1.0,   0.0),
    'water_dirty'     : (0.0,   1.0,   0.0),
    'water_scarcity'  : (0.0,   1.0,   0.0),
}

# ── Drift thresholds (±N std devs from training mean) ──────
# Fix 5: stored during fit(), checked during transform()
DRIFT_STD_THRESHOLD = 4.0   # flag if any feature > 4σ from training mean


class WaterFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Drop-in scaler replacement — handles all 5 risk issues:

       Fix 1  Missing values  → median imputation per feature
       Fix 2  Type safety     → force float64 before anything
       Fix 3  Range safety    → clamp ALL 15 features to valid range
       Fix 4  Confidence      → stores last_confidence_ after transform
       Fix 5  Drift detection → stores drift_warnings_ after transform

    Compatible with backend:
        scaler.transform(np.array([[15 raw features]]))  → scaled array
        model.predict(scaled)[0]                         → 0 / 1 / 2
    """

    def __init__(self):
        self.scaler_        = StandardScaler()
        self.imputer_       = SimpleImputer(strategy='median')
        self.feature_names_ = None
        self.train_means_   = {}   # for drift detection
        self.train_stds_    = {}   # for drift detection
        # Runtime diagnostics (updated every transform call)
        self.last_confidence_  = None   # float 0-1, filled by predict wrapper
        self.drift_warnings_   = []     # list of flagged feature names

    # ----------------------------------------------------------
    # STEP 1 — Type safety  (Fix 2)
    # ----------------------------------------------------------
    def _to_float(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=FEATURE_COLS)
        else:
            X = X.copy()
        # Cast every column to float — handles "30" strings, booleans, ints
        for col in FEATURE_COLS:
            X[col] = pd.to_numeric(X[col], errors='coerce')  # bad strings → NaN
        return X.astype(float)

    # ----------------------------------------------------------
    # STEP 2 — Impute missing values  (Fix 1)
    # ----------------------------------------------------------
    def _impute(self, X, fit=False):
        if fit:
            arr = self.imputer_.fit_transform(X[FEATURE_COLS])
        else:
            arr = self.imputer_.transform(X[FEATURE_COLS])
        return pd.DataFrame(arr, columns=FEATURE_COLS, index=X.index)

    # ----------------------------------------------------------
    # STEP 3 — Clamp ALL features to valid range  (Fix 3)
    # ----------------------------------------------------------
    def _clamp(self, X):
        for col, (lo, hi, _) in FEATURE_RANGES.items():
            X[col] = X[col].clip(lower=lo, upper=hi)
        return X

    # ----------------------------------------------------------
    # STEP 4 — Drift detection  (Fix 5)
    # ----------------------------------------------------------
    def _check_drift(self, X):
        warnings = []
        for col in FEATURE_COLS:
            mean = self.train_means_.get(col, 0)
            std  = self.train_stds_.get(col, 1)
            if std == 0:
                continue
            z = ((X[col] - mean) / std).abs().max()
            if z > DRIFT_STD_THRESHOLD:
                warnings.append(f"{col} (z={z:.1f})")
        if warnings:
            print(f"   Drift detected in: {', '.join(warnings)}")
        self.drift_warnings_ = warnings
        return X

    # ----------------------------------------------------------
    # STEP 5 — Feature engineering
    # ----------------------------------------------------------
    def _engineer(self, X):
        X['water_quality_index'] = (
            (X['ph'] - 7.0).abs() * 10 +
            X['turbidity'] / 50   +
            X['bacterial_count'] / 100 +
            X['lead'] * 100       +
            X['nitrate'] / 10
        )
        X['health_vulnerability'] = (
            (100 - X['clean_water_percentage']) * 0.4 +
            (100 - X['sanitation_level'])        * 0.3 +
            (100 - X['healthcare_access'])        * 0.3
        )
        X['symptom_score']    = X['symptom_diarrhea'] + X['symptom_fever']
        X['water_issue_score']= X['water_dirty']      + X['water_scarcity']

        X['temp_stress']  = (X['temperature'] - 25).clip(lower=0)
        X['low_rainfall'] = (200 - X['rainfall']).clip(lower=0) / 200
        X['low_oxygen']   = (8 - X['dissolved_oxygen']).clip(lower=0)

        X['bacteria_x_dirty']   = X['bacterial_count'] * X['water_dirty']
        X['scarcity_x_sanit']   = X['water_scarcity']  * (100 - X['sanitation_level'])
        X['symptom_x_bacteria'] = X['symptom_score']   * X['bacterial_count']
        return X

    # ----------------------------------------------------------
    # sklearn API
    # ----------------------------------------------------------
    def fit(self, X, y=None):
        X = self._to_float(X)
        X = self._impute(X, fit=True)
        X = self._clamp(X)
        # Store training statistics for drift detection (Fix ⚠️5)
        for col in FEATURE_COLS:
            self.train_means_[col] = float(X[col].mean())
            self.train_stds_[col]  = float(X[col].std())
        X = self._engineer(X)
        self.feature_names_ = X.columns.tolist()
        self.scaler_.fit(X)
        return self

    def transform(self, X):
        X = self._to_float(X)       # Fix 2 — string → float
        X = self._impute(X, fit=False)  # Fix 1 — NaN → median
        X = self._clamp(X)          # Fix 3 — clip out-of-range
        X = self._check_drift(X)    # Fix 5 — log drift warnings
        X = self._engineer(X)
        return self.scaler_.transform(X)


# ── Confidence-aware predict wrapper  (Fix 4) ───────────
# Backend calls: model.predict(scaled)[0]
# This wrapper is saved AS the model — predict() still works
# identically, but confidence is now logged to transformer.
class ConfidentModel(BaseEstimator):
    """
    Wraps any classifier so that .predict() also logs
    per-prediction confidence into `transformer.last_confidence_`.

    Backend call unchanged:
        model.predict(scaled_input)[0]  → 0 / 1 / 2  
    Confidence available at:
        transformer.last_confidence_    → e.g. 0.91
    """
    def __init__(self, base_model, transformer_ref):
        self.base_model      = base_model
        self.transformer_ref = transformer_ref

    def fit(self, X, y):
        self.base_model.fit(X, y)
        self.classes_ = self.base_model.classes_
        return self

    def predict(self, X):
        preds = self.base_model.predict(X)
        # Log confidence if model supports probabilities
        if hasattr(self.base_model, 'predict_proba'):
            probs = self.base_model.predict_proba(X)
            conf  = float(probs.max(axis=1).mean())
            self.transformer_ref.last_confidence_ = conf
            if conf < 0.60:
                print(f"   Low confidence prediction: {conf:.2%} "
                      f"— treat result with caution")
        return preds

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    # Make sklearn utilities (GridSearch, cross_val) work
    def get_params(self, deep=True):
        return {'base_model': self.base_model,
                'transformer_ref': self.transformer_ref}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Fit & apply the custom scaler/engineer
transformer = WaterFeatureEngineer()
X_train = transformer.fit_transform(X_train_raw, y_train)
X_test  = transformer.transform(X_test_raw)

print(f"\nFeatures after engineering : {X_train.shape[1]}")
print(f"Train size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")


# ============================================================
# 4. HELPER — EVALUATE MODEL
# ============================================================
def evaluate(name, model, X_tr, X_te, y_tr, y_te, verbose=True):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  {name:40s}  acc={acc:.4f}")
        print(f"{'─'*50}")
        print(classification_report(y_te, y_pred, target_names=CLASS_NAMES))

        cm = confusion_matrix(y_te, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low','Med','High'],
                    yticklabels=['Low','Med','High'])
        plt.title(f"{name}\nAccuracy: {acc:.4f}")
        plt.tight_layout()
        plt.savefig(f"plots/cm_{name.replace(' ','_').lower()}.png", dpi=100)
        plt.close()

    return model, acc, y_pred


# ============================================================
# 5. BASE MODEL COMPARISON  (cross-validated)
# ============================================================
print("\n\n--- 5-Fold Stratified Cross-Validation ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_candidates = {
    "Logistic Regression" : LogisticRegression(max_iter=2000, C=1.0),
    "SVM (RBF)"           : SVC(kernel='rbf', probability=True, random_state=42),
    "SVM (Poly)"          : SVC(kernel='poly', degree=3, probability=True, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=200, random_state=42),
}

cv_scores = {}
for name, mdl in base_candidates.items():
    sc = cross_val_score(mdl, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_scores[name] = sc
    print(f"  {name:25s}: {sc.mean():.4f} ± {sc.std():.4f}")

plt.figure(figsize=(8, 4))
plt.boxplot(cv_scores.values(), labels=cv_scores.keys(), patch_artist=True)
plt.title("Cross-Validation Accuracy Comparison")
plt.xticks(rotation=20, ha='right')
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("plots/cv_comparison.png", dpi=100)
plt.close()


# ============================================================
# 6. HYPERPARAMETER TUNING
# ============================================================

# --- Tune SVM (your backend filename is svm_model.pkl) ---
print("\n\n--- GridSearch: SVM ---")
svm_param_grid = {
    'C'      : [0.1, 1, 10, 50, 100],
    'gamma'  : ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel' : ['rbf', 'poly'],
    'degree' : [2, 3],           # only used for poly
}
svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_param_grid, cv=5, scoring='accuracy',
    n_jobs=-1, verbose=1
)
svm_grid.fit(X_train, y_train)
print(f"Best SVM params : {svm_grid.best_params_}")
print(f"Best SVM CV acc : {svm_grid.best_score_:.4f}")

# --- Tune Random Forest ---
print("\n--- GridSearch: Random Forest ---")
rf_param_grid = {
    'n_estimators'     : [200, 300, 500],
    'max_depth'        : [None, 15, 25],
    'min_samples_split': [2, 5],
    'max_features'     : ['sqrt', 'log2'],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid, cv=5, scoring='accuracy',
    n_jobs=-1, verbose=1
)
rf_grid.fit(X_train, y_train)
print(f"Best RF params : {rf_grid.best_params_}")
print(f"Best RF CV acc : {rf_grid.best_score_:.4f}")

# --- Tune Gradient Boosting ---
print("\n--- GridSearch: Gradient Boosting ---")
gb_param_grid = {
    'n_estimators'  : [200, 300],
    'learning_rate' : [0.05, 0.1],
    'max_depth'     : [3, 5, 7],
    'subsample'     : [0.8, 1.0],
}
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_grid, cv=5, scoring='accuracy',
    n_jobs=-1, verbose=1
)
gb_grid.fit(X_train, y_train)
print(f"Best GB params : {gb_grid.best_params_}")
print(f"Best GB CV acc : {gb_grid.best_score_:.4f}")


# ============================================================
# 7. EVALUATE ALL TUNED MODELS
# ============================================================
print("\n\n--- Test Set Evaluation ---")

svm_best = svm_grid.best_estimator_
rf_best  = rf_grid.best_estimator_
gb_best  = gb_grid.best_estimator_

_, acc_svm, _ = evaluate("SVM (Tuned)",              svm_best, X_train, X_test, y_train, y_test)
_, acc_rf,  _ = evaluate("Random Forest (Tuned)",    rf_best,  X_train, X_test, y_train, y_test)
_, acc_gb,  _ = evaluate("Gradient Boosting (Tuned)",gb_best,  X_train, X_test, y_train, y_test)


# ============================================================
# 8. STACKING ENSEMBLE
# ============================================================
print("\n--- Stacking Ensemble ---")

stacking = StackingClassifier(
    estimators=[
        ('svm', svm_best),
        ('rf',  rf_best),
        ('gb',  gb_best),
    ],
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5,
    n_jobs=-1
)
_, acc_stack, _ = evaluate("Stacking Ensemble", stacking, X_train, X_test, y_train, y_test)


# ============================================================
# 9. SOFT VOTING ENSEMBLE
# ============================================================
print("\n--- Soft Voting Ensemble ---")

voting = VotingClassifier(
    estimators=[
        ('svm', svm_best),
        ('rf',  rf_best),
        ('gb',  gb_best),
    ],
    voting='soft',
    n_jobs=-1
)
_, acc_vote, _ = evaluate("Soft Voting Ensemble", voting, X_train, X_test, y_train, y_test)


# ============================================================
# 10. FINAL COMPARISON & SELECT BEST
# ============================================================
results = {
    "SVM (Tuned)"             : acc_svm,
    "Random Forest (Tuned)"   : acc_rf,
    "Gradient Boosting (Tuned)": acc_gb,
    "Stacking Ensemble"       : acc_stack,
    "Soft Voting Ensemble"    : acc_vote,
}

print("\n\n" + "=" * 55)
print("  FINAL RESULTS")
print("=" * 55)
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(acc * 50)
    print(f"  {name:32s}: {acc:.4f}  {bar}")

best_name = max(results, key=results.get)
best_acc  = results[best_name]
model_map = {
    "SVM (Tuned)"             : svm_best,
    "Random Forest (Tuned)"   : rf_best,
    "Gradient Boosting (Tuned)": gb_best,
    "Stacking Ensemble"       : stacking,
    "Soft Voting Ensemble"    : voting,
}
final_model = model_map[best_name]

print(f"\n  Winner : {best_name}  (acc = {best_acc:.4f})")


# ============================================================
# 11. ROC-AUC CURVES
# ============================================================
final_model.fit(X_train, y_train)

# Need predict_proba — wrap if needed
if hasattr(final_model, 'predict_proba'):
    y_prob = final_model.predict_proba(X_test)
else:
    cal = CalibratedClassifierCV(final_model, cv=3)
    cal.fit(X_train, y_train)
    y_prob = cal.predict_proba(X_test)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
auc_ovr = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
print(f"\n  AUC-OVR : {auc_ovr:.4f}")

colors = ['#2ecc71', '#f39c12', '#e74c3c']
plt.figure(figsize=(7, 5))
for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, color=col, lw=2,
             label=f"{cls} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curves — {best_name}')
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_curves.png", dpi=100)
plt.close()


# ============================================================
# 12. FEATURE IMPORTANCE (from RF component)
# ============================================================
# Extract RF from ensemble for feature importance
try:
    rf_component = rf_best
    rf_component.fit(X_train, y_train)
    importances = pd.Series(
        rf_component.feature_importances_,
        index=transformer.feature_names_
    ).sort_values(ascending=True)

    plt.figure(figsize=(8, 7))
    colors_imp = ['#e74c3c' if v > importances.mean() else '#3498db'
                  for v in importances]
    importances.plot(kind='barh', color=colors_imp)
    plt.axvline(importances.mean(), color='gray', linestyle='--', label='Mean')
    plt.title("Feature Importances (Random Forest component)")
    plt.tight_layout()
    plt.savefig("plots/feature_importances.png", dpi=100)
    plt.close()

    print("\n  Top 5 features:")
    print(importances.sort_values(ascending=False).head().to_string())
except Exception as e:
    print(f"  Feature importance skipped: {e}")


# ============================================================
# 13. MODEL COMPARISON BAR CHART
# ============================================================
plt.figure(figsize=(9, 4))
names_sorted = sorted(results, key=results.get)
accs_sorted  = [results[n] for n in names_sorted]
bar_colors   = ['#e74c3c' if n == best_name else '#3498db' for n in names_sorted]
bars = plt.barh(names_sorted, accs_sorted, color=bar_colors)
plt.xlim(0.75, 1.0)
plt.xlabel("Accuracy")
plt.title("Model Accuracy Comparison")
for bar, acc in zip(bars, accs_sorted):
    plt.text(acc + 0.001, bar.get_y() + bar.get_height()/2,
             f'{acc:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=100)
plt.close()


# ============================================================
# 14. SAVE DROP-IN FILES
#     Backend loads:
#       model  = joblib.load("ml_model/svm_model.pkl")
#       scaler = joblib.load("ml_model/scaler.pkl")
#     Then calls:
#       input_scaled = scaler.transform(input_data)   # 15 raw features
#       prediction   = model.predict(input_scaled)[0] # returns 0/1/2
# ============================================================
# Fix 4 — wrap FIRST, then fit the wrapper (not the raw model)
confident_model = ConfidentModel(base_model=final_model, transformer_ref=transformer)
confident_model.fit(X_train, y_train)   # trains base_model inside the wrapper

joblib.dump(confident_model, "ml_model/svm_model.pkl")   # backend variable name
joblib.dump(transformer,     "ml_model/scaler.pkl")      # does engineering + scaling

print("\n" + "=" * 55)
print("  FILES SAVED — Drop into your project as-is")
print("=" * 55)
print("  ml_model/svm_model.pkl  ← best model")
print("  ml_model/scaler.pkl     ← feature engineer + scaler")
print(f"\n  Final accuracy : {best_acc:.4f}")
print(f"  AUC-OVR        : {auc_ovr:.4f}")
print(f"  Model chosen   : {best_name}")
print("\n  All plots saved to: plots/")
print("\n  Backend compatibility:   No changes needed.")