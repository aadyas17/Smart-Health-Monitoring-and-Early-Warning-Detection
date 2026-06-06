# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Original models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Improved / additional models
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,   # NEW: boosted trees - usually stronger than RF
    ExtraTreesClassifier,          # NEW: extremely randomized trees
    VotingClassifier,              # NEW: combines multiple models
    StackingClassifier,            # NEW: meta-learner stacking ensemble
)
from sklearn.neural_network import MLPClassifier  # NEW: deep neural network


# ================================
# 2. LOAD DATASET
# ================================
df = pd.read_csv("waterborne_disease_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())


# ================================
# 3. DATA ANALYSIS
# ================================
print("\nDataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

print("\nClass Distribution:\n")
print(df['outbreak_risk'].value_counts())


# ================================
# 4. EDA VISUALIZATIONS
# ================================

# Distribution of target
plt.figure()
sns.countplot(x='outbreak_risk', data=df)
plt.title("Outbreak Risk Distribution")
plt.savefig("target_distribution.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# Example feature distributions
features = ['temperature', 'ph', 'turbidity', 'bacterial_count']

for f in features:
    plt.figure()
    sns.histplot(df[f], kde=True)
    plt.title(f"Distribution of {f}")
    plt.savefig(f"{f}_distribution.png")
    plt.close()


# ================================
# 5. PREPROCESSING
# ================================
X = df.drop("outbreak_risk", axis=1)
y = df["outbreak_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify keeps class balance
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")


# ================================
# 6. MODEL 1: LOGISTIC REGRESSION
# ================================
lr = LogisticRegression(max_iter=1000, C=0.5, solver='lbfgs', multi_class='auto')
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print("\nLogistic Regression Accuracy:", acc_lr)

cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure()
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title("LR Confusion Matrix")
plt.savefig("lr_confusion_matrix.png")
plt.close()


# ================================
# 7. MODEL 2: SVM (tuned)
# ================================
svm = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr', random_state=42)
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("\nSVM Accuracy:", acc_svm)

cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure()
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.savefig("svm_confusion_matrix.png")
plt.close()


# ================================
# 8. MODEL 3: RANDOM FOREST (tuned)
# ================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Accuracy:", acc_rf)

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure()
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title("RF Confusion Matrix")
plt.savefig("rf_confusion_matrix.png")
plt.close()


# ================================
# 9. MODEL 4: GRADIENT BOOSTING (NEW)
# ================================
# Gradient Boosting builds trees sequentially, each correcting the previous.
# Often the strongest single model for tabular data.
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42
)
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)
print("\nGradient Boosting Accuracy:", acc_gb)

cm_gb = confusion_matrix(y_test, y_pred_gb)
plt.figure()
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues')
plt.title("Gradient Boosting Confusion Matrix")
plt.savefig("gb_confusion_matrix.png")
plt.close()


# ================================
# 10. MODEL 5: EXTRA TREES (NEW)
# ================================
# Extra Trees is faster than RF and often more robust due to random thresholds.
et = ExtraTreesClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
et.fit(X_train, y_train)

y_pred_et = et.predict(X_test)
acc_et = accuracy_score(y_test, y_pred_et)
print("\nExtra Trees Accuracy:", acc_et)

cm_et = confusion_matrix(y_test, y_pred_et)
plt.figure()
sns.heatmap(cm_et, annot=True, fmt='d', cmap='Blues')
plt.title("Extra Trees Confusion Matrix")
plt.savefig("et_confusion_matrix.png")
plt.close()


# ================================
# 11. MODEL 6: MLP NEURAL NETWORK (NEW)
# ================================
# Multi-Layer Perceptron — learns non-linear feature combinations.
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=800,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
print("\nMLP Neural Network Accuracy:", acc_mlp)

cm_mlp = confusion_matrix(y_test, y_pred_mlp)
plt.figure()
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues')
plt.title("MLP Neural Network Confusion Matrix")
plt.savefig("mlp_confusion_matrix.png")
plt.close()


# ================================
# 12. MODEL 7: STACKING ENSEMBLE (NEW — BEST)
# ================================
# Stacking uses base models as feature generators, then a meta-model
# combines their predictions. This is typically the highest-accuracy approach.
base_estimators = [
    ('rf',  RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
    ('gb',  GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)),
    ('et',  ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                          early_stopping=True, random_state=42)),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
]

stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, C=1.0),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method='predict_proba',
    n_jobs=-1
)
stack.fit(X_train, y_train)

y_pred_stack = stack.predict(X_test)
acc_stack = accuracy_score(y_test, y_pred_stack)
print("\nStacking Ensemble Accuracy:", acc_stack)

cm_stack = confusion_matrix(y_test, y_pred_stack)
plt.figure()
sns.heatmap(cm_stack, annot=True, fmt='d', cmap='Blues')
plt.title("Stacking Ensemble Confusion Matrix")
plt.savefig("stack_confusion_matrix.png")
plt.close()

print("\nDetailed Report — Stacking Ensemble:\n")
print(classification_report(y_test, y_pred_stack, target_names=['Low', 'Medium', 'High']))


# ================================
# 13. COMPARISON GRAPH (all 7 models)
# ================================
models_names = [
    'Logistic\nRegression', 'SVM', 'Random\nForest',
    'Gradient\nBoosting', 'Extra\nTrees', 'MLP Neural\nNetwork', 'Stacking\nEnsemble'
]
accuracies = [acc_lr, acc_svm, acc_rf, acc_gb, acc_et, acc_mlp, acc_stack]

colors = ['#4C72B0'] * 6 + ['#DD8452']   # highlight the stacking bar

plt.figure(figsize=(12, 6))
bars = plt.bar(models_names, accuracies, color=colors, edgecolor='white', linewidth=0.8)
plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.axhline(y=1/3, color='red', linestyle='--', linewidth=1.2, label='Random Baseline (33.3%)')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{acc:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()


# ================================
# 14. SAVE BEST MODEL
# ================================
all_models = {
    'logistic_regression.pkl':  (lr,    acc_lr),
    'svm_model.pkl':            (svm,   acc_svm),
    'random_forest.pkl':        (rf,    acc_rf),
    'gradient_boosting.pkl':    (gb,    acc_gb),
    'extra_trees.pkl':          (et,    acc_et),
    'mlp_neural_network.pkl':   (mlp,   acc_mlp),
    'stacking_ensemble.pkl':    (stack, acc_stack),
}

best_model_name = max(all_models, key=lambda k: all_models[k][1])
best_model_obj, best_accuracy = all_models[best_model_name]

joblib.dump(best_model_obj, best_model_name)

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
for name, (_, acc) in sorted(all_models.items(), key=lambda x: x[1][1], reverse=True):
    marker = " <-- BEST" if name == best_model_name else ""
    print(f"  {name.replace('.pkl',''):<28} {acc:.4f}{marker}")

print(f"\nBest Model Saved as : {best_model_name}")
print(f"Best Accuracy       : {best_accuracy:.4f}")
print("="*50)
