import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# === Load and Prepare Data ===
data = pd.read_csv("outputs/day5_output.csv", parse_dates=['date'])
data.set_index('date', inplace=True)

# === Feature Engineering ===
data['SMA_diff'] = data['SMA_5'] - data['SMA_20']
data['pct_change'] = data['US10Y'].pct_change()
data['lag_1'] = data['US10Y'].shift(1)
data['lag_2'] = data['US10Y'].shift(2)
data['volatility_3d'] = data['Yield_Return'].abs().rolling(3).mean()
data['Target'] = (data['Yield_Return'].shift(-1) > 0).astype(int)
data.dropna(inplace=True)

# === Feature Selection ===
feature_cols = ['SMA_5', 'SMA_20', 'SMA_diff', 'RSI_14', 'pct_change', 'lag_1', 'lag_2', 'volatility_3d', 'Signal']
X = data[feature_cols]
y = data['Target']

# === Train/Test Split ===
X_train = X.loc[:'2019-12-31']
X_test = X.loc['2020-01-01':]
y_train = y.loc[:'2019-12-31']
y_test = y.loc['2020-01-01':]

# === Train Models ===
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
xgb = XGBClassifier(n_estimators=100, max_depth=3, scale_pos_weight=1, eval_metric='logloss')

logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# === Predict Probabilities ===
p_logreg = logreg.predict_proba(X_test)[:, 1]
p_rf = rf.predict_proba(X_test)[:, 1]
p_xgb = xgb.predict_proba(X_test)[:, 1]

# === Ensemble & Signal Boost Logic ===
ensemble_probs = (p_logreg + p_rf + p_xgb) / 3
threshold = 0.49
bias_boost = .20

# Boost probability slightly if Signal = 1
adjusted_probs = ensemble_probs + (X_test['Signal'] * bias_boost)

# Final predictions
combined_preds = (adjusted_probs > threshold).astype(int)

# === Evaluation ===
print("Ensemble Model + Signal Boost (Avg Probs + Signal > 0.45):")
print(classification_report(y_test, combined_preds))

cm = confusion_matrix(y_test, combined_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: Ensemble + Signal Boost")
plt.grid(False)
plt.tight_layout()
plt.show()

plt.hist(ensemble_probs, bins=50)
plt.axvline(threshold, color='red', linestyle='--')
plt.title("Ensemble Probabilities Distribution")
plt.xlabel("Predicted Probability of Up Move")
plt.ylabel("Frequency")
plt.show()