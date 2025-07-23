import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("outputs/day7_output.csv", parse_dates=['date'])
data.set_index('date', inplace=True)

# target is if the return is greater than 0
data['Target'] = (data['Yield_Return'].shift(-1) > 0).astype(int)
data.dropna(inplace=True)

# features set
features = ['momentum_3d', 'roc_5d', 'rolling_sharpe_5', 'z_score_10', 'volatility_band', 'SMA_diff', 'lag_1', 'lag_2', 'volatility_3d']
if 'Signal' in data.columns:
    features.append('Signal')

# add the new features to test the feature set
data['momentum_z'] = data['momentum_3d'] * data['z_score_10']
data['roc_volatility'] = data['roc_5d'] * data['volatility_band']
features += ['momentum_z', 'roc_volatility']

# final sets
X = data[features]
y = data['Target']

# split for pre covid
X_train = X.loc[:'2011-12-31']
X_test = X.loc['2018-01-01':]
y_train = y.loc[:'2011-12-31']
y_test = y.loc['2018-01-01':]

# scale the sets 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
xgb = XGBClassifier(n_estimators=100, max_depth=3, scale_pos_weight=1, eval_metric='logloss')
logreg.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# predictions
p_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
p_rf = rf.predict_proba(X_test)[:, 1]
p_xgb = xgb.predict_proba(X_test)[:, 1]

# ensemble again
ensemble_probs = (0.2 * p_logreg + 0.4 * p_rf + 0.4 * p_xgb)
if len(X_train) > 1000:
    threshold = 0.56
else:
    threshold = 0.48 # use a lower threadhold for older data due to the the way the data is trained
    
y_pred = (ensemble_probs > threshold).astype(int)

print("Ensemble with Expanded Features:")
print(classification_report(y_test, y_pred))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix – Day 8 Ensemble")
plt.grid(False)
plt.tight_layout()
plt.show()

# gives the feature importance
importances = rf.feature_importances_
feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
feat_df.sort_values(by="Importance", ascending=False, inplace=True)

plt.figure(figsize=(8, 4))
plt.bar(feat_df["Feature"], feat_df["Importance"])
plt.title("Feature Importance – Random Forest")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

data.to_csv("outputs/day8_output.csv")