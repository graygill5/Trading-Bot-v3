import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("outputs/day8_output.csv", parse_dates=['date'])
data.set_index('date', inplace=True)

# features
features = ['momentum_3d', 'roc_5d', 'rolling_sharpe_5', 'z_score_10', 'volatility_band','SMA_diff', 'lag_1', 'lag_2', 'volatility_3d', 'Signal','momentum_z', 'roc_volatility']
target = 'Target'

X = data[features]
y = data[target]

# look for outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=X)
plt.title("Boxplot of All Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# look for low variance
print("Missing values:\n", X.isna().sum())
print("\nLow-variance features (nunique):\n", X.nunique().sort_values())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# get rid of useless features
rfe = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=7)
rfe.fit(X, y)
selected = list(pd.Series(features)[rfe.support_])
print("\nSelected Features by RFE:", selected)

# grid serach to further refine the parameters
tscv = TimeSeriesSplit(n_splits=5)
params = {'n_estimators': [100, 200], 'max_depth': [3, 5, 10], 'min_samples_split': [2, 5],}
grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), params, cv=tscv, scoring='f1')
grid.fit(X[selected], y)

# final model for parameters chosen
model = grid.best_estimator_
print("\nBest RF Parameters:", grid.best_params_)

# predicts
y_pred = model.predict(X[selected])
print("\n📊 Final Evaluation Report:")
print(classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix – Final RF (Day 9)")
plt.grid(False)
plt.tight_layout()
plt.show()

data.to_csv("outputs/day9_output.csv")