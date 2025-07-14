import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_importance

df = pd.read_csv('Dataset.csv')

X = df.drop(columns=['MPG'])
y = df['MPG']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=1000,
    learning_rate = 0.05,
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42,
)

model.fit(X_train, y_train)
model.save_model("mpg_predictor.json")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

'''
MAE:  1.7902198886871339
RMSE: 6.255419627250442
R2:   0.8836556931989678
'''

importances = model.feature_importances_
features = df.columns
indices = importances.argsort()

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title('Feature Importance')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()

plot_importance(model)
plt.tight_layout()
plt.show()