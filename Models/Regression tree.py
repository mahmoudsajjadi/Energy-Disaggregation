
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import tree

file_path = 'C:\\Users\\Sanic\\OneDrive\\Documents\\Github\\Energy-Disaggregation\\Datasets\\REDD\\redd_house1_0.csv'

data = pd.read_csv(file_path)

data['week'] = data.index // 168  # Assuming hourly data
weekly_data = data.groupby('week').sum()

X = weekly_data.drop(columns=['main', 'Unnamed: 0'])
y = weekly_data['main']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

feature_importances = regressor.feature_importances_
features = X_train.columns
sorted_indices = feature_importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), feature_importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), features[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Prediction vs Actual')
plt.xlabel('Actual Energy Consumption')
plt.ylabel('Predicted Energy Consumption')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Diagonal line
plt.show()

plt.figure(figsize=(20,10))
tree.plot_tree(regressor, max_depth=3, feature_names=X_train.columns, filled=True, rounded=True)
plt.show()
