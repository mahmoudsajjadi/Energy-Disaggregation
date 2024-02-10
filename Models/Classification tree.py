import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

file_path = 'C:\\Users\\Sanic\\OneDrive\\Documents\\Github\\Energy-Disaggregation\\Datasets\\REDD\\redd_house1_0.csv'
data = pd.read_csv(file_path)

data['week'] = data.index // 168  # Assuming hourly data
weekly_data = data.groupby('week').sum()

percentiles = weekly_data['main'].quantile([0.33, 0.66]).values
weekly_data['energy_category'] = pd.cut(weekly_data['main'],
                                         bins=[-float('inf'), percentiles[0], percentiles[1], float('inf')],
                                         labels=['Low', 'Medium', 'High'])

weekly_data.reset_index(inplace=True)

X = weekly_data.drop(columns=['main', 'Unnamed: 0', 'energy_category', 'week'])  # 'week' column excluded from features
y = weekly_data['energy_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print(f"Best cross-validated accuracy: {grid_search.best_score_}")

classifier = grid_search.best_estimator_
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
new_accuracy = accuracy_score(y_test, y_pred)
print(f"New accuracy: {new_accuracy}")
print(f"New Classification Report:\n{classification_report(y_test, y_pred)}")
