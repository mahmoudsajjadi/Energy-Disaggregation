
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns

file_path = 'C:\\Users\\Sanic\\OneDrive\\Documents\\Github\\Energy-Disaggregation\\Datasets\\REDD\\redd_house1_0.csv'

data = pd.read_csv(file_path)

# Assuming each row represents a fixed time interval, aggregate data weekly
data['week'] = data.index // 168  # Assuming hourly data
weekly_data = data.groupby('week').sum()

percentiles = weekly_data['main'].quantile([0.33, 0.66]).values
weekly_data['energy_category'] = pd.cut(weekly_data['main'],
                                         bins=[-float('inf'), percentiles[0], percentiles[1], float('inf')],
                                         labels=['Low', 'Medium', 'High'])

X = weekly_data.drop(columns=['main', 'Unnamed: 0', 'energy_category'])
y = weekly_data['energy_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

feature_importances = classifier.feature_importances_
features = X_train.columns
sorted_indices = feature_importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances in Classification Tree")
plt.bar(range(X_train.shape[1]), feature_importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), features[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classifier.classes_, yticklabels=classifier.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

plt.figure(figsize=(20,10))
tree.plot_tree(classifier, max_depth=3, feature_names=X_train.columns, filled=True, rounded=True, class_names=classifier.classes_)
plt.show()
