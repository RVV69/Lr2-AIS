import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import matplotlib.pyplot as plt
import os

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

data = pd.read_csv('income_data.txt', delimiter=",")

data.columns = data.columns.str.strip()

data_encoded = pd.get_dummies(data.drop(columns=['<=50K']))

data_encoded['<=50K'] = data['<=50K']

data_encoded['<=50K'] = data_encoded['<=50K'].apply(lambda x: 0 if x == ' <=50K' else 1)

data_sampled = data_encoded.sample(n=1000, random_state=42)

X = data_sampled.drop(columns=['<=50K']).values
y = data_sampled['<=50K'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = [
    ('LR', LogisticRegression(solver='liblinear')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

results = []
names = []
scoring = 'accuracy'
for name, model in models:
    cv_results = cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.3f} ({cv_results.std():.3f})")

plt.boxplot(results, tick_labels=names)
plt.title("Порівняння точності моделей на income_data")
plt.show()

best_model = SVC(gamma='auto')
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("Точність:", accuracy_score(y_test, y_pred))
print("Матриця плутанини:\n", confusion_matrix(y_test, y_pred))
print("Звіт класифікації:\n", classification_report(y_test, y_pred, zero_division=1))