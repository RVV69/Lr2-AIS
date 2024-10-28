from sklearn.datasets import load_iris
from pandas import read_csv, DataFrame
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

iris_dataset = load_iris()

print("Ключі iris_dataset:\n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Назви відповідей:\n{}".format(iris_dataset['target_names']))
print("Назви ознак:\n{}".format(iris_dataset['feature_names']))

print("Форма масиву data:\n{}".format(iris_dataset['data'].shape))

print("Перші 5 прикладів:\n{}".format(iris_dataset['data'][:5]))

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

print("Розмір датасету:", dataset.shape)

print("Перші 20 рядків датасету:\n", dataset.head(20))

print("Статистичне резюме:\n", dataset.describe())

print("Кількість екземплярів для кожного класу:\n", dataset.groupby('class').size())

dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()

# Багатовимірна візуалізація - матриця діаграм розсіювання
scatter_matrix(dataset)
plt.show()

# КРОК 4: Підготовка даних для тренування та тестування
# Виділяємо ознаки та цільові значення
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]

# Розділення на навчальну та тестову вибірки
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# КРОК 5: Побудова та оцінка моделей
# Список моделей для тестування
models = []
models.append(('LR', OneVsRestClassifier(LogisticRegression(solver='liblinear'))))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Оцінка моделей з використанням 10-кратної стратифікованої крос-валідації
results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.3f} ({cv_results.std():.3f})")

# Порівняння моделей на графіку
plt.boxplot(results, tick_labels=names)
plt.title("Порівняння точності моделей")
plt.show()

# КРОК 6: Прогнозування та оцінка на тестовій вибірці
# Вибір кращої моделі (наприклад, SVM)
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Step 7: Оцінка моделі
accuracy = accuracy_score(Y_validation, predictions)
conf_matrix = confusion_matrix(Y_validation, predictions)
class_report = classification_report(Y_validation, predictions)
print(f"Точність: {accuracy:.3f}")
print("Матриця плутанини:\n", conf_matrix)
print("Звіт класифікації:\n", class_report)

# Step 8: Прогнозувати за новими даними
X_new = np.array([[5, 2.9, 1, 0.2]])
print("Форма X_new:", X_new.shape)
new_prediction = model.predict(X_new)
print("Прогнозування:", new_prediction)
print("Прогнозована мітка класу:", new_prediction[0])



