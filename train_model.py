from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Загрузка данных MNIST
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Создание и обучение модели RandomForestClassifier
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train, y_train)

# Оценка точности модели
accuracy = clf.score(X_test, y_test)
print(f"Точность модели: {accuracy:.2f}")

# Сохранение модели в файл
with open('mnist_model.pkl', 'wb') as f:
    pickle.dump(clf, f)