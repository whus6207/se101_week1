from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()
print(iris.keys())

print("feature names are :")
print(iris.feature_names)

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("training data size:")
print(X_train.shape)
print("testing data size:")
print(X_test.shape)

model = DecisionTreeClassifier(max_depth=1)
model.fit(X_train, y_train)

print("error: "+str(model.score(X_test, y_test)))