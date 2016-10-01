from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X = iris.data
y = iris.target
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

kf = KFold(n_splits=2)

max_score = 0
best_depth = 1.0
for tree_depth in range(1, 10):
    scores = []
    print("trying tree depth:"+str(tree_depth))
    for train_id, validate_id in kf.split(X_tr):
        X_train = X_tr[train_id]
        X_val = X_tr[validate_id]
        y_train = y_tr[train_id]
        y_val = y_tr[validate_id]
        
        model = DecisionTreeClassifier(max_depth=tree_depth)
        model.fit(X_train, y_train)
        
        scores.append(model.score(X_val, y_val))
    score = np.mean(scores)
    print("score:"+str(np.mean(score))+"\n")
    if score > max_score:
        best_depth = tree_depth
        
print("best depth: ")
print(best_depth)

X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

best_model = DecisionTreeClassifier(max_depth=best_depth)
best_model.fit(X_tr, y_tr)

print("score:")
print(best_model.score(X_test, y_test))
