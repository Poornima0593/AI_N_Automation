from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(8,6))
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.show()



mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X, y= mnist["data"], mnist["target"]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Create a smaller subset for faster experimentation
X_train_small, y_train_small = X_train[:10000], y_train[:10000]

print(X_test.shape)
print(y_test.shape)

scaler = StandardScaler()
# Scale the smaller subset for SGD
X_train_small_scaled = scaler.fit_transform(X_train_small.astype("float64"))

#---------------------------------------------------
####KNN Serach############
#---------------------------------------------------


param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=3, verbose=3)
grid_search.fit(X_train, y_train)

print(f"KNN best parameters : {grid_search.best_params_}")  
print(f"KNN best score : {grid_search.best_score_}")    
KNN_predit=grid_search.predict(X_test)
knn_acc=accuracy_score(y_test, KNN_predit)
print(f"KNN best accuracy value:", knn_acc)


scores = cross_val_score(knn_clf, X_train, y_train, cv=5)  # 5-fold CV on entire dataset
print("Cross-validation scores:", scores)
print("Mean CV accuracy:", scores.mean())

best_knn = KNeighborsClassifier(**grid_search.best_params_)
plot_learning_curve(best_knn, X_train, y_train, title="KNN Learning Curve")


conf_mx = confusion_matrix(y_test, KNN_predit)

plt.figure(figsize=(8,8))
sns.heatmap(conf_mx, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")
plt.show()


#---------------------------------------------------
####SGD Serach############
#---------------------------------------------------

# Create a smaller subset for faster experimentation
X_train_small, y_train_small = X_train[:10000], y_train[:10000]

print(X_test.shape)
print(y_test.shape)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler = StandardScaler()
# Scale the smaller subset for SGD
X_train_small_scaled = scaler.fit_transform(X_train_small.astype("float64"))

####SGD Serach############

sgd_clf = SGDClassifier(random_state=42)
#It's not necessary to fit the base classifier before the grid search

param_grid = {
    'loss': ['hinge', 'log_loss'],     # 'log' is deprecated, use 'log_loss' for Logistic Regression
    'alpha': [0.0001, 0.001],    # regularization strength
    'penalty': ['l2', 'l1'],           # regularization type
    'max_iter': [1000]                  # number of passes over data
}

SGDgrid_search = GridSearchCV(sgd_clf, param_grid, cv=3, verbose=3, n_jobs=-1)
# Use the scaled data for better performance
SGDgrid_search.fit(X_train_scaled, y_train)

print(f"Best parameters found: {SGDgrid_search.best_params_}")
SGD_predit=SGDgrid_search.predict(X_test_scaled)
sgd_acc=accuracy_score(y_test, SGD_predit)          
print(f"SGD best accuracy value:", sgd_acc)


SGDconf_mx = confusion_matrix(y_test, SGD_predit)   


plt.figure(figsize=(8,8))
sns.heatmap(SGDconf_mx, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SGD Confusion Matrix")
plt.show()



#---------------------------------------------------
####RandomForest Classifier Serach############
#---------------------------------------------------

rf = RandomForestClassifier(random_state=42)

# Use a smaller grid for faster testing
param_grid= {
    'n_estimators': [100, 200],        # Reduced number of trees
    'max_depth': [None, 10, 20],            # Limit tree depth
    'min_samples_split': [2, 5, 7],
    'min_samples_leaf': [1, 2, 3],
}


grid_search_rf = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=3)
# Fit on the smaller, unscaled subset
grid_search_rf.fit(X_train, y_train)

print(f"\nRandom Forest Best parameters found: {grid_search_rf.best_params_}")
print(f"Random Forest Best score: {grid_search_rf.best_score_}")

RF_predit=grid_search_rf.predict(X_test)
rf_acc=accuracy_score(y_test, RF_predit)  
print(f"Random Forest best accuracy value:", rf_acc)
RFconf_mx = confusion_matrix(y_test, RF_predit)

best_rf = RandomForestClassifier(**grid_search_rf.best_params_, random_state=42)
plot_learning_curve(best_rf, X_train_small, y_train_small, title="Random Forest Learning Curve")

plt.figure(figsize=(8,8))
sns.heatmap(RFconf_mx, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()


# --- Compare Accuracies ---
print(f"KNN best accuracy value: {knn_acc * 100:.2f}%")
print(f"SGD best accuracy value: {sgd_acc * 100:.2f}%")
print(f"Random Forest best accuracy value: {rf_acc * 100:.2f}%")

# Optional: Bar plot for visual comparison

models = ['KNN', 'SGDClassifier', 'RandomForest']
accuracies = [knn_acc * 100, sgd_acc * 100, rf_acc * 100]


plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.ylabel('Accuracy')
plt.title('Classifier Accuracy Comparison')
plt.ylim(0.8, 1.0)
plt.show()