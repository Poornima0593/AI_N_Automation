# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, precision_score,
                             recall_score, f1_score)


# --- Load and prepare MNIST ---
def load_and_prepare_mnist():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    x, y = mnist.data, mnist.target.astype(int)


    # Use standard MNIST split: 60k train, 10k test
    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

    # --- Set train and test sizes ---
    # x_train, x_test = x[:5000], x[60000:70000]   # 5000 for training, 10000 for testing
    # y_train, y_test = y[:5000], y[60000:70000]

    # # --- Set train and test sizes ---
    # # --- Set train and test sizes ---
    # x_train, x_test = x[:10000], x[10000:]   # 10000 for training, 60000 for testing
    # y_train, y_test = y[:10000], y[10000:]

# Scale the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.astype("float64"))
    x_test_scaled = scaler.transform(x_test.astype("float64"))

    print(f"x_train_scaled: {x_train_scaled.shape}, y_train: {y_train.shape}")
    print(f"x_test_scaled: {x_test_scaled.shape}, y_test: {y_test.shape}")

    print(f"Data loaded: {x_train.shape[0]} train samples, {x_test.shape[0]} test samples")

    return x_train_scaled, x_test_scaled, y_train, y_test, x, y


# --- Visualize MNIST sample ---
def visualize_mnist_sample(x, y, n=25):
    plt.figure(figsize=(5, 5))
    for i in range(n):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x[i].reshape(28, 28), cmap='gray')
        plt.title(y[i])
        plt.axis('off')
    plt.suptitle("Sample MNIST Digits")
    plt.show()


# --- Generic training function with metrics ---
def train_with_gridsearch(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    print(f"\n{'='*10} {model_name.upper()} GridSearch Starting {'='*10}")
    start_time = time.time()
    
    grid_search = GridSearchCV(model, param_grid, cv=3, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    conf_mx = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    
    train_time = time.time() - start_time

    print(f"\nBest Params: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")
    print(f"Training Time: {train_time:.2f} seconds")
    
    return best_model, acc, conf_mx, precision, recall, f1, train_time, report


# --- Model-specific functions ---
def train_svm_model(kernel, X_train, y_train, X_test, y_test):
    param_grid = {}
    if kernel == 'linear':
        param_grid = {'C': [0.1, 1, 10]}
    elif kernel == 'poly':
        param_grid = {'C': [1, 10], 'degree': [2, 3], 'gamma': ['scale'], 'coef0': [0, 1]}
    elif kernel == 'rbf':
        param_grid = {'C': [1, 10], 'gamma': ['scale', 0.1]}
    else:
        raise ValueError("Unsupported kernel")
    model = SVC(kernel=kernel)
    return train_with_gridsearch(model, param_grid, X_train, y_train, X_test, y_test, f"SVM-{kernel}")


def train_knn(X_train, y_train, X_test, y_test):
    param_grid = {'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}
    model = KNeighborsClassifier()
    return train_with_gridsearch(model, param_grid, X_train, y_train, X_test, y_test, "KNN")


def train_sgd(X_train, y_train, X_test, y_test):
    param_grid = {
        'loss': ['hinge', 'log_loss'],
        'alpha': [0.0001, 0.001],
        'penalty': ['l2', 'l1'],
        'max_iter': [1000]
    }
    model = SGDClassifier(random_state=42)
    return train_with_gridsearch(model, param_grid, X_train, y_train, X_test, y_test, "SGD")


def train_random_forest(X_train, y_train, X_test, y_test):
    # param_grid = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 5, 7],
    #     'min_samples_leaf': [1, 2, 3],
    # }

    param_grid = {
    'n_estimators': [500, 750, 1000],      # number of trees
    'max_depth': [30, 50, None],           # depth of trees
    'min_samples_split': [2, 5, 10],       # min samples to split a node
    'min_samples_leaf': [1, 2, 4],         # min samples in a leaf
    #'max_features': ['sqrt', 'log2']       # number of features per split
}
    
    model = RandomForestClassifier(random_state=42)
    return train_with_gridsearch(model, param_grid, X_train, y_train, X_test, y_test, "Random Forest")


# --- Plot confusion matrix ---
def plot_confusion_matrix(cm, title, save_path=None):
    """
    Plots a confusion matrix, saves it to a file (if a path is provided),
    displays it, and then closes the figure.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Image saved to: {save_path}")

    # Display the plot briefly, then close it
    plt.show(block=False)
    plt.pause(2)   # display for 2 seconds (adjust as needed)
    plt.close()

# --- Function to compare models and display best ---

def plot_model_comparison(results, save_path=None):
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    svm_models = {k: v for k, v in results.items() if 'SVM' in k}
    other_models = {k: v for k, v in results.items() if k not in svm_models}

    # --- Accuracy plots ---
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    models = list(svm_models.keys())
    accuracies = [svm_models[m]['accuracy'] for m in models]
    plt.bar(models, accuracies, color='blue')
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title("SVM Models Accuracy")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.subplot(1,2,2)
    models = list(other_models.keys())
    accuracies = [other_models[m]['accuracy'] for m in models]
    plt.bar(models, accuracies, color=['green','orange','purple'])
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title("KNN, SGD, Random Forest Accuracy")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        acc_path = os.path.join(save_path, "accuracy_comparison.png")
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {acc_path}")

    # Display briefly and then close
    plt.show(block=False)
    plt.pause(0.1)  # short pause to render plot
    plt.close()     # forcefully closes the figure

    # --- Training time plots ---
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    models = list(svm_models.keys())
    times = [svm_models[m]['train_time'] for m in models]
    plt.bar(models, times, color='blue')
    plt.ylabel("Training Time (s)")
    plt.title("SVM Models Training Time")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.subplot(1,2,2)
    models = list(other_models.keys())
    times = [other_models[m]['train_time'] for m in models]
    plt.bar(models, times, color=['green','orange','purple'])
    plt.ylabel("Training Time (s)")
    plt.title("KNN, SGD, Random Forest Training Time")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        time_path = os.path.join(save_path, "training_time_comparison.png")
        plt.savefig(time_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {time_path}")

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

    # --- Best SVM vs others ---
    best_svm_name = max(svm_models, key=lambda k: svm_models[k]['accuracy'])
    best_svm_acc = svm_models[best_svm_name]['accuracy']

    comparison_models = list(other_models.keys()) + [best_svm_name]
    comparison_acc = [other_models[m]['accuracy'] for m in other_models] + [best_svm_acc]

    plt.figure(figsize=(8,5))
    colors = ['green','orange','purple','blue']
    plt.bar(comparison_models, comparison_acc, color=colors)
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title(f"Comparison: Best SVM ({best_svm_name}) vs Other Models")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    if save_path:
        comp_path = os.path.join(save_path, "best_svm_vs_others.png")
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {comp_path}")

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


def display_best_model(results, metric='accuracy'):
    """
    Display all model metrics and highlight the best model based on the chosen metric.

    Parameters:
    - results: dict containing model metrics (accuracy, precision, recall, f1, train_time)
    - metric: which metric to use for selecting the best model ('accuracy', 'precision', 'recall', 'f1')
    """
    print("\n=== Model Comparison Summary ===")

    best_model = None
    best_score = 0

    for name, res in results.items():
        print(f"{name}: Accuracy={res['accuracy']:.4f}, Precision={res['precision']:.4f}, "
              f"Recall={res['recall']:.4f}, F1={res['f1']:.4f}, Training Time={res['train_time']:.2f}s")
        
        if res[metric] > best_score:
            best_score = res[metric]
            best_model = name

    print(f"\n🏆 Best Model based on {metric.capitalize()}: {best_model} with {metric}={best_score:.4f}")



# --- Main ---
def main():
    # --- Load data ---
    X_train_scaled, X_test_scaled, y_train, y_test, X, y = load_and_prepare_mnist()
    visualize_mnist_sample(X, y, n=25)
    
    results = {}

    # # --- Train SVM kernels ---
    for kernel in ['linear', 'poly', 'rbf']:
    #for kernel in ['poly']:
        model, acc, cm, prec, rec, f1, t_time, report = train_svm_model(kernel, X_train_scaled, y_train, X_test_scaled, y_test)
        results[f"SVM-{kernel}"] = {'accuracy': acc, 'confusion_matrix': cm, 'precision': prec,
                                    'recall': rec, 'f1': f1, 'train_time': t_time, 'report': report}
        plot_confusion_matrix(cm, f"SVM-{kernel} Confusion Matrix",save_path = f"C:/Users/Poornima Sarwadi/Documents/Assignment/Assignment-5/{kernel}.png")

    # --- Train KNN ---
    model, acc, cm, prec, rec, f1, t_time, report = train_knn(X_train_scaled, y_train, X_test_scaled, y_test)
    results['KNN'] = {'accuracy': acc, 'confusion_matrix': cm, 'precision': prec,
                      'recall': rec, 'f1': f1, 'train_time': t_time, 'report': report}
    plot_confusion_matrix(cm, "KNN Confusion Matrix",save_path="C:/Users/Poornima Sarwadi/Documents/Assignment/Assignment-5/KNN.png")

    # --- Train SGD ---
    model, acc, cm, prec, rec, f1, t_time, report = train_sgd(X_train_scaled, y_train, X_test_scaled, y_test)
    results['SGD'] = {'accuracy': acc, 'confusion_matrix': cm, 'precision': prec,
                       'recall': rec, 'f1': f1, 'train_time': t_time, 'report': report}
    plot_confusion_matrix(cm, "SGD Confusion Matrix", save_path="C:/Users/Poornima Sarwadi/Documents/Assignment/Assignment-5/SGD.png")

    #--- Train Random Forest ---
    model, acc, cm, prec, rec, f1, t_time, report = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    results['RandomForest'] = {'accuracy': acc, 'confusion_matrix': cm, 'precision': prec,
                               'recall': rec, 'f1': f1, 'train_time': t_time, 'report': report}
    plot_confusion_matrix(cm, "Random Forest Confusion Matrix", save_path="C:/Users/Poornima Sarwadi/Documents/Assignment/Assignment-5/RF.png")

    # #After training all models and storing metrics in `results`
    plot_model_comparison(results, save_path="C:/Users/Poornima Sarwadi/Documents/Assignment/Assignment-5/")

    

    # --- Compare metrics ---
    print("\n=== Model Comparison Summary ===")
    for name, res in results.items():
        print(f"{name}: Accuracy={res['accuracy']:.4f}, Precision={res['precision']:.4f}, "
              f"Recall={res['recall']:.4f}, F1={res['f1']:.4f}, Training Time={res['train_time']:.2f}s")
    
    display_best_model(results, metric='accuracy')

 
if __name__ == "__main__":
    main()
