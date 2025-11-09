import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner import RandomSearch
import time
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping



# -----------------------------
# 1️⃣ Load and preprocess MNIST dataset
# -----------------------------
def load_dataset():
    # Load the MNIST dataset (handwritten digits)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Flatten the 28x28 images into a 1D array of 784 pixels
    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    return x_train, y_train, x_test, y_test

# -----------------------------
# 2️⃣ Train baseline deep MLP
# -----------------------------
def train_baseline_model(x_train, y_train, x_test, y_test):
    print("Training baseline Deep MLP model...")
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=2
    )
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Baseline Model Test Accuracy: {test_acc:.4f}\n")
    return model, test_acc, history


def hyperparameter_tuning(x_train, y_train, x_test, y_test):
    print("Starting hyperparameter tuning with Keras Tuner...")

    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(784,)))

        # Manually add 3 hidden layers
        units1 = hp.Int("units_1", 128, 512, step=64)
        model.add(layers.Dense(units=units1, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        units2 = hp.Int("units_2", 64, 256, step=32)
        model.add(layers.Dense(units=units2, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        units3 = hp.Int("units_3", 32, 128, step=32)
        model.add(layers.Dense(units=units3, activation="relu"))
        model.add(layers.BatchNormalization())

        model.add(layers.Dropout(0.2))

        # Output layer
        model.add(layers.Dense(10, activation="softmax"))

        # Learning rate tuning
        lr = hp.Choice("learning_rate", [0.01, 0.005, 0.001, 0.0005, 0.0001])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model


    tuner = RandomSearch(build_model,objective="val_accuracy",max_trials=10,executions_per_trial=1,directory="mnist_tuner",project_name="deep_mlp_tuning")
    #tuner = RandomSearch(build_model,objective="val_accuracy",max_trials=10)

    start_time = time.time()
    tuner.search(x_train, y_train, validation_split=0.2, epochs=10, batch_size=64, verbose=1)
    tuning_time = time.time() - start_time

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

   
    print("\n===== Best Hyperparameters Found =====")
    print(f"Units in Layer 1: {best_hps.get('units_1')}")
    print(f"Units in Layer 2: {best_hps.get('units_2')}")
    print(f"Units in Layer 3: {best_hps.get('units_3')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")


   # -------------------------
    # Evaluate accuracy on test set
    # -------------------------
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy of Tuned Model: {test_acc:.4f}")

    return best_model, best_hps, tuning_time


# -----------------------------
# 4️⃣ Plot training convergence
# -----------------------------
def analyze_training_convergence(history, title="Model Training Convergence"):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------
# 5️⃣ Summarize results
# -----------------------------

def summarize_results(best_hps, test_acc, tuning_time, baseline_acc=None):
    print("\n===== Hyperparameter Tuning Results =====")
    
    # Fixed 3 layers
    print(f"Units in Layer 1: {best_hps.get('units_1')}")
    print(f"Units in Layer 2: {best_hps.get('units_2')}")
    print(f"Units in Layer 3: {best_hps.get('units_3')}")
    
    # Learning rate
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    print(f"Tuned Model Test Accuracy: {test_acc:.4f}")
    print(f"Total Hyperparameter Tuning Time: {tuning_time/60:.2f} minutes")
    
    if baseline_acc is not None:
        print(f"Baseline Model Test Accuracy: {baseline_acc:.4f}")
        improvement = test_acc - baseline_acc
        print(f"Improvement over baseline: {improvement:.4f}")




# -----------------------------
# 6️⃣ Compare Baseline vs Tuned Model
# -----------------------------
def compare_models(baseline_acc, tuned_acc, history_baseline, history_tuned):

    # 1️⃣ Print comparison
    improvement = tuned_acc - baseline_acc

    print("\n===== Model Comparison =====")
    print(f"Baseline Model Validation Accuracy: {baseline_acc:.4f}")
    print(f"Tuned Model Validation Accuracy: {tuned_acc:.4f}")
    print(f"Improvement in Validation Accuracy: {improvement:.4f}")

    # 2️⃣ Plot convergence comparison
    plt.figure(figsize=(8,5))
    plt.plot(history_baseline.history['val_accuracy'], label='Baseline Validation Accuracy', linestyle='--')
    plt.plot(history_tuned.history['val_accuracy'], label='Tuned Validation Accuracy', linestyle='-')
    plt.title('Baseline vs Tuned Model Accuracy Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Load data
    x_train, y_train, x_test, y_test = load_dataset()

    # # Train baseline model
    baseline_model, baseline_acc, history_baseline = train_baseline_model(x_train, y_train, x_test, y_test)

    # Hyperparameter tuning
    best_model, best_hps, tuning_time = hyperparameter_tuning(x_train, y_train, x_test, y_test)

    # Display full architecture
    print("\n===== Tuned Model Full Architecture =====")
    for idx, layer in enumerate(best_model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            print(f"Layer {idx+1}: {layer.units} neurons, Activation: {layer.activation.__name__}")
    else:
        print(f"Layer {idx+1}: {layer.name} (not a Dense layer)")
    
    history = best_model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=64,
        verbose=2
    )
    
    # Evaluate on test set
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"Tuned Model Test Accuracy: {test_acc:.4f}")

    # Plot convergence
    analyze_training_convergence(history, title="Tuned Model Accuracy Convergence")

    # Summarize results
    summarize_results(best_hps, test_acc, tuning_time, baseline_acc=baseline_acc)

    # Compare baseline vs tuned models
    compare_models(baseline_acc, test_acc, history_baseline, history)





# -----------------------------
# Execute main
# -----------------------------
if __name__ == "__main__":
    main()
