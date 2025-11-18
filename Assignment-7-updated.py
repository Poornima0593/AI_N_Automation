# MNIST MLP Hyperparameter Tuning - Merged Version
# - Script A structure (functions + main)
# - Script B-style hyperparameter tuning (with batch size)
# - Baseline model = Script A's deep MLP (512-256-128)
# - No classification report / confusion matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import time
import shutil

# -----------------------------
# Reproducibility
# -----------------------------
np.random.seed(42)
tf.random.set_seed(42)


# -----------------------------
# 1️⃣ Load and preprocess MNIST dataset
# -----------------------------
def load_dataset():
    # Load the MNIST dataset (handwritten digits)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Flatten the 28x28 images into a 1D array of 784 pixels
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    return x_train, y_train, x_test, y_test


# -----------------------------
# 2️⃣ Train baseline deep MLP (Script A baseline: 512-256-128)
# -----------------------------
def train_baseline_model(x_train, y_train, x_test, y_test):
    print("\nSTEP 2: Training baseline Deep MLP model (512-256-128)...")
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

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )
    training_time = time.time() - start_time

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nBaseline Model Test Accuracy: {test_acc:.4f}")
    print(f"Baseline Model Test Loss: {test_loss:.4f}")
    print(f"Baseline Training Time: {training_time:.2f} seconds\n")

    return model, test_acc, history


# -----------------------------
# 3️⃣ Hyperparameter Tuning (Script B style, function-based)
# -----------------------------
def build_tunable_model(hp):
    """
    Build MLP model with tunable hyperparameters (Script B style).
    - Tunable input units
    - Tunable number of hidden layers
    - Tunable units per hidden layer
    - Tunable dropout
    - Tunable learning rate
    """

    model = keras.Sequential()

    # Input layer units (optimized range: 256–512)
    input_units = hp.Int('input_units', min_value=256, max_value=512, step=64)
    model.add(layers.Dense(input_units, activation='relu', input_shape=(784,)))

    # Hidden layers: 2–3 layers
    num_layers = hp.Int('num_layers', min_value=2, max_value=3)

    for i in range(num_layers):
        units = hp.Int(f'layer_{i}_units', min_value=128, max_value=256, step=32)
        model.add(layers.Dense(units, activation='relu'))

        dropout = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.3, step=0.1)
        model.add(layers.Dropout(dropout))

    # Output layer
    model.add(layers.Dense(10, activation='softmax'))

    # Learning rate (log sampling, 5e-4 to 3e-3)
    learning_rate = hp.Float(
        'learning_rate',
        min_value=5e-4,
        max_value=3e-3,
        sampling='log'
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def hyperparameter_tuning(x_train, y_train, x_test, y_test):
    print("\nSTEP 3: Hyperparameter Tuning (Random Search, Script B style)")
    print("Tuning: input units, hidden layers, units, dropout, learning rate, batch size")

    class MyHyperModel(kt.HyperModel):
        def build(self, hp):
            return build_tunable_model(hp)

        def fit(self, hp, model, *args, **kwargs):
            # Batch size is also a hyperparameter
            batch_size = hp.Choice('batch_size', [32, 64, 128, 256])
            return model.fit(*args, batch_size=batch_size, **kwargs)

    # Clean tuning directory
    shutil.rmtree("mnist_tuning_merged", ignore_errors=True)

    tuner = kt.RandomSearch(
        MyHyperModel(),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='mnist_tuning_merged',
        project_name='mlp_tuning_complete',
        overwrite=True
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    print("\nStarting hyperparameter search...\n")
    tuning_start = time.time()
    tuner.search(
        x_train, y_train,
        epochs=10,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=1
    )
    tuning_time = time.time() - tuning_start
    print(f"\nHyperparameter search completed in {tuning_time/60:.2f} minutes")

    # Best HPs and model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    # Quick evaluation before final training
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTuned Model (before final retraining) Test Accuracy: {test_acc:.4f}")
    print(f"Tuned Model (before final retraining) Test Loss: {test_loss:.4f}")

    # Print best hyperparameters
    print("\n===== Best Hyperparameters Found =====")
    print(f"Input layer units: {best_hps.get('input_units')}")
    num_layers = best_hps.get('num_layers')
    print(f"Number of hidden layers: {num_layers}")

    for i in range(num_layers):
        print(f"  Hidden layer {i+1} units: {best_hps.get(f'layer_{i}_units')}")
        print(f"  Hidden layer {i+1} dropout: {best_hps.get(f'dropout_{i}'):.2f}")

    print(f"Learning rate: {best_hps.get('learning_rate'):.6f}")
    print(f"Batch size: {best_hps.get('batch_size')}")

    return best_model, best_hps, tuning_time


# -----------------------------
# 4️⃣ Plot training convergence (accuracy)
# -----------------------------
def analyze_training_convergence(history, title="Model Training Convergence"):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------
# 5️⃣ Plot loss curves for a single model
# -----------------------------
def plot_loss_curves(history, title="Loss Curves (Training vs Validation)"):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------
# 6️⃣ Summarize tuning results
# -----------------------------
def summarize_results(best_hps, test_acc, tuning_time, baseline_acc=None):
    print("\n===== Hyperparameter Tuning Results Summary =====")

    print(f"Input layer units: {best_hps.get('input_units')}")
    num_layers = best_hps.get("num_layers")
    print(f"Number of hidden layers: {num_layers}")

    for i in range(num_layers):
        units = best_hps.get(f'layer_{i}_units')
        dropout = best_hps.get(f'dropout_{i}')
        print(f"  Hidden layer {i+1} units: {units}")
        print(f"  Hidden layer {i+1} dropout: {dropout:.2f}")
        print("-" * 40)

    print(f"Learning rate: {best_hps.get('learning_rate'):.6f}")
    print(f"Batch size: {best_hps.get('batch_size')}")
    print(f"Tuned Model Test Accuracy: {test_acc:.4f}")
    print(f"Tuning Time: {tuning_time/60:.2f} minutes")

    if baseline_acc is not None:
        improvement = test_acc - baseline_acc
        print(f"Baseline Model Test Accuracy: {baseline_acc:.4f}")
        print(f"Improvement over Baseline: {improvement:.4f}")


# -----------------------------
# 7️⃣ Compare Baseline vs Tuned Model (accuracy curves)
# -----------------------------
def compare_models(baseline_acc, tuned_acc, history_baseline, history_tuned):
    improvement = tuned_acc - baseline_acc

    print("\n===== Model Comparison =====")
    print(f"Baseline Model Test Accuracy: {baseline_acc:.4f}")
    print(f"Tuned Model Test Accuracy: {tuned_acc:.4f}")
    print(f"Improvement in Test Accuracy: {improvement:.4f}")

    # Training accuracy comparison
    plt.figure(figsize=(8, 5))
    plt.plot(history_baseline.history['accuracy'],
             label='Baseline Train Accuracy', linestyle='--')
    plt.plot(history_tuned.history['accuracy'],
             label='Tuned Train Accuracy', linestyle='-')
    plt.title('Baseline vs Tuned Model - Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Validation accuracy comparison
    plt.figure(figsize=(8, 5))
    plt.plot(history_baseline.history['val_accuracy'],
             label='Baseline Val Accuracy', linestyle='--')
    plt.plot(history_tuned.history['val_accuracy'],
             label='Tuned Val Accuracy', linestyle='-')
    plt.title('Baseline vs Tuned Model - Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------
# 8️⃣ Combined accuracy plot
# -----------------------------
def plot_combined_accuracy(history_baseline, history_tuned):
    plt.figure(figsize=(8, 6))

    # Training accuracy
    plt.plot(history_baseline.history['accuracy'],
             label='Baseline Train Accuracy', linestyle='--')
    plt.plot(history_tuned.history['accuracy'],
             label='Tuned Train Accuracy', linestyle='-')

    # Validation accuracy
    plt.plot(history_baseline.history['val_accuracy'],
             label='Baseline Val Accuracy', linestyle='--')
    plt.plot(history_tuned.history['val_accuracy'],
             label='Tuned Val Accuracy', linestyle='-')

    plt.title("Baseline vs Tuned Model - Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------
# 9️⃣ Accuracy comparison table
# -----------------------------
def print_accuracy_table(baseline_train_acc, baseline_val_acc,
                         tuned_train_acc, tuned_val_acc):

    print("\n===== Accuracy Comparison Table =====")
    print(f"{'Metric':<25}{'Baseline':<15}{'Tuned Model'}")
    print("-" * 55)
    print(f"{'Training Accuracy':<25}{baseline_train_acc:<15.4f}{tuned_train_acc:.4f}")
    print(f"{'Validation Accuracy':<25}{baseline_val_acc:<15.4f}{tuned_val_acc:.4f}")
    print("-" * 55)
    improvement = tuned_val_acc - baseline_val_acc
    print(f"{'Val Accuracy Improvement':<25}{'':<15}{improvement:.4f}")


# -----------------------------
# 🔟 Main pipeline
# -----------------------------
def main():
    # Load data
    x_train, y_train, x_test, y_test = load_dataset()

    # Train baseline model
    baseline_model, baseline_acc, history_baseline = train_baseline_model(
        x_train, y_train, x_test, y_test
    )

    # Hyperparameter tuning (Script B style)
    best_model, best_hps, tuning_time = hyperparameter_tuning(
        x_train, y_train, x_test, y_test
    )

    # Display tuned model architecture
    print("\n===== Tuned Model Full Architecture =====")
    for idx, layer in enumerate(best_model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            print(f"Layer {idx + 1}: {layer.units} neurons, Activation: {layer.activation.__name__}")
        else:
            print(f"Layer {idx + 1}: {layer.name} ({layer.__class__.__name__})")

    # Final training of tuned model with best batch size
    best_batch = best_hps.get("batch_size")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )

    print("\nSTEP 5: Training final tuned model with best hyperparameters...")
    start_time = time.time()
    history_tuned = best_model.fit(
        x_train, y_train,
        validation_split=0.15,
        epochs=10,
        batch_size=best_batch,
        callbacks=[early_stop],
        verbose=2
    )
    final_training_time = time.time() - start_time
    print(f"\nFinal tuned model training time: {final_training_time:.2f} seconds "
          f"({final_training_time/60:.2f} minutes)")

    # Loss curves for tuned model
    plot_loss_curves(history_tuned, title="Tuned Model Loss Convergence")

    # Accuracy statistics
    baseline_train_acc = history_baseline.history['accuracy'][-1]
    baseline_val_acc = history_baseline.history['val_accuracy'][-1]
    tuned_train_acc = history_tuned.history['accuracy'][-1]
    tuned_val_acc = history_tuned.history['val_accuracy'][-1]

    # Combined accuracy plot
    plot_combined_accuracy(history_baseline, history_tuned)

    # Final evaluation on test set
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTuned Model Final Test Accuracy: {test_acc:.4f}")
    print(f"Tuned Model Final Test Loss: {test_loss:.4f}")

    # Accuracy convergence plot
    analyze_training_convergence(history_tuned, title="Tuned Model Accuracy Convergence")

    # Summary of tuning
    summarize_results(best_hps, test_acc, tuning_time, baseline_acc=baseline_acc)

    # Baseline vs tuned comparison
    compare_models(baseline_acc, test_acc, history_baseline, history_tuned)

    # Accuracy comparison table
    print_accuracy_table(
        baseline_train_acc, baseline_val_acc,
        tuned_train_acc, tuned_val_acc
    )


# -----------------------------
# Execute main
# -----------------------------
if __name__ == "__main__":
    main()
