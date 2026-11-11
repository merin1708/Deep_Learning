import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.initializers import GlorotUniform, HeUniform
from tensorflow.keras.regularizers import l2

import numpy as np
from torchvision.datasets import MNIST

# 1. Load MNIST dataset using PyTorch torchvision (100% offline)
train = MNIST(root="dataset/MNIST/data", train=True, download=False)
test = MNIST(root="dataset/MNIST/data", train=False, download=False)

x_train, y_train = np.array(train.data), np.array(train.targets)
x_test, y_test = np.array(test.data), np.array(test.targets)

x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. Compare four techniques
names = ["Xavier", "Kaiming", "Dropout", "L2"]
accuracies = []

for name in names:

    print("\nTraining", name)

    # Select initialization
    init = GlorotUniform() if name == "Xavier" else HeUniform()

    # Define model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu',
              kernel_initializer=init,
              kernel_regularizer=l2(0.0001) if name == "L2" else None),
        Dropout(0.3) if name == "Dropout" else tf.keras.layers.Activation('linear'),
        Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=1)

    # Test model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    accuracies.append(accuracy * 100)
    print(name, "Accuracy:", round(accuracy * 100, 2), "%")

# 3. Plot accuracy
plt.bar(names, accuracies)
plt.title("MNIST Accuracy Comparison")
plt.xlabel("Technique")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.savefig("mnist_optim_comparison.png")
print("Saved comparison chart to mnist_optim_comparison.png")
plt.show()