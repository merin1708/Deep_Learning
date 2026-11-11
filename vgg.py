import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Resizing, Lambda, Flatten, Dense
from tensorflow.keras.applications import VGG19
from torchvision.datasets import MNIST

# 1. Load dataset
train = MNIST("dataset/MNIST/data", train=True, download=False)
test = MNIST("dataset/MNIST/data", train=False, download=False)

x_train = train.data.numpy()
y_train = np.array(train.targets)

x_test = test.data.numpy()
y_test = np.array(test.targets)

# 2. Preprocess images
x_train = np.repeat(x_train[..., None], 3, axis=-1) / 255.0
x_test = np.repeat(x_test[..., None], 3, axis=-1) / 255.0

# 3. Load pretrained VGG19
vgg = VGG19(weights='imagenet', include_top=False,
            input_shape=(32, 32, 3))
vgg.trainable = False

# 4. Build model
model = Sequential([
    Resizing(32, 32),
    Lambda(lambda x: tf.keras.applications.vgg19.preprocess_input(x * 255)),
    vgg,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 5. Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train model
model.fit(x_train, y_train, epochs=5, batch_size=64)

# 7. Test model
loss, acc = model.evaluate(x_test, y_test)
print("Accuracy:", acc * 100, "%")

# 8. Display predictions
for i in range(10):
    pred = model.predict(x_test[i:i+1], verbose=0).argmax()
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i])
    plt.title(f"Actual:{y_test[i]}\nPred:{pred}")
    plt.axis('off')

plt.show()