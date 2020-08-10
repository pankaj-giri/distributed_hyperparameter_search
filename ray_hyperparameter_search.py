import numpy as np
import ray
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def generate_hyperparameters():
    return {'learning_rate': 10 ** np.random.uniform((-5, 5)),
            'batch_size': np.random.randint(4, 16),
            'dropout': np.random.uniform(0, 1),
            'epochs' : np.random.randint(1,5)
            }

@ray.remote
def train_cnn_and_compute_accuracy(hyperparameters,
                                   train_images,
                                   train_labels,
                                   validation_images,
                                   validation_labels):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(hyperparameters['dropout']),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=hyperparameters['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(x=train_images, y=train_labels, epochs=hyperparameters['epochs'])
    return history.history


ray.init()
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = np.expand_dims(train_images, axis=3)
print(train_images.shape, train_labels.shape)

train_images = ray.put(train_images)
train_labels = ray.put(train_labels)

hyperparm_config = [generate_hyperparameters() for _ in range(3)]

results = []
for hyperparameters in hyperparm_config:
    results.append(train_cnn_and_compute_accuracy.remote(hyperparameters,
                                                                  train_images,
                                                                  train_labels,
                                                                  None,
                                                                  None))
print(ray.get(results))