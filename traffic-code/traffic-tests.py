import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])


    # # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    while True:
        # Get a compiled neural network
        # model = get_modeledx()
        print("--- main: get_modeledx8")
        model = get_modeledx8()

        # Fit model on training data
        model.fit(x_train, y_train, epochs=EPOCHS)

        # Evaluate neural network performance
        model.evaluate(x_test, y_test, verbose=2)



    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = list()
    labels = list()

    with os.scandir(data_dir) as subdirs:
        for sd in subdirs:
            if os.path.isdir(sd):
                with os.scandir(sd) as files:
                    for f in files:
                        path = os.path.join(sd, f.name)
                        # img = cv2.imread(path, cv2.IMREAD_COLOR)
                        img = cv2.imread(path) / 255.0
                        # img = cv2.imread(path)
                        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), 3)
                        images.append(img)
                        labels.append(sd.name)

    return (images, labels)

def get_modeledx():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation="relu"))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(200))
    # model.add(tf.keras.layers.Dense(200))
    # model.add(tf.keras.layers.Dense(200))
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
def get_modeledx2():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(200))
    # model.add(tf.keras.layers.Dense(200))
    model.add(tf.keras.layers.Dense(200))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    #     # 46s 92ms / step
    #     # accuracy: 0.9438
    return model

def get_model2(model_dir = None):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    if model_dir is None:

        model = tf.keras.models.Sequential()
        model.add.data_augmentation
        model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255))
        # Add an convolutional input layer, learn 64 filters using a 5x5 filter
        # This layer helps normalizing image variations
        model.add(tf.keras.layers.Conv2D(
            16, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
        # Normalize data
        model.add(tf.keras.layers.BatchNormalization())
        # Add a Max-pooling layer to reduce data size
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        # Add a Max-pooling layer to reduce data size
        # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))


        # Add a hidden layer deep learning to propgate erros to previous layer in order to update weights
        model.add(tf.keras.layers.Dense(16, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))


        # Add a layer to drop out small portion of the images to
        # avoid heavy reliance on certai area
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(16, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

        model.summary()

        # Flatten the nodes for neural network
        model.add(tf.keras.layers.Flatten())

        # Create the final output layer with NUM_CATEGORIE units
        model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid"))

        # model.summary()

        # compile neural network model
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
    else:
        model = tf.keras.models.load_model.model(model_dir)
    print(f"---get_model")

    return model

def get_modeledx3():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    print("--- modeledx3")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dense(200))
    model.add(tf.keras.layers.Dense(200))

    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    # 60s 120ms
    # 333 / 333 - 5s - loss: 0.0273 - accuracy: 0.9291
    return model

def get_modeledx4():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    print("--- modeledx4")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128, activation='softmax'))
    # model.add(tf.keras.layers.Dense(200))


    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    # 53s 106ms/step
    # 333 / 333 - 5s - loss: 0.0139 - accuracy: 0.9478
    return model

def get_modeledx5():
    print("--- get_modeledx5")
    model = tf.keras.models.Sequential()
    # Add an convolutional input layer, learn 64 filters using a 5x5 filter
    # This layer helps normalizing image variations
    model.add(tf.keras.layers.Conv2D(
        64, (5, 5), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Normalize data
    model.add(tf.keras.layers.BatchNormalization())

    # Add a Max-pooling layer to reduce data size
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # Add a hidden layer deep learning to propgate erros to previous layer in order to update weights
    model.add(tf.keras.layers.Dense(8, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Add a layer to drop out small portion of the images to
    # avoid heavy reliance on certai area
    model.add(tf.keras.layers.Dropout(0.5))

    # Flatten the nodes for neural network
    model.add(tf.keras.layers.Flatten())

    # Create the final output layer with NUM_CATEGORIE units
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # 22s 43ms/step
    # loss: 0.0281 - accuracy: 0.9698
    return model

def get_modeledx6():
    print("--- get_modeledx6: added a batchnormalize from edx5")
    model = tf.keras.models.Sequential()
    # Add an convolutional input layer, learn 64 filters using a 5x5 filter
    # This layer helps normalizing image variations
    model.add(tf.keras.layers.Conv2D(
        64, (5, 5), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Normalize data
    model.add(tf.keras.layers.BatchNormalization())

    # Add a Max-pooling layer to reduce data size
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # Add a hidden layer deep learning to propgate erros to previous layer in order to update weights
    model.add(tf.keras.layers.Dense(8, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Add a layer to drop out small portion of the images to
    # avoid heavy reliance on certai area
    model.add(tf.keras.layers.Dropout(0.5))

    # Normalize data
    model.add(tf.keras.layers.BatchNormalization())

    # Flatten the nodes for neural network
    model.add(tf.keras.layers.Flatten())

    # Create the final output layer with NUM_CATEGORIE units
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # batchnormalization: faster and lower loss rate. acc decreased less than 0.002
    # 19s 39ms/step
    # 2s - loss: 0.0130 - accuracy: 0.9669
    return model

def get_modeledx7():
    print("--- get_modeledx7: changed drop out rate")
    model = tf.keras.models.Sequential()
    # Add an convolutional input layer, learn 64 filters using a 5x5 filter
    # This layer helps normalizing image variations
    model.add(tf.keras.layers.Conv2D(
        64, (5, 5), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Normalize data
    model.add(tf.keras.layers.BatchNormalization())

    # Add a Max-pooling layer to reduce data size
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # Add a hidden layer deep learning to propgate erros to previous layer in order to update weights
    model.add(tf.keras.layers.Dense(8, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Add a layer to drop out small portion of the images to
    # avoid heavy reliance on certai area
    model.add(tf.keras.layers.Dropout(0.9))

    # Normalize data
    model.add(tf.keras.layers.BatchNormalization())

    # Flatten the nodes for neural network
    model.add(tf.keras.layers.Flatten())

    # Create the final output layer with NUM_CATEGORIE units
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # batchnormalization: droppout increased from 0.5 to 0.9:
    # loss increased to 0.0319 accuracy, decreased by 0.12 (12%)
    # 19s 38ms/step
    # 3s - loss: 0.0319 - accuracy: 0.08485
    return model

def get_modeledx8():
    print("--- get_modeledx8: changed from edx6: decreased dropout from 0.5 to 0.2")
    model = tf.keras.models.Sequential()
    # Add an convolutional input layer, learn 64 filters using a 5x5 filter
    # This layer helps normalizing image variations
    model.add(tf.keras.layers.Conv2D(
        64, (5, 5), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Normalize data
    model.add(tf.keras.layers.BatchNormalization())

    # Add a Max-pooling layer to reduce data size
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # Add a hidden layer deep learning to propgate erros to previous layer in order to update weights
    model.add(tf.keras.layers.Dense(8, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Add a layer to drop out small portion of the images to
    # avoid heavy reliance on certai area
    model.add(tf.keras.layers.Dropout(0.2))

    # Normalize data
    model.add(tf.keras.layers.BatchNormalization())

    # Flatten the nodes for neural network
    model.add(tf.keras.layers.Flatten())

    # Create the final output layer with NUM_CATEGORIE units
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # batchnormalization: with load img/255.0
    # 19s 37ms/step
    # 2s - loss: 0.0056 - accuracy: 0.9837
    return model
if __name__ == "__main__":
    main()
