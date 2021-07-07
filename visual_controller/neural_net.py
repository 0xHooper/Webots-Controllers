import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json

from sklearn.model_selection import train_test_split
from random import randint

DATA_SET_ROOT = 'photo/'


def load_data(data_dir):
    categories = os.listdir(data_dir)
    category_count = len(categories)

    X = []
    Y = []

    for category_id, category in enumerate(categories):
        data_points = os.listdir(os.path.join(data_dir, category))
        for data_point in data_points:
            x = cv2.imread(os.path.join(data_dir, category, data_point))
            X.append(x)
            y = np.zeros(category_count)
            y[category_id] = 1
            Y.append(y)

    return np.array(X), np.array(Y), categories


def build_and_compile_neural_network(input_shape, category_count):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(category_count, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


if __name__ == '__main__':
    X, Y, categories = load_data(DATA_SET_ROOT)
    print(f'X shape = {X.shape}')
    print(f'Y shape = {Y.shape}')
    for category_id, category in enumerate(categories):
        print(f' {category_id} ---> {category}')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    print(f'X_train shape = {X_train.shape}')
    print(f'Y_train shape = {Y_train.shape}')
    print(f'X_test shape = {X_test.shape}')
    print(f'Y_test shape = {Y_test.shape}')

    model = build_and_compile_neural_network(X.shape[1:], Y.shape[1])
    check_pointer = ModelCheckpoint(
        filepath="controlling_robot_model",
        verbose=1,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        mode='max'
    )

    history = model.fit(X_train, Y_train, batch_size=32, epochs=1000,
                        callbacks=[early_stop, check_pointer],
                        validation_data=(X_test, Y_test))
    # saving again to have model files in demanded extensions
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model_h5df.h5')

    plt.plot(history.history['accuracy'], label='train set')
    plt.plot(history.history['val_accuracy'], label='test set')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    for _ in range(20):
        image_idx = randint(0, X.shape[0])
        image = X[image_idx]
        category_id = np.argmax(model.predict(image.reshape(1, image.shape[0], image.shape[1], image.shape[2])))
        true_category_id = np.argmax(Y[image_idx])
        plt.title(f'{categories[category_id]}/{categories[true_category_id]}')
        plt.imshow(image)
        plt.show()
