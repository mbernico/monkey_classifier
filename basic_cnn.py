# Using 10 Monkey Species Dataset without Transfer Learning
# https://www.kaggle.com/slothkong/10-monkey-species
# Author: Mike Bernico @mikebernico mike.bernico@gmail.com

# these seeds are both required for reproducibility
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Input
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import os
from transfer_learning_config import Configuration


def build_model():
    inputs = Input(shape=(299,299,3), name="input")

    # convolutional block 1
    conv1 = Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv_1")(inputs)
    batch1 = BatchNormalization(name="batch_norm_1")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool_1")(batch1)

    # convolutional block 2
    conv2 = Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv_2")(pool1)
    batch2 = BatchNormalization(name="batch_norm_2")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool_2")(batch2)

    # fully connected layers
    flatten = Flatten()(pool2)
    fc1 = Dense(1024, activation="relu", name="fc1")(flatten)

    # output layer
    output = Dense(10, activation="softmax", name="softmax")(fc1)

    # finalize and compile
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    return model


def create_callbacks(name):
    tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tensorboard_log", name), write_graph=True, write_grads=False)
    checkpoint_callback = ModelCheckpoint(filepath="./model-weights-" + name + ".{epoch:02d}-{val_loss:.6f}.hdf5", monitor='val_loss',
                                          verbose=0, save_best_only=True)
    return [tensorboard_callback]


def setup_data(train_data_dir, val_data_dir, img_width=299, img_height=299, batch_size=16):
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    return train_generator, validation_generator


def fit_model(model, train_generator, val_generator, batch_size, epochs, name):
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.n // batch_size,
        callbacks=create_callbacks(name=name),
        verbose=1)
    return model


def eval_model(model, val_generator, batch_size):
    scores = model.evaluate_generator(val_generator, steps=val_generator.n // batch_size)
    print("Loss: " + str(scores[0]) + " Accuracy: " + str(scores[1]))


def main():
    config = Configuration()
    train_generator, val_generator = setup_data(config.data_dir, config.val_dir, batch_size=config.batch_size)

    # Feature Extraction
    model = build_model()
    model = fit_model(model, train_generator, val_generator,
                      batch_size=config.batch_size,
                      epochs=config.epochs_without_transfer_learning,
                      name="without_transfer_learning")

    print("Training Complete.")
    eval_model(model, val_generator, batch_size=config.batch_size)

    print("Saving Model...")
    model.save("no_transfer_learning_model.h5")


if __name__ == "__main__":
    main()


