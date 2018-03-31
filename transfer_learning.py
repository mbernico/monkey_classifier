# Transfer Learning Example using 10 Monkey Species Dataset
# https://www.kaggle.com/slothkong/10-monkey-species
# Author: Mike Bernico @mikebernico mike.bernico@gmail.com

# these seeds are both required for reproducibility
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import os
from transfer_learning_config import Configuration


def build_model_feature_extraction():
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # make sure we're back at a 2d tensor
    x = Dense(1024, activation='relu')(x)  # add one fully connected layer
    predictions = Dense(10, activation='softmax')(x)  # 10 monkey, therefore 10 output units w softmax activation

    model = Model(inputs=base_model.input, outputs=predictions)

    # make the base model untrainable (frozen)
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_model_fine_tuning(model, learning_rate=0.0001, momentum=0.9):

        for layer in model.layers[:249]:
            layer.trainable = False
        for layer in model.layers[249:]:
            layer.trainable = True
        model.compile(optimizer=SGD(lr=learning_rate, momentum=momentum), loss='categorical_crossentropy', metrics=['accuracy'])
        return model


def create_callbacks(name):
    tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tensorboard_log", name), write_graph=True, write_grads=False)
    checkpoint_callback = ModelCheckpoint(filepath="./model-weights-ls" + name + ".{epoch:02d}-{val_loss:.6f}.hdf5", monitor='val_loss',
                                          verbose=0, save_best_only=True)
    return [tensorboard_callback, checkpoint_callback]


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
    model = build_model_feature_extraction()
    model = fit_model(model, train_generator, val_generator,
                      batch_size=config.batch_size,
                      epochs=config.feature_extraction_epochs,
                      name="feature_extraction")

    print("Feature Extraction Complete.")
    eval_model(model, val_generator, batch_size=config.batch_size)

    # Fine Tuning
    model = build_model_fine_tuning(model)
    model = fit_model(model, train_generator, val_generator,
                      batch_size=config.batch_size,
                      epochs=config.fine_tuning_epochs,
                      name="fine_tuning")

    print("Fine Tuning Complete.")
    eval_model(model, val_generator, batch_size=config.batch_size)

    print("Saving Model...")
    model.save("transfer_learning_model.h5")


if __name__ == "__main__":
    main()


