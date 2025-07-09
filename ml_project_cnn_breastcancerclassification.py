# ml_project_cnn_breastcancerclassification.py


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt

def cnn_builder(hp):

    model = Sequential()

    hp_filters1 = hp.Int('filters1',min_value=32,max_value=128,step=8)
    hp_kernelrow = hp.Int('kernel_row',min_value=2,max_value=8,step=1)
    hp_kernelcol = hp.Int('kernel_col',min_value=2,max_value=8,step=1)

    model.add(Conv2D(filters = hp_filters1, kernel_size = (hp_kernelrow,hp_kernelcol), activation = 'relu', input_shape = (IMG_SIZE, IMG_SIZE, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    hp_filters2 = hp.Int('filters2',min_value=32,max_value=128,step=8)
    model.add(Conv2D(filters = hp_filters2, kernel_size = (hp_kernelrow,hp_kernelcol), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    hp_regularization = hp.Float('regularization',min_value=0.0,max_value=0.5,step=0.1)
    hp_units = hp.Int('units',min_value=64,max_value=512,step=8)
    model.add(Dense(units=hp_units, activation='relu', kernel_regularizer= keras.regularizers.l2(hp_regularization)))

    model.add(Dense(1,activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate',values=[1e-2,1e-3,1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),loss='binary_crossentropy',metrics=['accuracy'])

    return model

IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8,1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_dir = '/content/drive/MyDrive/DataSets/Breast Cancer/train'
val_dir = '/content/drive/MyDrive/DataSets/Breast Cancer/test'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

tuner = kt.Hyperband(
    cnn_builder,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='/content/TunerLogs',
    project_name='Breast Cancer Classification'
)

tuner.search(train_generator, epochs=10, validation_data=val_generator)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples //BATCH_SIZE,
    epochs=30,
    validation_data=val_generator,
    validation_steps=val_generator.samples //BATCH_SIZE,
    callbacks = [EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)]
)

loss, accuracy = model.evaluate(val_generator, steps=val_generator.samples // BATCH_SIZE )
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

keras.models.save_model(model,"BreastCancerClassifier.keras")