''' Model Training code || Trained using GPU in in Colab '''

# importing required libraries
import keras

# defining model parameters
NUM_CLASSES = 5
IMG_ROWS, IMG_COLS = 48, 48
BS = 10

# training & validation dataset
data_path = '/content/drive/MyDrive/ML Projects/Facial-Expression-Detection/CK+48'

from keras.preprocessing.image import ImageDataGenerator

# performing data augmentation on the training dataset
datagen = ImageDataGenerator(rescale=1 / 128.,
                            validation_split = 0.3)


train_data = datagen.flow_from_directory(data_path,
                                        color_mode='grayscale',
                                        target_size = (IMG_ROWS, IMG_COLS),
                                        batch_size = BS,
                                        class_mode = 'categorical',
                                        subset = 'training', 
                                        shuffle = True)

validation_data = datagen.flow_from_directory(data_path,
                                            color_mode='grayscale',
                                            target_size =(IMG_ROWS, IMG_COLS),
                                            batch_size = BS,
                                            class_mode = 'categorical', 
                                            subset = 'validation',
                                            shuffle = False)

# dislaying the class indices
train_data.class_indices

# defining the NeuralNet architecture
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization


model = Sequential()

model.add(Conv2D(64, (3,3),
          activation='elu',
          padding='same',
          kernel_initializer='he_normal',
          input_shape=(IMG_ROWS, IMG_COLS, 1),
          name='conv2d_1'))

model.add(BatchNormalization(name='batchnorm_1'))

model.add(Conv2D(64, (3,3),
          activation='elu',
          padding='same',
          kernel_initializer='he_normal',
          name='conv2d_2'))

model.add(BatchNormalization(name='batchnorm_2'))

model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1'))
model.add(Dropout(0.3, name='dropout_1'))

model.add(Conv2D(128, (3,3),
          activation='elu',
          padding='same',
          kernel_initializer='he_normal',
          name='conv2d_3'))

model.add(BatchNormalization(name='batchnorm_3'))

model.add(Conv2D(128, (3,3),
          activation='elu',
          padding='same',
          kernel_initializer='he_normal',
          name='conv2d_4'))

model.add(BatchNormalization(name='batchnorm_4'))

model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2'))
model.add(Dropout(0.3, name='dropout_2'))

model.add(Conv2D(256, (3,3),
          activation='elu',
          padding='same',
          kernel_initializer='he_normal',
          name='conv2d_5'))

model.add(BatchNormalization(name='batchnorm_5'))

model.add(Conv2D(256, (3,3),
          activation='elu',
          padding='same',
          kernel_initializer='he_normal',
          name='conv2d_6'))

model.add(BatchNormalization(name='batchnorm_6'))

model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_3'))
model.add(Dropout(0.3, name='dropout_3'))

model.add(Flatten(name='flatten'))

model.add(Dense(128,
                activation='elu',
                kernel_initializer='he_normal',
                name='dense1'))

model.add(BatchNormalization(name='batchnorm_7'))
model.add(Dropout(0.4, name='dropout_4'))

model.add(Dense(NUM_CLASSES,
                activation='softmax',
                name='output_layer'))

print(model.summary())

from tensorflow.keras import optimizers

optim = optimizers.Adam(0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optim,
              metrics=['accuracy'])

# plotting the model
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='/content/drive/MyDrive/ML Projects/Facial-Expression-Detection/final_architecture.jpg')

from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss',
                                min_delta=0.00008,
                                patience=12,
                                verbose=1,
                                restore_best_weights=True)

lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy',
                                  min_delta=0.0001,
                                  factor=0.4,
                                  patience=6,
                                  min_lr=1e-7,
                                  verbose=1)

model_checkpoint = ModelCheckpoint('/content/drive/MyDrive/ML Projects/Facial-Expression-Detection/finalModel.h5', 
                                    monitor='val_loss', save_best_only = True, verbose=1)

CALLBACKS = [early_stopping, lr_scheduler, model_checkpoint]
EPOCHS = 60

# fitting the model on the training data
history = model.fit_generator(generator=train_data,
                              steps_per_epoch=train_data.n//train_data.batch_size,
                              epochs=EPOCHS,
                              validation_data = validation_data,validation_steps=validation_data.n//validation_data.batch_size,
                              callbacks = CALLBACKS)  

## **loss: 0.0073 - accuracy: 0.9994 - val_loss: 0.0958 - val_accuracy: 0.9636**

# visualising model training
import matplotlib.pyplot as plt
import numpy as np

N = 60
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss & Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/ Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

