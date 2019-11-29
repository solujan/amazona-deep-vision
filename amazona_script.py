import tensorflow as tf
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator  
from keras.applications import densenet  
from keras.models import Sequential, Model, load_model  
from keras.layers import Conv2D, MaxPooling2D  
from keras.layers import Activation, Dropout, Flatten, Dense  
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback  
from keras import regularizers  
from keras import backend as K  
from tqdm import tqdm
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, VGG16
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, "0", "1", etc.
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    
class DataPreparation:

    def __init__(self, file_name):
        self.file_name = file_name
        
    def load_file(self):
        dataset = pd.read_pickle(self.file_name)
        labels = list(dataset.columns.values[2:])
        print("Loaded dataset shape: ", dataset.shape)
        #dataset = dataset[:10000]
        #print("New dataset shape: ", dataset.shape)
        return dataset, labels
    
class Generators:
    def __init__(self, dataset, labels, batch_size, img_size):
        self.batch_size=batch_size
        self.img_size=(img_size, img_size)
        
        _datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True, 
            vertical_flip=True,
            validation_split=0.2)
        
        self.train_generator = _datagen.flow_from_dataframe(
            dataframe=dataset,
            directory='',
            x_col="image_name",
            y_col="tags",
            classes=labels,
            target_size=self.img_size,
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            subset='training')
        print('Train generator created')
        # Validation generator
        self.val_generator = _datagen.flow_from_dataframe(
            dataframe=dataset,
            directory='',
            x_col="image_name",
            y_col="tags",
            classes=labels,
            target_size=self.img_size,
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            subset='validation')    
        print('Validation generator created')
        
    
class ModelTrainer:
    
    def __init__(self, generators):
        self.generators = generators
        self.img_width = generators.img_size[0]
        self.img_height = generators.img_size[1]
        
    def create_basic(self):
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(self.img_width, self.img_height,3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        return model
      
    def create_resnet50_weights(self):
        base_model = ResNet50(input_shape=(self.img_width, self.img_height, 3),  
                      weights='imagenet',                           
                      include_top=False,                            
                      pooling='avg')
        
        x = base_model.output
        local_dense = Dense(1024, activation='relu',name='local_dense')(x)
        dropout = Dropout(0.3)(local_dense)
        # and a logistic layer
        predictions = Dense(17, activation='sigmoid',name='local_output')(dropout)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        
        for layer in model.layers[:99]:
            layer.trainable = False
        for layer in model.layers[99:]:
            layer.trainable = True

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])            
        return model
        
    def create_resnet50(self):
        optimizer = Adam(0.003, decay=0.0005)
        base_model = ResNet50(weights=None, include_top=False, input_shape=(self.img_width, self.img_height, 3))

        for layer in base_model.layers:
            layer.trainable = True

        model = Sequential([
            base_model,
            Flatten(), 
            Dense(128, activation='relu',name='local_dense'),
            Dropout(0.2),
            Dense(17, activation='sigmoid',name='local_output')
            
        ])
        
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
          
        return model

    def create_vgg16(self):
        optimizer = Adam(0.003, decay=0.0005)
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(self.img_width, self.img_height, 3))

        for layer in base_model.layers:
            layer.trainable = False
            
        model = Sequential([
            base_model,
            Flatten(), 
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(17, activation='sigmoid')  
        ])

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model

    def add_callback(self, exp_name):
        if not os.path.exists(exp_name):
          os.makedirs(exp_name)
        self.callbacks = [
            EarlyStopping(monitor='val_loss', 
                          min_delta=0, 
                          patience=2, 
                          verbose=1, 
                          mode='auto', 
                          baseline=None, 
                          restore_best_weights=False),
            ModelCheckpoint(filepath=os.path.join(exp_name, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                    monitor='val_loss', 
                    verbose=1, 
                    save_best_only=False, 
                    save_weights_only=False, 
                    mode='auto')
        ]

    def train_model(self, model, epochs):
        model_history = model.fit_generator(
                        self.generators.train_generator,
                        steps_per_epoch = self.generators.val_generator.samples // self.generators.batch_size,
                        validation_data = self.generators.val_generator, 
                        validation_steps = self.generators.val_generator.samples // self.generators.batch_size,
                        epochs=epochs,
                        callbacks=self.callbacks,
                        verbose=1)
        return model_history


class Evaluator:
     
    def __init__(self, training):
        self.training = training

    def plot_history(self):
        ## Trained model analysis and evaluation
        f, ax = plt.subplots(1,2, figsize=(12,3))
        ax[0].plot(self.training.history['loss'], label="Loss")
        ax[0].plot(self.training.history['val_loss'], label="Validation loss")
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        # Accuracy
        ax[1].plot(self.training.history['acc'], label="Accuracy")
        ax[1].plot(self.training.history['val_acc'], label="Validation accuracy")
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        plt.tight_layout()
        plt.show()
    

def main():
    # Train/test/validation split with balanced labels in train
    data_prep = DataPreparation("train-jpg-labels.pkl")
    dataset, labels = data_prep.load_file()
    
    IMG_SIZE = 256
    BATCH_SIZE = 128
    EPOCHS = 5
    NET_IMG_ROWS = 128
    NET_IMG_COLS = 128

    # Create generators        
    generators = Generators(dataset, labels,BATCH_SIZE, IMG_SIZE)
    print("Generators created")
    
    trainer = ModelTrainer(generators)
    model = trainer.create_vgg16()
    trainer.add_callback('vgg16_weights')
    print(model.summary())
    model_history = trainer.train_model(model, EPOCHS)
    
    # Create evaluator instance
    evaluator = Evaluator(model_history)
    # Draw accuracy and loss charts
    evaluator.plot_history()
	
if __name__ == "__main__":
	main()	
	
