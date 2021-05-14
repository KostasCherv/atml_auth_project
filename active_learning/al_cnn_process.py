import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from modAL.models import ActiveLearner
from utils import get_initial_indexes
from tensorflow.keras.utils import to_categorical
from dataset import Dataset

class AL_CNN_Process:
  def __init__(self, queries=10, instances=10, experiments=3, n_initial=100, classes='all', dataset=Dataset['CIFAR10']):
    self.queries=queries
    self.instances=instances
    self.experiments=experiments
    self.classes=classes
    self.dataset=dataset
    self.n_initial=n_initial

    self.load_data()

  def train(self, strategy):
    performance_history = []

    for i in tqdm(range(self.experiments)):
      model = ActiveLearner(
        estimator = KerasClassifier(self.get_model),
        X_training = self.X_initial.copy(), y_training = self.y_initial.copy(),
        query_strategy = strategy,
      )

      h=[]
      X = self.X_pool.copy()
      y = self.y_pool.copy()
      
      for idx in tqdm(range(self.queries)):
        query_idx, query_instance = model.query(X, n_instances=self.instances)
        model.teach(X=X[query_idx], y=y[query_idx])
        acc = model.score(self.X_test, self.y_test)
        h.append(acc)

        # remove queried instance from pool
        X = np.delete(X, query_idx, axis=0)
        y = np.delete(y, query_idx, axis=0)
      performance_history.append(h)

    return model, np.mean(performance_history, axis=0)


  def get_model(self):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.IMG_WIDTH, self.IMG_HEIGHT, self.CHANNELS)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(self.N_CLASSES, activation='softmax'))

    opt = Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

  def load_data(self):
    (X_train, y_train), (X_test, y_test) = self.dataset.load_data()
    self.IMG_WIDTH = X_train.shape[1]
    self.IMG_HEIGHT = X_train.shape[2]
    self.CHANNELS = 1 if len(X_train.shape) == 3 else X_train.shape[3]
    selected_classes = np.unique(y_train) if self.classes == 'all' else self.classes

    filtered_train_indexes = [i for i, v in enumerate(y_train.reshape(len(y_train))) if v in selected_classes]
    filtered_test_indexes = [i for i, v in enumerate(y_test.reshape(len(y_test))) if v in selected_classes]

    self.N_CLASSES = len(selected_classes)
    X_train = X_train[filtered_train_indexes]
    y_train = y_train[filtered_train_indexes]
    X_test = X_test[filtered_test_indexes]
    y_test = y_test[filtered_test_indexes]

    X_train = X_train.reshape((len(X_train), self.IMG_WIDTH, self.IMG_HEIGHT, self.CHANNELS))
    X_test = X_test.reshape((len(X_test), self.IMG_WIDTH, self.IMG_HEIGHT, self.CHANNELS))

    #normalizing 
    X_train = X_train.astype('float32') 
    X_test = X_test.astype('float32') 
    X_train = X_train / 255.0 
    X_test = X_test / 255.0

    n_initial = self.n_initial
    initial_idx = get_initial_indexes(y_train, n_initial)
    
    y_train = to_categorical(y_train, self.N_CLASSES)
    y_test = to_categorical(y_test, self.N_CLASSES)

    self.X_initial = X_train[initial_idx]
    self.y_initial = y_train[initial_idx]

    self.X_pool = np.delete(X_train, initial_idx, axis=0)
    self.y_pool = np.delete(y_train, initial_idx, axis=0)

    self.X_test = X_test
    self.y_test = y_test


