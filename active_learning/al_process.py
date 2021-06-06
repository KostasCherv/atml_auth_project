
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from modAL.models import ActiveLearner, Committee
from utils import get_initial_indexes
from dataset import Dataset

class AL_Process:
  def __init__(self, queries=10, instances=10, experiments=3, n_initial=100, classes='all', dataset=Dataset['CIFAR10']):
    self.queries=queries
    self.instances=instances
    self.experiments=experiments
    self.classes=classes
    self.dataset=dataset
    self.n_initial=n_initial

    self.load_data()

  def train_adaBoost(self, strategy):
    
    performance_history = []

    for i in tqdm(range(self.experiments)):
      h=[]
      X = self.X_pool.copy()
      y = self.y_pool.copy()
      model = ActiveLearner(
        estimator = AdaBoostClassifier(),
        X_training = self.X_initial.copy(), y_training = self.y_initial.copy(),
        query_strategy = strategy,
      )

      for idx in tqdm(range(self.queries)):
        query_idx, _ = model.query(X, n_instances=self.instances)
        model.teach(X=X[query_idx], y=y[query_idx])
        acc = model.score(self.X_test, self.y_test)
        h.append(acc)

        # remove queried instance from pool
        X = np.delete(X, query_idx, axis=0)
        y = np.delete(y, query_idx, axis=0)
      performance_history.append(h)

    return model, np.mean(performance_history, axis=0)


  def train_committee(self, strategy):
    performance_history = []

    for i in tqdm(range(self.experiments)):
      learner_1 = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=strategy,
        X_training=self.X_initial.copy(), y_training=self.y_initial.copy()
      )

      learner_2 = ActiveLearner(
        estimator=AdaBoostClassifier(),
        query_strategy=strategy,
        X_training=self.X_initial.copy(), y_training=self.y_initial.copy()
      )

      model = Committee(learner_list=[learner_1, learner_2])
      h=[]
      X = self.X_pool.copy()
      y = self.y_pool.copy()

      for idx in tqdm(range(self.queries)):
        query_idx, _ = model.query(X, n_instances=self.instances)
        model.teach(X=X[query_idx], y=y[query_idx])
        acc = model.score(self.X_test, self.y_test)
        h.append(acc)

        # remove queried instance from pool
        X = np.delete(X, query_idx, axis=0)
        y = np.delete(y, query_idx, axis=0)
      
      performance_history.append(h)

    return model, np.mean(performance_history, axis=0)

  def load_data(self):
    (X_train, y_train), (X_test, y_test) = self.dataset.load_data()
    X_train = X_train
    y_train = y_train
    self.IMG_WIDTH = X_train.shape[1]
    self.IMG_HEIGHT = X_train.shape[2]
    self.CHANNELS = 1 if len(X_train.shape) == 3 else X_train.shape[3]
    selected_classes = np.unique(y_train) if self.classes == 'all' else self.classes

    filtered_train_indexes = [i for i, v in enumerate(y_train.reshape(len(y_train))) if v in selected_classes]
    filtered_test_indexes = [i for i, v in enumerate(y_test.reshape(len(y_test))) if v in selected_classes]

    self.N_CLASSES= len(selected_classes)
    X_train = X_train[filtered_train_indexes]
    y_train = y_train[filtered_train_indexes]
    X_test = X_test[filtered_test_indexes]
    y_test = y_test[filtered_test_indexes]

    X_train = X_train.reshape((len(X_train), self.IMG_WIDTH * self.IMG_HEIGHT * self.CHANNELS))
    X_test = X_test.reshape((len(X_test), self.IMG_WIDTH * self.IMG_HEIGHT * self.CHANNELS))

    y_test = y_test.reshape((len(y_test)))
    y_train = y_train.reshape((len(y_train)))

    N_COMPONENTS = 50
    X_train = PCA(n_components=N_COMPONENTS).fit_transform(X_train, y_train)
    X_test = PCA(n_components=N_COMPONENTS).fit_transform(X_test, y_test)

    #normalizing 
    X_train = X_train.astype('float32') 
    X_test = X_test.astype('float32') 
    X_train = X_train / 255.0 
    X_test = X_test / 255.0

    n_initial= self.n_initial
    print(F'NUMBER OF INITIAL SAMPLES: {n_initial}')
    initial_idx = get_initial_indexes(y_train, n_initial)
    self.X_initial = X_train[initial_idx]
    self.y_initial = y_train[initial_idx]

    self.X_pool = np.delete(X_train, initial_idx, axis=0)
    self.y_pool = np.delete(y_train, initial_idx, axis=0)

    self.X_test = X_test
    self.y_test = y_test

