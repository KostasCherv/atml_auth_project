
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from modAL.models import ActiveLearner
from utils import get_initial_indexes

class AL_Process:
  def __init__(self, queries=10, instances_per_q=10, experiments=3):
    self.init_data()
    self.queries = queries
    self.instances = instances_per_q
    self.experiments = experiments

  def train(self, model):
    performance_history = []

    for i in tqdm(range(self.experiments)):
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

  def train_adaBoost(self, strategy):
    model = ActiveLearner(
      estimator = AdaBoostClassifier(random_state=0),
      X_training = self.X_initial.copy(), y_training = self.y_initial.copy(),
      query_strategy = strategy,
    )

    return self.train(model)


  def train_committee(self, strategy):
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

    return self.train(model)

  def init_data(self):
    # IMG_WIDTH = 28
    # IMG_HEIGHT = 28
    # CHANNELS=1
    # (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    IMG_WIDTH = 32
    IMG_HEIGHT = 32
    CHANNELS=3
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    selected_classes = np.unique(y_train)
    filtered_train_indexes = [i for i, v in enumerate(y_train.reshape(len(y_train))) if v in selected_classes]
    filtered_test_indexes = [i for i, v in enumerate(y_test.reshape(len(y_test))) if v in selected_classes]

    self.N_CLASSES= len(selected_classes)
    X_train = X_train[filtered_train_indexes]
    y_train = y_train[filtered_train_indexes]
    X_test = X_test[filtered_test_indexes]
    y_test = y_test[filtered_test_indexes]

    X_train = X_train.reshape((len(X_train), IMG_WIDTH * IMG_HEIGHT * CHANNELS))
    X_test = X_test.reshape((len(X_test), IMG_WIDTH * IMG_HEIGHT * CHANNELS))

    y_test = y_test.reshape((len(y_test)))
    y_train = y_train.reshape((len(y_train)))

    N_COMPONENTS = 100
    X_train = PCA(n_components=N_COMPONENTS).fit_transform(X_train, y_train)
    X_test = PCA(n_components=N_COMPONENTS).fit_transform(X_test, y_test)

    #normalizing 
    X_train = X_train.astype('float32') 
    X_test = X_test.astype('float32') 
    X_train = X_train / 255.0 
    X_test = X_test / 255.0

    n_initial= 5 * self.N_CLASSES
    initial_idx = get_initial_indexes(y_train, n_initial)
    self.X_initial = X_train[initial_idx]
    self.y_initial = y_train[initial_idx]

    self.X_pool = np.delete(X_train, initial_idx, axis=0)
    self.y_pool = np.delete(y_train, initial_idx, axis=0)

    self.X_test = X_test
    self.y_test = y_test

