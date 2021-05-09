from enum import Enum
import tensorflow as tf

Dataset = {
  "MNIST": tf.keras.datasets.mnist,
  "CIFAR10": tf.keras.datasets.cifar10,
  "CIFAR100": tf.keras.datasets.cifar100,
  "FASHION_MNIST": tf.keras.datasets.fashion_mnist
}