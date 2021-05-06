
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Active Learning application on Object Classification</h3>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This application of active learning uses the modAL python library to train a Deep Neural Network model and an AdaBoost model on the Cifar-10 dataset. So it is a pool-based application with the usage of entropy, uncertainty, and committee query strategies compared to random sampling.

The process will show comparing charts between the scores of the different strategies and the progress in time.

### Built With
* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [modAL](https://modal-python.readthedocs.io/en/latest)

### Installation
For the installation, it is better to use the pipenv tool. This will install all the necessary dependencies.

 ``pipenv install``

<!-- USAGE EXAMPLES -->
## Usage
To start the whole process you can run the main.py file.

``python main.py``




