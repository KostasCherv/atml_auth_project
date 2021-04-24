
from al_training import AL_Process
from strategies import random_sampling
from utils import plot_scores
from modAL.uncertainty import uncertainty_sampling, entropy_sampling
from al_cnn_training import AL_CNN_Process

process = AL_Process(50, 10, 5)

_, ran = process.train_adaBoost(random_sampling)
_, entr = process.train_adaBoost(entropy_sampling)
_, uncert = process.train_adaBoost(uncertainty_sampling)
_, com_ran = process.train_committee(random_sampling)
_, com_entr = process.train_committee(entropy_sampling)
_, com_uncent = process.train_committee(uncertainty_sampling)

plot_scores(
    [ran, entr, uncert, com_ran, com_entr, com_uncent],
    ['Random', 'Entropy', 'Uncertainty', 'Committee Random', 'Committee Entropy', 'Committee Uncertainty'])

process = (50, 10)

_, ran = process.train(random_sampling)
_, entr = process.train(entropy_sampling)
_, uncert = process.train(uncertainty_sampling)

plot_scores(
    [ran, entr, uncert],
    ['Random', 'Entropy', 'Uncertainty'])

