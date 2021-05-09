
from al_process import AL_Process
from strategies import random_sampling
from utils import plot_scores
from modAL.uncertainty import entropy_sampling, uncertainty_sampling
from modAL.expected_error import expected_error_reduction
from al_cnn_process import AL_CNN_Process
from dataset import Dataset


process = AL_Process(queries=50, instances=1, experiments=3, dataset=Dataset['MNIST'])

_, ran = process.train_adaBoost(random_sampling)
_, entr = process.train_adaBoost(entropy_sampling)
_, uncent = process.train_adaBoost(uncertainty_sampling)
_, com_ran = process.train_committee(random_sampling)
_, com_entr = process.train_committee(entropy_sampling)
_, com_uncent = process.train_committee(uncertainty_sampling)

plot_scores(
    [ran, entr, uncent, com_ran, com_entr, com_uncent],
    ['Random', 'Entropy', 'Uncertainty', 'Committee Random', 'Committee Entropy', 'Committee Uncertainty'])


# process = AL_CNN_Process(queries=50, instances=10)

# _, ran = process.train(random_sampling)
# _, entr = process.train(entropy_sampling)
# _, uncert = process.train(uncertainty_sampling)

# plot_scores(
#     [ran, entr, uncert],
#     ['Random', 'Entropy', 'Uncertainty'])

