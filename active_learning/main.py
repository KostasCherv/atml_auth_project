
from al_process import AL_Process
from strategies import random_sampling
from utils import plot_scores
from modAL.uncertainty import entropy_sampling, uncertainty_sampling
from modAL.expected_error import expected_error_reduction
from al_cnn_process import AL_CNN_Process
from dataset import Dataset



def train_process(queries=50, instances=1, experiments=3, ds='MNIST', classes='all'):
    chart_name = '_'.join([str(queries), str(instances), str(experiments), ds, str(classes)])
    process = AL_Process(queries=queries, instances=instances, experiments=experiments, dataset=Dataset[ds])

    _, ran = process.train_adaBoost(random_sampling)
    _, entr = process.train_adaBoost(entropy_sampling)
    _, uncent = process.train_adaBoost(uncertainty_sampling)
    _, com_ran = process.train_committee(random_sampling)
    _, com_entr = process.train_committee(entropy_sampling)
    _, com_uncent = process.train_committee(uncertainty_sampling)

    plot_scores(
        [ran, entr, uncent, com_ran, com_entr, com_uncent],
        ['Random', 'Entropy', 'Uncertainty', 'Committee Random', 'Committee Entropy', 'Committee Uncertainty'],
        chart_name
    )



def main(): 
    train_process(50, 1, 3, 'MNIST', classes=[1,2,3])
    train_process(50, 5, 3, 'MNIST', classes=[1,2,3])

    train_process(50, 1, 3, 'CIFAR10', classes=[1,2,3])
    train_process(50, 5, 3, 'CIFAR10', classes=[1,2,3])

    train_process(50, 1, 3, 'MNIST', classes='all')
    train_process(50, 5, 3, 'MNIST', classes='all')

    train_process(50, 1, 3, 'CIFAR10', classes='all')
    train_process(50, 5, 3, 'CIFAR10', classes='all')

if __name__ == '__main__':
    main()