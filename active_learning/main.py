
from al_process import AL_Process
from strategies import random_sampling
from utils import plot_scores
from modAL.uncertainty import entropy_sampling, uncertainty_sampling
from al_cnn_process import AL_CNN_Process
from dataset import Dataset



def train_process(queries,instances,experiments,initial,ds, classes):
    chart_name = '_'.join(['ADA', str(queries), str(instances), str(experiments), str(initial), ds, str(classes)])
    process = AL_Process(queries=queries, instances=instances, experiments=experiments, n_initial=initial,  dataset=Dataset[ds], classes=classes)

    _, ran = process.train_adaBoost(random_sampling)
    _, entr = process.train_adaBoost(entropy_sampling)
    _, uncent = process.train_adaBoost(uncertainty_sampling)
    _, com_ran = process.train_committee(random_sampling)
    _, com_entr = process.train_committee(entropy_sampling)
    _, com_uncent = process.train_committee(uncertainty_sampling)

    plot_scores(
        [ran, entr, uncent, com_ran, com_entr, com_uncent],
        ['Random', 'Entropy', 'Uncertainty','Committee Random', 'Committee Entropy', 'Committee Uncertainty'],
        chart_name
    )


def cnn_train_process(queries,instances,experiments,initial,ds,classes):
    chart_name = '_'.join(['CNN', str(queries), str(instances), str(experiments), str(initial), ds, str(classes)])
    process = AL_CNN_Process(queries=queries, instances=instances, experiments=experiments, n_initial=initial,  dataset=Dataset[ds], classes=classes)

    _, ran = process.train(random_sampling)
    _, entr = process.train(entropy_sampling)
    _, uncent = process.train(uncertainty_sampling)

    plot_scores(
        [ran, entr, uncent,],
        ['Random', 'Entropy', 'Uncertainty'],
        chart_name
    )


def main(): 
    # train_process(queries=50, instances=20, experiments=5, initial=100, ds='FASHION_MNIST', classes=[1,2,3])
    # train_process(queries=50, instances=20, experiments=5, initial=500, ds='FASHION_MNIST', classes=[1,2,3])

    # train_process(queries=50, instances=50, experiments=5, initial=100, ds='FASHION_MNIST', classes='all')
    # train_process(queries=50, instances=50, experiments=5, initial=500, ds='FASHION_MNIST', classes='all')

    # train_process(queries=50, instances=20, experiments=5, initial=100, ds='MNIST', classes=[1,2,3])
    # train_process(queries=50, instances=20, experiments=5, initial=500, ds='MNIST', classes=[1,2,3])

    # train_process(queries=50, instances=50, experiments=5, initial=100, ds='MNIST', classes='all')
    # train_process(queries=50, instances=50, experiments=5, initial=500, ds='MNIST', classes='all')

    # train_process(queries=50, instances=20, experiments=5, initial=100, ds='CIFAR10', classes=[1,2,3])
    # train_process(queries=50, instances=20, experiments=5, initial=500, ds='CIFAR10', classes=[1,2,3])

    # train_process(queries=50, instances=50, experiments=5, initial=100, ds='CIFAR10', classes='all')
    # train_process(queries=50, instances=50, experiments=5, initial=500, ds='CIFAR10', classes='all')

    # cnn_train_process(queries=50, instances=20, experiments=5, initial=100, ds='CIFAR10', classes=[0,1,2])
    # cnn_train_process(queries=50, instances=20, experiments=5, initial=500, ds='CIFAR10', classes=[0,1,2])

    # cnn_train_process(queries=50, instances=50, experiments=5, initial=100, ds='CIFAR10', classes='all')
    # cnn_train_process(queries=50, instances=50, experiments=5, initial=500, ds='CIFAR10', classes='all')

    cnn_train_process(queries=50, instances=20, experiments=5, initial=100, ds='FASHION_MNIST', classes=[0,1,2])
    cnn_train_process(queries=50, instances=20, experiments=5, initial=500, ds='FASHION_MNIST', classes=[0,1,2])

    # cnn_train_process(queries=50, instances=50, experiments=5, initial=100, ds='FASHION_MNIST', classes='all')
    # cnn_train_process(queries=50, instances=50, experiments=5, initial=500, ds='FASHION_MNIST', classes='all')

    # cnn_train_process(queries=50, instances=20, experiments=5, initial=100, ds='MNIST', classes=[0,1,2])
    # cnn_train_process(queries=50, instances=20, experiments=5, initial=500, ds='MNIST', classes=[0,1,2])

    # cnn_train_process(queries=50, instances=50, experiments=5, initial=100, ds='MNIST', classes='all')
    # cnn_train_process(queries=50, instances=50, experiments=5, initial=500, ds='MNIST', classes='all')

if __name__ == '__main__':
    main()