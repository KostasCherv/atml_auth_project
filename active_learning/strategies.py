import numpy as np

def random_sampling(classifier, X, n_instances: int = 1, **uncertainty_measure_kwargs):
    """
    Random sampling query strategy.
    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
            measure function.
    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    
    query_idx = np.random.choice(range(len(X)), size=n_instances, replace=False)
    
    return query_idx, X[query_idx]