'''
This script provides IVFS algorithm.
'''

import numpy as np
from random import sample
import random
from scipy.spatial.distance import pdist


def IVFS(X, subsets=1000, num_features=1, num_samples=1, score_type='infinity', seed=0):
    """
    Inclusion Value Feature Selection Algorithm

    Parameters
    ----------
    X: DataFrame n*d
    Original covariate data.
    subsets: int
    Number of subsets.
    num_features: int
    Number of features used for each subset.
    num_samples: int
    Number of samples for each subset.
    score_type: string from 'infinity', 'L1' or 'L2'
    The norm of loss or score function

    Returns
    -------
    score (float) np.array: score of each feature
    """

    n, d = X.shape

    # Check the usage condition of the algorithm
    if num_samples > n:
        # print('The number of samples for each subset is more than the total number of data')
        num_samples = n

    if num_features > d:
        # print('The number of features used for each subset is more than the total number of features')
        num_features = d

    # Calculate norms between distance matrix, or the score function
    def cal_score(X1, X2, score_type='infinity'):
        X1 = np.array(X1)
        X2 = np.array(X2)
        D1 = pdist(X1, metric='euclidean')
        D2 = pdist(X2, metric='euclidean')

        # Normalize all distances to [0,1]
        D1 = D1 / np.max(D1)
        D2 = D2 / np.max(D2)

        if score_type == 'infinity':
            return np.linalg.norm(D1 - D2, ord=np.inf)
        elif score_type == 'L1':
            return np.linalg.norm(D1 - D2, ord=1)
        elif score_type == 'L2':
            return np.linalg.norm(D1 - D2, ord=2)

    random.seed(seed)

    # Counters for each feature
    C = np.zeros(d)
    # Cumulative score for each feature
    S = np.zeros(d)

    for i in range(subsets):
        # Randomly sample a size num_features feature setF
        sub_f = sample(range(d), num_features)
        # Randomly sub-sample, with num_samples observations and features in F
        sub_s = sample(range(n), num_samples)

        # Update counter
        C[sub_f] += 1
        # Update score
        S[sub_f] -= cal_score(X.iloc[sub_s, :], X.iloc[sub_s, sub_f], score_type=score_type)

        '''
        if (i + 1) % 10 == 0:
            print('Step [{}/{}]'.format(i + 1, subsets))
        '''

    score = (S / C)  # Get average score

    return score
