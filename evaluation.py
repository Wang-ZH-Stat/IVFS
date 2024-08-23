'''
This script provides several functions to evaluate the effect of feature selection.
'''

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import pdist, squareform
from random import sample
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from skfeature.function.similarity_based import SPEC
from skfeature.function.sparse_learning_based import MCFS
from skfeature.utility import unsupervised_evaluation
import persim
import ripser
import time
from algorithm import IVFS


def knn_accuracy(X, y, repeat=10, seed=0):
    '''
    This function is used to calculate the KNN accuracy of selected features

    :param X: Filtered data
    :param y: True labels
    :param repeat: Number of KNN repeats

    :return: avg_acc - Average KNN accuracy
    '''

    # Divide training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Standardization
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train))
    X_test = pd.DataFrame(sc.fit_transform(X_test))

    total_acc = 0
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()
    for i in range(repeat):
        acc = 0
        for j in [1, 3, 5, 10]:
            # KNN algorithm
            knn = KNeighborsClassifier(n_neighbors=j, p=2, metric='minkowski')
            knn.fit(X_train, y_train)
            temp_acc = knn.score(X_test, y_test)
            if temp_acc > acc:
                acc = temp_acc
        total_acc += acc

    # Get the average accuracy of several repeated experiments
    avg_acc = total_acc / repeat

    return avg_acc


def clustering_accuracy_nmi(X, y, num_cluster, repeat=10):
    '''
    This function is used to calculate the Kmeans accuracy and normalized mutual information (nmi) of selected features

    :param X: Filtered data
    :param y: True labels
    :param num_cluster: Kmeans clustering number
    :param repeat: Number of Kmeans repeats

    :return: avg_nmi - Average normalized mutual information
    :return: avg_acc - Average Kmeans accuracy
    '''

    # Standardization
    sc = StandardScaler()
    X = pd.DataFrame(sc.fit_transform(X))

    X = np.array(X)
    y = np.array(y).flatten()
    nmi_total = 0
    acc_total = 0
    for i in range(0, repeat):
        # Using function from skfeature package to calculate nmi and acc
        nmi, acc = unsupervised_evaluation.evaluation(X_selected=X, n_clusters=num_cluster, y=y)
        nmi_total += nmi
        acc_total += acc

    # Get the average values of several repeated experiments
    avg_nmi = nmi_total / repeat
    avg_acc = acc_total / repeat

    return avg_nmi, avg_acc


def distances_between_persistent_diagrams(X, X_F, alpha=0.8, epsilon=0.05):
    '''
    This function is used to calculate the distances between persistent diagrams

    :param X: Original data
    :param X_F: Filtered data
    :param alpha: The threshold when building persistent diagrams
    :param epsilon: Barcodes with length less than a small number epsilon are regarded as noise

    :return: wasserstein_distance - The Wasserstein distance between persistent diagrams
    :return: bottleneck_distance - The Bottleneck distance between persistent diagrams
    '''

    # Normalize the filtration function to be bounded in [0, 1]
    D = pdist(X, metric='euclidean')
    D_F = pdist(X_F, metric='euclidean')
    X = X / np.max(D)
    X_F = X_F / np.max(D_F)

    # Build persistence diagrams
    dgm = ripser.ripser(X, thresh=alpha)['dgms'][1]
    dgm_F = ripser.ripser(X_F, thresh=alpha)['dgms'][1]

    # Normalize all birth / death time to [0,1]
    dgm = dgm / np.max(dgm)
    dgm_F = dgm / np.max(dgm_F)

    # Drop all barcodes with existing time less than epsilon
    dgm = np.array([x for x in dgm if x[1]-x[0] > epsilon])
    dgm_F = np.array([x for x in dgm_F if x[1] - x[0] > epsilon])

    # Calculate bottleneck distance
    bottleneck_distance = persim.bottleneck(dgm, dgm_F)

    # Calculate wasserstein distance
    dgm = np.delete(dgm.flatten(), np.argwhere(dgm.flatten() == np.inf))
    dgm_F = np.delete(dgm_F.flatten(), np.argwhere(dgm_F.flatten() == np.inf))
    if len(dgm) == 0 or len(dgm_F) == 0:
        wasserstein_distance = 0.0
    else:
        wasserstein_distance = stats.wasserstein_distance(dgm, dgm_F)

    return wasserstein_distance, bottleneck_distance


def norms_between_distance_matrix(X, X_F):
    '''
    This function is used to calculate the norms between distance matrix

    :param X: Original data
    :param X_F: Filtered data

    :return: L_inf - The L_infinite norm between distance matrix
    :return: L_1 - The L_1 norm between distance matrix
    :return: L_2 - The L_2 norm between distance matrix
    '''

    X = np.array(X)
    X_F = np.array(X_F)
    n = np.shape(X)[0]  # Number of samples
    D = pdist(X, metric='euclidean')
    D_F = pdist(X_F, metric='euclidean')

    # Normalize all distances to [0,1]
    D = D / np.max(D)
    D_F = D_F / np.max(D_F)

    # Calculate various types of norm between D and D_F
    L_inf = np.linalg.norm(D - D_F, ord=np.inf)
    L_1 = (np.linalg.norm(D - D_F, ord=1) / n ** 2) * 2
    L_2 = np.linalg.norm(D - D_F, ord=2) * 2

    return L_inf, L_1, L_2


# Stability under bootstrap
def bootstrap_stability(X, num_target_features=300, size=0.8, repeat=5, seed=0):
    '''
    This function is used to calculate the number of different selected features between original data and bootstrap data

    :param X: Standardized data
    :param num_target_features: Target dimension
    :param size: Proportion of bootstrap data to original data
    :param repeat: Number of bootstrap repeats

    :return: avg_num_IVFS - The average number of different selected features by IVFS
    :return: avg_num_SPEC - The average number of different selected features by SPEC
    :return: avg_num_MCFS - The average number of different selected features by MCFS
    '''

    random.seed(seed)

    # Record the number of different selected features
    diff_num_IVFS = 0
    diff_num_SPEC = 0
    diff_num_MCFS = 0

    n, d = X.shape

    # Experimental parameter setting
    kwargs_IVFS = {'subsets': 1000, 'num_features': int(0.3 * d), 'num_samples': int(0.1 * n), 'score_type': 'infinity'}
    kwargs_SPEC = {'style': 0}
    kwargs_MCFS = {'n_clusters': 10}

    # Feature selection form original data
    score_IVFS = IVFS(X, **kwargs_IVFS)
    selected_features_IVFS = np.argsort(-score_IVFS, 0)[:num_target_features]
    score_SPEC = SPEC.spec(np.array(X), **kwargs_SPEC)
    selected_features = np.argsort(-score_SPEC, 0)[:num_target_features]
    score_MCFS = MCFS.mcfs(np.array(X), **kwargs_MCFS)
    selected_features_MCFS = np.argsort(-score_MCFS, 0)[:num_target_features]

    # Standardization
    sc = StandardScaler()

    # Feature selection from bootstrap data
    for i in range(repeat):
        # Get bootstrap data
        b = sample(range(n), int(n * size))
        X_b = X.iloc[b, :]
        X_b = pd.DataFrame(sc.fit_transform(X_b))

        # By IVFS
        score_IVFS_b = IVFS(X_b, **kwargs_IVFS)
        selected_features_IVFS_b = np.argsort(-score_IVFS_b, 0)[:num_target_features]
        intersection_IVFS = [x for x in selected_features_IVFS_b if x in selected_features_IVFS]
        diff_num_IVFS += (num_target_features - len(intersection_IVFS))

        # By SPEC
        score_SPEC_b = SPEC.spec(np.array(X_b), **kwargs_SPEC)
        selected_features_b = np.argsort(-score_SPEC_b, 0)[:num_target_features]
        intersection_SPEC = [x for x in selected_features_b if x in selected_features]
        diff_num_SPEC += (num_target_features - len(intersection_SPEC))

        # By MCFS
        score_MCFS_b = MCFS.mcfs(np.array(X_b), **kwargs_MCFS)
        selected_features_MCFS_b = np.argsort(-score_MCFS_b, 0)[:num_target_features]
        intersection_MCFS = [x for x in selected_features_MCFS_b if x in selected_features_MCFS]
        diff_num_MCFS += (num_target_features - len(intersection_MCFS))

    # Get the average numbers
    avg_num_IVFS = diff_num_IVFS / repeat
    avg_num_SPEC = diff_num_SPEC / repeat
    avg_num_MCFS = diff_num_MCFS / repeat

    return avg_num_IVFS, avg_num_SPEC, avg_num_MCFS


def running_time(X):
    '''
    This function is used to calculate the running time for a single run, with fixed parameter setting

    :param X: Standardized data

    :return: dtime_IVFS - The running time for a single run by IVFS
    :return: dtime_SPEC - The running time for a single run by SPEC
    :return: dtime_MCFS - The running time for a single run by MCFS
    '''

    n, d = X.shape

    # Experimental parameter setting
    kwargs_IVFS = {'subsets': 1000, 'num_features': int(0.3 * d), 'num_samples': int(0.1 * n), 'score_type': 'infinity'}
    kwargs_SPEC = {'style': 0}
    kwargs_MCFS = {'n_clusters': 10}

    # Record the running time by IVFS
    starttime_IVFS = time.time()
    _ = IVFS(X, **kwargs_IVFS)
    endtime_IVFS = time.time()
    dtime_IVFS = endtime_IVFS - starttime_IVFS

    # Record the running time by SPEC
    starttime_SPEC = time.time()
    _ = SPEC.spec(np.array(X), **kwargs_SPEC)
    endtime_SPEC = time.time()
    dtime_SPEC = endtime_SPEC - starttime_SPEC

    # Record the running time by MCFS
    starttime_MCFS = time.time()
    _ = MCFS.mcfs(np.array(X), **kwargs_MCFS)
    endtime_MCFS = time.time()
    dtime_MCFS = endtime_MCFS - starttime_MCFS

    return dtime_IVFS, dtime_SPEC, dtime_MCFS
