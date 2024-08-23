'''
This script is used for main numerical experiments for all datasets.
The design of numerical experiments is based on the original article.
The results will be recorded in the log file.
'''

import warnings
warnings.filterwarnings('ignore')

from skfeature.function.similarity_based import SPEC
from skfeature.function.sparse_learning_based import MCFS
import pandas as pd
import numpy as np
import random
import os
import scipy.io
from sklearn.preprocessing import StandardScaler
from algorithm import IVFS
from evaluation import knn_accuracy, norms_between_distance_matrix, bootstrap_stability, \
    clustering_accuracy_nmi, distances_between_persistent_diagrams, running_time

# Seed for random
seed = 42
random.seed(seed)
np.random.seed(seed)

# List all data paths
dir_path = './data/'
data_list = []  # Get the data list
for root, dirs, files in os.walk(dir_path):
    data_list.extend(files)


def eval(X, selected_features, y, C):
    '''
    This function is used to evaluate the features selected by each algorithm under given parameters

    :param X: Original data
    :param selected_features: The index of the features selected
    :param y: True labels
    :param C: The number of calssed

    :return: Evaluation metric values. See below for details.
    '''

    # Get the filtered data
    X_filter = X[selected_features]

    # Start evaluation
    line_4 = 'Start evaluation: '
    print(line_4)
    with open('../log.txt', 'a') as f:
        f.write(line_4)

    # Calculate KNN accuracy
    KNN_acc = knn_accuracy(X_filter, y, repeat=10)
    line_5 = 'KNN accuracy is {:.4f}, '.format(KNN_acc)
    print(line_5, end='')
    with open('../log.txt', 'a') as f:
        f.write(line_5)

    # Calculate Normalized Mutual Information and Kmeans accuracy
    NMI, Kmeans_acc = clustering_accuracy_nmi(X_filter, y, num_cluster=C)
    line_6 = 'NMI is {:.4f}, Kmeans accuracy is {:.4f}, '.format(NMI, Kmeans_acc)
    print(line_6, end='')
    with open('../log.txt', 'a') as f:
        f.write(line_6)

    # Calculate distances between persistent diagrams
    wasserstein_distance, bottleneck_distance = distances_between_persistent_diagrams(X, X_filter)
    line_7 = 'Wasserstein Distance is {:.4f}, Bottleneck Distance is {:.4f}, '. \
        format(wasserstein_distance, bottleneck_distance)
    print(line_7, end='')
    with open('../log.txt', 'a') as f:
        f.write(line_7)

    # Calculate norms between distance matrix
    L_inf, L_1, L_2 = norms_between_distance_matrix(X, X_filter)
    line_8 = 'L_inf norm is {:.4f}, L_1 norm is {:.6f}, L_2 norm is {:.4f}.'. \
        format(L_inf, L_1, L_2)
    print(line_8)
    with open('../log.txt', 'a') as f:
        f.write(line_8)
        f.write('\r')

    # Return each value to tune the parameters
    return KNN_acc, NMI, Kmeans_acc, wasserstein_distance, bottleneck_distance, L_inf, L_1, L_2


def train(data_path):
    '''
    This function is the process of feature selection and evaluation for each dataset

    :param data_path: The path of the dataset to be used

    :return: There is no return value, the important thing is the log file.
    '''

    # Load data
    mat = scipy.io.loadmat(dir_path + data_path)
    X = pd.DataFrame(mat['X'].astype(float))
    y = pd.DataFrame(mat['Y'][:, 0])

    # Basic information of the dataset
    n, d = X.shape
    C = len(np.unique(y))

    # Write the log file, and the following is omitted
    line_1 = 'The dataset name is {}, the number of samples is {:.0f}, the dimensionality is {:.0f} and ' \
             'the number of classes is {:.0f}.'.format(data_path.replace('.mat', ''), n, d, C)
    print(line_1)
    with open('../log.txt', 'a') as f:
        f.write(line_1)
        f.write('\r\n')

    # Standardized to mean zero and unit variance
    sc = StandardScaler()
    X_std = pd.DataFrame(sc.fit_transform(X))

    # Number of features to be selected
    num_target_features = 300

    # Start feature selection
    for method in ['IVFS', 'SPEC', 'MCFS']:

        line_2 = 'Start feature selection, Use {}:'.format(method)
        print(line_2)
        with open('../log.txt', 'a') as f:
            f.write(line_2)
            f.write('\r')

        if method == 'IVFS':
            # IVFS

            # Record the best value of each evaluation metric
            best_eval = np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 500.0])

            # Tuning parameters
            for num_features in [0.1 * d, 0.3 * d, 0.5 * d]:

                # The selection of number of samples for each subset depends on the full sample size.
                if n <= 200:
                    num_samples_list = [0.1 * n, 0.3 * n, 0.5 * n]
                else:
                    num_samples_list = [100]

                for num_samples in num_samples_list:

                    for subsets in [1000, 2000, 3000]:

                        line_3 = 'd^tilde = {:.0f}, n^tilde = {:.0f}, k = {:.0f}'.\
                            format(num_features, num_samples, subsets)
                        print(line_3)
                        with open('../log.txt', 'a') as f:
                            f.write(line_3)
                            f.write('\r')

                        kwargs_IVFS = {'subsets': subsets, 'num_features': int(num_features),
                                  'num_samples': int(num_samples),
                                  'score_type': 'infinity'}
                        score = IVFS(X_std, **kwargs_IVFS)
                        selected_features = np.argsort(-score, 0)[:num_target_features]

                        # Evaluate the features selected
                        eval_value = eval(X=X, selected_features=selected_features, y=y, C=C)

                        # Update the best value of each evaluation metric
                        for i in range(3):
                            if eval_value[i] > best_eval[i]:
                                best_eval[i] = eval_value[i]
                        for i in range(3, 8):
                            if eval_value[i] < best_eval[i]:
                                best_eval[i] = eval_value[i]

            line_11 = 'The best KNN accuracy is {:.4f}, the bset NMI is {:.4f}, the bset Kmeans accuracy is {:.4f}, ' \
                      'the bset Wasserstein Distance is {:.4f}, the best Bottleneck Distance is {:.4f}, ' \
                      'the best L_inf norm is {:.4f}, the best L_1 norm is {:.6f}, the best L_2 norm is {:.4f}.'. \
                format(best_eval[0], best_eval[1], best_eval[2], best_eval[3],
                       best_eval[4], best_eval[5], best_eval[6], best_eval[7])
            print(line_11)
            with open('../log.txt', 'a') as f:
                f.write(line_11)
                f.write('\r\n')

        elif method == 'SPEC':
            # SPEC

            # There is no need to tune the parameter
            kwargs_SPEC = {'style': 0}  # Second type algorithm
            score = SPEC.spec(np.array(X_std), **kwargs_SPEC)
            selected_features = np.argsort(-score, 0)[:num_target_features]

            # Evaluate the features selected
            best_eval = eval(X=X, selected_features=selected_features, y=y, C=C)

            line_11 = 'The best KNN accuracy is {:.4f}, the bset NMI is {:.4f}, the bset Kmeans accuracy is {:.4f}, ' \
                      'the bset Wasserstein Distance is {:.4f}, the best Bottleneck Distance is {:.4f}, ' \
                      'the best L_inf norm is {:.4f}, the best L_1 norm is {:.6f}, the best L_2 norm is {:.4f}.'. \
                format(best_eval[0], best_eval[1], best_eval[2], best_eval[3],
                       best_eval[4], best_eval[5], best_eval[6], best_eval[7])
            print(line_11)
            with open('../log.txt', 'a') as f:
                f.write(line_11)
                f.write('\r\n')

        else:
            # MCFS

            # Record the best value of each evaluation metric
            best_eval = np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 500.0])

            # Tuning parameters
            for n_clusters in [5, 10, 20, 30]:

                line_3 = 'number of clusters = {:.0f}'.format(n_clusters)
                print(line_3)
                with open('../log.txt', 'a') as f:
                    f.write(line_3)
                    f.write('\r')

                kwargs_MCFS = {'n_clusters': n_clusters}
                score = MCFS.mcfs(np.array(X_std), **kwargs_MCFS)
                selected_features = np.argsort(-score, 0)[:num_target_features]

                # Evaluate the features selected
                eval_value = eval(X=X, selected_features=selected_features, y=y, C=C)

                # Update the best value of each evaluation metric
                for i in range(3):
                    if eval_value[i] > best_eval[i]:
                        best_eval[i] = eval_value[i]
                for i in range(3, 8):
                    if eval_value[i] < best_eval[i]:
                        best_eval[i] = eval_value[i]

            line_11 = 'The best KNN accuracy is {:.4f}, the bset NMI is {:.4f}, the bset Kmeans accuracy is {:.4f}, ' \
                      'the bset Wasserstein Distance is {:.4f}, the best Bottleneck Distance is {:.4f}, ' \
                      'the best L_inf norm is {:.4f}, the best L_1 norm is {:.6f}, the best L_2 norm is {:.4f}.'. \
                format(best_eval[0], best_eval[1], best_eval[2], best_eval[3],
                       best_eval[4], best_eval[5], best_eval[6], best_eval[7])
            print(line_11)
            with open('../log.txt', 'a') as f:
                f.write(line_11)
                f.write('\r\n')

    # Calculate the number of different selected features between original data and bootstrap data
    stability_IVFS, stability_SPEC, stability_MCFS = \
        bootstrap_stability(X_std, num_target_features=num_target_features)

    line_9 = 'The number of different selected features is {:.0f} by IVFS, ' \
             '{:.0f} by SPEC, and {:.0f} by MCFS.'. \
        format(stability_IVFS, stability_SPEC, stability_MCFS)
    print(line_9)
    with open('../log.txt', 'a') as f:
        f.write(line_9)
        f.write('\r\n')

    # Calculate the running time of each algorithm under given parameters
    dtime_IVFS, dtime_SPEC, dtime_MCFS = running_time(X_std)

    line_10 = 'The running time is {:.2f} by IVFS, ' \
              '{:.2f} by SPEC, and {:.2f} by MCFS.'. \
        format(dtime_IVFS, dtime_SPEC, dtime_MCFS)
    print(line_10)
    with open('../log.txt', 'a') as f:
        f.write(line_10)
        f.write('\r\n\r\n')


if __name__ == '__main__':
    # Start training for each dataset
    for data_path in data_list:
        train(data_path)
