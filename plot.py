'''
This script is used to give visual analysis results similar to the original article.
'''

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler
from skfeature.function.similarity_based import SPEC
from skfeature.function.sparse_learning_based import MCFS
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
from random import sample
from matplotlib import cm
from algorithm import IVFS
from evaluation import norms_between_distance_matrix, distances_between_persistent_diagrams

dir_path = '../data/'


#################################################################
# Part 1: l2 norm / Bottleneck distance vs. number of features
#################################################################

def plot_with_number_features(data_path, evaluation_list):
    '''
    This function is used to draw the change of L2 nrom between distance matrix and bottleneck distances
    between persistent diagrams with the increase of target dimension in a given dataset  results
    by different algorithms.

    :param data_path: The path of the dataset to be used
    :param evaluation_list: Including 'L2 norm' and 'bottleneck distance'

    :return: There is no explicit return value. Picture is important.
    '''

    # Load data
    mat = scipy.io.loadmat(dir_path + data_path)
    X = pd.DataFrame(mat['X'].astype(float))
    y = pd.DataFrame(mat['Y'][:, 0])

    # Basic information of the dataset
    n, d = X.shape
    C = len(np.unique(y))

    # Standardized to mean zero and unit variance
    sc = StandardScaler()
    X_std = pd.DataFrame(sc.fit_transform(X))

    # Set the parameters of each algorithm
    kwargs_IVFS_infinity = {'subsets': 3000, 'num_features': int(0.5 * d),
                            'num_samples': int(0.5 * n), 'score_type': 'infinity'}
    kwargs_IVFS_L1 = {'subsets': 3000, 'num_features': int(0.5 * d),
                            'num_samples': int(0.5 * n), 'score_type': 'L1'}
    kwargs_IVFS_L2 = {'subsets': 3000, 'num_features': int(0.5 * d),
                            'num_samples': int(0.5 * n), 'score_type': 'L2'}
    kwargs_SPEC = {'style': 0}
    kwargs_MCFS = {'n_clusters': 10}

    # Feature selection using each algorithm
    score_IVFS_infinity = IVFS(X_std, **kwargs_IVFS_infinity)
    score_IVFS_L1 = IVFS(X_std, **kwargs_IVFS_L1)
    score_IVFS_L2 = IVFS(X_std, **kwargs_IVFS_L2)
    score_SPEC = SPEC.spec(np.array(X_std), **kwargs_SPEC)
    score_MCFS = MCFS.mcfs(np.array(X_std), **kwargs_MCFS)

    # Get L2 norm and bottleneck distance by each algorithm
    def get_eval_value(score, num_target_features):
        selected_features = np.argsort(-score, 0)[:num_target_features]
        X_filter = X[selected_features]
        _, bottleneck_distance = distances_between_persistent_diagrams(X, X_filter)
        _, _, L_2 = norms_between_distance_matrix(X, X_filter)
        return L_2, bottleneck_distance

    # There are two evaluation metrics
    for evaluation in evaluation_list:
        if evaluation == 'L2 norm':
            # Record value
            L_2_IVFS_infinity = []
            L_2_IVFS_L1 = []
            L_2_IVFS_L2 = []
            L_2_SPEC = []
            L_2_MCFS = []

            # [20, 300] for number of target features
            for num_target_features in range(20, 301):
                L_2, _ = get_eval_value(score_IVFS_infinity, num_target_features)
                L_2_IVFS_infinity.append(L_2)

                L_2, _ = get_eval_value(score_IVFS_L1, num_target_features)
                L_2_IVFS_L1.append(L_2)

                L_2, _ = get_eval_value(score_IVFS_L2, num_target_features)
                L_2_IVFS_L2.append(L_2)

                L_2, _ = get_eval_value(score_SPEC, num_target_features)
                L_2_SPEC.append(L_2)

                L_2, _ = get_eval_value(score_MCFS, num_target_features)
                L_2_MCFS.append(L_2)

            # Plot
            index = range(20, 301)
            plt.plot(index, L_2_SPEC, label='SPEC', linestyle='--')
            plt.plot(index, L_2_MCFS, label='MCFS', linestyle='--')
            plt.plot(index, L_2_IVFS_infinity, label='IVFS-$l_{\infty}$', linestyle='-')
            plt.plot(index, L_2_IVFS_L1, label='IVFS-$l_1$', linestyle='-')
            plt.plot(index, L_2_IVFS_L2, label='IVFS-$l_2$', linestyle='-')
            plt.legend()
            plt.xticks([20, 100, 200, 300])
            plt.title('Dataset: ' + data_path.replace('.mat', ''))
            plt.xlabel('# features')
            plt.ylabel('$L_2$ norm')
            plt.grid()
            plt.savefig('../pictures/' + data_path.replace('.mat', '') + '_' + evaluation + '.jpg', dpi=900)
            plt.show()

        else:
            # Record value
            bottleneck_distance_IVFS_infinity = []
            bottleneck_distance_IVFS_L1 = []
            bottleneck_distance_IVFS_L2 = []
            bottleneck_distance_SPEC = []
            bottleneck_distance_MCFS = []

            # [20, 300] for number of target features
            for num_target_features in range(20, 301):
                _, bottleneck_distance = get_eval_value(score_IVFS_infinity, num_target_features)
                bottleneck_distance_IVFS_infinity.append(bottleneck_distance)

                _, bottleneck_distance = get_eval_value(score_IVFS_L1, num_target_features)
                bottleneck_distance_IVFS_L1.append(bottleneck_distance)

                _, bottleneck_distance = get_eval_value(score_IVFS_L2, num_target_features)
                bottleneck_distance_IVFS_L2.append(bottleneck_distance)

                _, bottleneck_distance = get_eval_value(score_SPEC, num_target_features)
                bottleneck_distance_SPEC.append(bottleneck_distance)

                _, bottleneck_distance = get_eval_value(score_MCFS, num_target_features)
                bottleneck_distance_MCFS.append(bottleneck_distance)

            # Plot
            index = range(20, 301)
            plt.plot(index, bottleneck_distance_SPEC, label='SPEC', linestyle='--')
            plt.plot(index, bottleneck_distance_MCFS, label='MCFS', linestyle='--')
            plt.plot(index, bottleneck_distance_IVFS_infinity, label='IVFS-$l_{\infty}$', linestyle='-')
            plt.plot(index, bottleneck_distance_IVFS_L1, label='IVFS-$l_1$', linestyle='-')
            plt.plot(index, bottleneck_distance_IVFS_L2, label='IVFS-$l_2$', linestyle='-')
            plt.legend()
            plt.xticks([20, 100, 200, 300])
            plt.title('Dataset: ' + data_path.replace('.mat', ''))
            plt.xlabel('# features')
            plt.ylabel('Bottleneck Distance')
            plt.grid()
            plt.savefig('../pictures/' + data_path.replace('.mat', '') + '_' + evaluation + '.jpg', dpi=900)
            plt.show()


# Get the visualization results
plot_with_number_features('Lymphoma.mat', ['L2 norm'])
plot_with_number_features('Orlraws10P.mat', ['L2 norm'])
plot_with_number_features('WarpPIE10P.mat', ['L2 norm'])
plot_with_number_features('Prostate-GE.mat', ['L2 norm'])
plot_with_number_features('Pixraw10P.mat', ['L2 norm', 'bottleneck distance'])
plot_with_number_features('SMK-CAN-187.mat', ['L2 norm', 'bottleneck distance'])


###########################################################################
# Part 2: the number of different selected features  vs. number of subsets
###########################################################################

def bootstrap_stability_IVFS(X, subsets=1000, num_target_features=300, size=0.8, repeat=5, seed=0):
    '''
    This function is is similar to 'bootstrap_stability from' 'evaluation' module.
    Here we only consider IVFS algorithm.

    :param X: Standardized data
    :param subsets: The number of subsets
    :param num_target_features: Target dimension
    :param size: Proportion of bootstrap data to original data
    :param repeat: Number of bootstrap repeats

    :return: avg_num_IVFS - The average number of different selected features by IVFS under given number of subsets
    '''

    random.seed(seed)

    # Record the number of different selected features
    diff_num_IVFS = 0

    n, d = X.shape

    # Experimental parameter setting
    kwargs_IVFS = {'subsets': subsets, 'num_features': int(0.3 * d), 'num_samples': int(0.1 * n), 'score_type': 'infinity'}

    # Feature selection form original data
    score_IVFS = IVFS(X, **kwargs_IVFS)
    selected_features_IVFS = np.argsort(-score_IVFS, 0)[:num_target_features]

    # Standardization
    sc = StandardScaler()

    # Feature selection from bootstrap data
    for i in range(repeat):
        # Get bootstrap data
        b = sample(range(n), int(n * size))
        X_b = X.iloc[b, :]
        X_b = pd.DataFrame(sc.fit_transform(X_b))

        score_IVFS_b = IVFS(X_b, **kwargs_IVFS)
        selected_features_IVFS_b = np.argsort(-score_IVFS_b, 0)[:num_target_features]
        intersection_IVFS = [x for x in selected_features_IVFS_b if x in selected_features_IVFS]
        diff_num_IVFS += (num_target_features - len(intersection_IVFS))

    # Get the average numbers
    avg_num_IVFS = diff_num_IVFS / repeat

    return avg_num_IVFS

def plot_with_subsets(data_path):
    '''
    This function is used to draw the number of different selected features between original
    data and bootstrap data vs the number of subsets.

    :param data_path: The path of the dataset to be used

    :return: There is no explicit return value. Picture is important.
    '''

    # Load data
    mat = scipy.io.loadmat(dir_path + data_path)
    X = pd.DataFrame(mat['X'].astype(float))

    subsets_list = [500, 1000, 1500, 2000, 3000, 4000, 5000]
    different_numbers = []
    for subsets in subsets_list:
        avg_num = bootstrap_stability_IVFS(X, subsets)
        different_numbers.append(avg_num)

    # Plot
    plt.plot(subsets_list, different_numbers, linestyle='--', marker='x', markersize=8)
    plt.xticks(subsets_list)
    plt.xlabel('# subsets')
    plt.ylabel('# different selected features')
    plt.title('Dataset: ' + data_path.replace('.mat', ''))
    plt.grid()
    plt.savefig('../pictures/' + data_path.replace('.mat', '') + '_different selected features.jpg', dpi=900)
    plt.show()

# Get the visualization results
plot_with_subsets('Isolet.mat')
plot_with_subsets('OLR.mat')


##########################################################################################
# Part 3: Performance comparison across different number of subsets and sub-sampling ratio
##########################################################################################

# Get the grid coordinates
X = [0.1, 0.3, 0.5]
Y = [1000, 2000, 3000]
X, Y = np.meshgrid(X, Y)

# Get the Bottleneck and L2 norm data from all datasets
# The data comes from the previous log file
list_Bottleneck = np.array(
    [
        [0.0540, 0.1748, 0.0443, 0.1112, 0.0849, 0.0755, 0.1008, 0.0924],
        [0.0562, 0.1533, 0.0446, 0.1100, 0.0873, 0.0808, 0.0958, 0.0913],
        [0.0620, 0.0672, 0.0446, 0.1106, 0.0801, 0.0813, 0.0989, 0.0927],
        [0.0552, 0.2418, 0.0448, 0.1077, 0.0801, 0.0822, 0.0889, 0.0870],
        [0.0622, 0.0918, 0.0432, 0.1069, 0.0782, 0.0872, 0.0881, 0.0878],
        [0.0619, 0.1410, 0.0420, 0.1121, 0.0916, 0.0822, 0.0918, 0.0877],
        [0.0563, 0.1024, 0.0443, 0.1101, 0.0748, 0.0893, 0.0885, 0.0976],
        [0.0606, 0.2319, 0.0432, 0.1085, 0.0871, 0.0933, 0.0940, 0.0920],
        [0.0545, 0.1150, 0.0434, 0.1113, 0.0777, 0.0889, 0.0867, 0.0958]
     ])
list_L2 = np.array(
    [
        [11.0182, 10.2986, 3.3996, 3.2021, 4.3191, 2.5956, 5.9585, 3.9482],
        [11.5914, 8.4595, 3.5464, 3.2132, 2.9624, 2.8315, 6.8364, 4.2798],
        [10.8392, 6.0230, 3.3374, 4.3444, 3.6354, 2.7988, 9.9454, 4.2056],
        [13.3203, 8.1619, 3.8602, 4.1865, 3.9622, 3.3053, 6.4518, 4.6149],
        [13.8932, 9.8411, 4.1695, 3.0001, 3.1889, 2.7228, 7.3433, 5.2846],
        [11.2148, 10.9239, 5.0405, 4.1608, 3.6973, 2.9010, 8.4685, 5.5010],
        [11.8991, 11.7212, 3.3875, 4.2075, 4.0002, 2.4296, 10.9854, 6.3820],
        [11.7877, 7.1440, 4.0229, 3.8920, 3.3369, 4.5243, 7.8360, 4.9448],
        [9.6875, 33.2421, 4.3747, 3.4996, 4.2857, 3.1466, 6.7988, 5.9299]
     ])

# Normalization to [0.9, 1.1]
minmax = preprocessing.MinMaxScaler()
list_Bottleneck_std = minmax.fit_transform(list_Bottleneck) / 5 + 0.9
list_L2_std = minmax.fit_transform(list_L2) / 5 + 0.9

# Get the mean value over all datasets
mean_Bottleneck = np.array([])
mean_L2 = np.array([])
for i in range(len(list_Bottleneck_std)):
    mean_Bottleneck = np.append(mean_Bottleneck, np.mean(list_Bottleneck_std[i]))
    mean_L2 = np.append(mean_L2, np.mean(list_L2_std[i]))
mean_Bottleneck = mean_Bottleneck.reshape((3, 3))
mean_L2 = mean_L2.reshape((3, 3))

# Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X.T, Y.T, mean_L2, cmap=cm.coolwarm, linewidth=0)
plt.xticks([0.1, 0.3, 0.5])
plt.yticks([1000, 2000, 3000])
plt.title('$L_2$ norm')
ax.set_xlabel('sub-sampling ratio')
ax.set_ylabel('number of subsets')
fig.colorbar(surf, shrink=0.3, aspect=8)
plt.savefig('../pictures/performance comparison_L2 norm.jpg', dpi=900)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X.T, Y.T, mean_Bottleneck, cmap=cm.coolwarm, linewidth=0)
plt.xticks([0.1, 0.3, 0.5])
plt.yticks([1000, 2000, 3000])
plt.title('Bottleneck')
ax.set_xlabel('sub-sampling ratio')
ax.set_ylabel('number of subsets')
fig.colorbar(surf, shrink=0.3, aspect=8)
plt.savefig('../pictures/performance comparison_Bottleneck.jpg', dpi=900)
plt.show()
