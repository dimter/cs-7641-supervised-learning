from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import datasets
import logging
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
import sklearn.model_selection
import sklearn.tree
import sklearn.metrics
import experiments

logging.basicConfig(level=logging.INFO)

wine_dataset = datasets.WineDataset()
wine_dataset.load()
wine_dataset.split_train_test(test_size=0.3)

breast_cancer_dataset = datasets.BreastCancerDataset()
breast_cancer_dataset.load()
breast_cancer_dataset.split_train_test(test_size=0.3)


def decision_tree_experiments(dataset):
    decision_tree_experiment = experiments.DecisionTreeExperiment(dataset)
    decision_tree_experiment.explore_hyperparameter_tuning(
        'min_samples_split',
        range(2, 300, 5),
        f'{decision_tree_experiment.get_title_preffix()}Min Samples Split analysis')
    decision_tree_experiment.explore_hyperparameter_tuning(
        'ccp_alpha',
        np.linspace(0.001, 0.1, 300),
        f'{decision_tree_experiment.get_title_preffix()}CCP Alpha analysis')
    decision_tree_experiment.model_complexity_analysis()
    parameters = decision_tree_experiment.get_parameters_for_tuning()
    decision_tree_experiment.tune_hyperparameters(parameters)
    decision_tree_experiment.learning_curve()
    decision_tree_experiment.compare_classifier_performance()
    return decision_tree_experiment


def boosting_experiments(dataset):
    experiment = experiments.BoostingExperiment(dataset)
    _, _, best_n_estimators = experiment.model_complexity_analysis()
    experiment.tuned_hyperparameters = {
        'n_estimators': best_n_estimators
    }
    experiment.learning_curve()
    experiment.compare_classifier_performance()
    return experiment


def knn_experiments(dataset):
    experiment = experiments.KNeighborsExperiment(dataset)
    _, _, best_n_neighbours = experiment.model_complexity_analysis()
    parameters = experiment.get_parameters_for_tuning()
    experiment.tune_hyperparameters(parameters)
    experiment.learning_curve()
    experiment.compare_classifier_performance()
    return experiment


def ann_experiments(dataset):
    experiment = experiments.NeuralNetworkExperiment(dataset)
    experiment.model_complexity_analysis()
    experiment.alpha_analysis()
    experiment.learning_rate_analysis()
    experiment.epoch_analysis()
    parameters = experiment.get_parameters_for_tuning()
    experiment.tune_hyperparameters(parameters)
    experiment.learning_curve()
    experiment.compare_classifier_performance()
    return experiment


def svm_experiments(dataset):
    experiment = experiments.SVMExperiment(dataset)
    experiment.tuned_hyperparameters = {}
    experiment.learning_curve()
    experiment.compare_classifier_performance()
    return experiment


w_tree = decision_tree_experiments(wine_dataset)
b_tree = decision_tree_experiments(breast_cancer_dataset)
w_boost = boosting_experiments(wine_dataset)
b_boost = boosting_experiments(breast_cancer_dataset)
w_knn = knn_experiments(wine_dataset)
b_knn = knn_experiments(breast_cancer_dataset)
w_ann = ann_experiments(wine_dataset)
b_ann = ann_experiments(breast_cancer_dataset)
w_svm = svm_experiments(wine_dataset)
b_svm = svm_experiments(breast_cancer_dataset)
