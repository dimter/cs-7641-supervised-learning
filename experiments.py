import numpy as np
from abc import ABC, abstractmethod
import json
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.tree
import sklearn.neighbors
import sklearn.neural_network
import sklearn.ensemble
import sklearn.svm
import time
import logging


def plot_with_erros(
        x_axis_range,
        x_axis_name,
        train_scores,
        test_scores,
        figure_name,
        log_scale=False):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    lw = 2
    plt.figure()
    plt.fill_between(
        x_axis_range,
        train_scores_mean -
        train_scores_std,
        train_scores_mean +
        train_scores_std,
        alpha=0.2,
        lw=lw)
    plt.fill_between(
        x_axis_range,
        test_scores_mean -
        test_scores_std,
        test_scores_mean +
        test_scores_std,
        alpha=0.2,
        lw=lw)
    plt.title(figure_name)
    if log_scale:
        plt.semilogx(x_axis_range, np.mean(train_scores, axis=1), 'o-', label='Training score')
        plt.semilogx(
            x_axis_range,
            np.mean(
                test_scores,
                axis=1),
            'o-',
            label='Cross-validation score')
    else:
        plt.plot(x_axis_range, np.mean(train_scores, axis=1), 'o-', label='Training score')
        plt.plot(x_axis_range, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
    plt.xlabel(x_axis_name)
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.savefig(f'results/{figure_name.lower()}.png')


def plot_without_erros(
        x_axis_range,
        x_axis_name,
        train_scores,
        test_scores,
        figure_name):
    plt.figure()
    plt.title(figure_name)
    plt.plot(x_axis_range, train_scores, label='Training score')
    plt.plot(x_axis_range, test_scores, label='Cross-validation score')
    plt.xlabel(x_axis_name)
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.savefig(f'results/{figure_name.lower()}.png')


def simple_plot(x_axis_range, x_axis_name, y_values, y_axis_name, figure_name):
    y_values_mean = np.mean(y_values, axis=1)
    y_values_std = np.std(y_values, axis=1)
    plt.clf()
    plt.plot(x_axis_range, y_values_mean, 'o-')
    plt.fill_between(x_axis_range, y_values_mean - y_values_std,
                     y_values_mean + y_values_std, alpha=0.1)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(figure_name)
    plt.savefig(f'results/{figure_name.lower()}.png')


def ultra_simple_plot(x_axis_range, x_axis_name, y_values, y_axis_name, figure_name):
    plt.clf()
    plt.plot(x_axis_range, y_values)
    plt.ylabel(y_axis_name)
    plt.title(figure_name)
    plt.savefig(f'results/{figure_name.lower()}.png')


def save_hyperparameters(filename, hyperparameters):
    with open(filename, 'w') as f:
        for key, value in hyperparameters.items():
            f.write(f'{key}={value}\n')


class Experiment(ABC):
    def __init__(
            self,
            classifier_name,
            classifier_class,
            default_classifier_parameters,
            dataset,
            cv: int = 5,
            n_jobs: int = -1):
        if 'random_state' not in default_classifier_parameters:
            default_classifier_parameters['random_state'] = 42
        elif default_classifier_parameters['random_state'] is None:
            default_classifier_parameters.pop('random_state')
        self.classifier_name = classifier_name
        self.classifier_class = classifier_class
        self.default_classifier_parameters = default_classifier_parameters
        self.dataset = dataset
        self.cv = cv
        self.n_jobs = n_jobs
        self.tuned_hyperparameters = None
        self.tuned_model = None
        self.training_time = None
        self.query_time = None
        self.default_accuracy = None
        self.tuned_accuracy = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f'Initializing experiment for dataset {self.dataset.name}')

    @abstractmethod
    def get_parameters_for_tuning(self, *args, **kwargs):
        pass

    @abstractmethod
    def model_complexity_analysis(self):
        pass

    def get_title_preffix(self):
        return f'{self.classifier_name} ({self.dataset.name} dataset) - '

    def get_tuned_hyperparameters_filename(self):
        return f'results/{self.get_title_preffix().lower()}Tuned Hyperparameters.txt'.lower()

    def tune_hyperparameters(self, parameters, save_to_file=True):
        self.logger.info(
            f'Tuning hyperparameters ({parameters.keys()}) for dataset {self.dataset.name}.')
        model = sklearn.model_selection.GridSearchCV(
            estimator=self.classifier_class(**dict(self.default_classifier_parameters)),
            param_grid=parameters,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring='accuracy')
        model.fit(self.dataset.x_train, self.dataset.y_train)
        self.tuned_hyperparameters = dict(model.best_params_)
        self.tuned_hyperparameters['random_state'] = 42
        self.tuned_model = model
        if save_to_file:
            save_hyperparameters(
                self.get_tuned_hyperparameters_filename(),
                self.tuned_hyperparameters)
        self.logger.info(
            f'Tuning hyperparameters ({parameters.keys()}) for dataset {self.dataset.name}. Done!')
        return self.tuned_hyperparameters

    def explore_hyperparameter_tuning(
            self,
            parameter_name,
            parameter_range,
            figure_name,
            plot_results=True,
            log_scale=False):
        self.logger.info(f'Tuning {parameter_name} for dataset {self.dataset.name}')
        train_scores, test_scores = sklearn.model_selection.validation_curve(
            self.classifier_class(**self.default_classifier_parameters),
            self.dataset.x_train,
            self.dataset.y_train,
            param_name=parameter_name,
            param_range=parameter_range,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring='accuracy')
        if plot_results:
            parameter_range_for_plot = parameter_range
            if isinstance(parameter_range[0], tuple):
                parameter_range_for_plot = [p[0] for p in parameter_range]
            plot_with_erros(
                parameter_range_for_plot,
                parameter_name,
                train_scores,
                test_scores,
                figure_name,
                log_scale=log_scale)
        self.logger.info(
            f'Tuning {parameter_name} for dataset {self.dataset.name} was completed')
        best_value = parameter_range[np.argmax(np.mean(test_scores, axis=1))]
        best_score = np.max(np.mean(test_scores, axis=1))
        self.logger.info(f'Best value for {parameter_name}={best_value} (score={best_score})')
        return train_scores, test_scores, best_value

    def learning_curve(self):
        self.logger.info(f'Computing learning curve for dataset {self.dataset.name}')
        parameters = dict(self.default_classifier_parameters)
        parameters.update(self.tuned_hyperparameters)
        train_sizes, train_scores, test_scores, fit_times, _ = sklearn.model_selection.learning_curve(
            self.classifier_class(**parameters),
            self.dataset.x_train,
            self.dataset.y_train,
            cv=self.cv,
            n_jobs=self.n_jobs,
            train_sizes=np.linspace(0.1, 1.0, 5),
            return_times=True,
            scoring='accuracy')

        plot_with_erros(train_sizes, 'Training examples', train_scores, test_scores,
                        f'{self.classifier_name} ({self.dataset.name} dataset) - Learning Curve')

        simple_plot(
            train_sizes,
            'Training examples',
            fit_times,
            'Training Time',
            f'{self.classifier_name} ({self.dataset.name} dataset) - Scalability of the model')

        simple_plot(
            np.mean(fit_times, axis=1),
            'Training Time',
            test_scores,
            'Accuracy',
            f'{self.classifier_name} ({self.dataset.name} dataset) - Performance of the model')
        self.logger.info(f'Computing learning curve for dataset {self.dataset.name}. Done!')

    def compare_classifier_performance(self):
        self.logger.info(f'Comparing classifier performance for dataset {self.dataset.name}.')
        classifier = self.classifier_class(**self.default_classifier_parameters)
        classifier.fit(self.dataset.x_train, self.dataset.y_train)
        self.default_accuracy = sklearn.metrics.accuracy_score(
            self.dataset.y_test, classifier.predict(
                self.dataset.x_test))
        classifier = self.classifier_class(**self.tuned_hyperparameters)
        start_time = time.time()
        classifier.fit(self.dataset.x_train, self.dataset.y_train)
        self.training_time = time.time() - start_time
        start_time = time.time()
        predictions = classifier.predict(self.dataset.x_test)
        self.query_time = time.time() - start_time
        self.tuned_accuracy = sklearn.metrics.accuracy_score(self.dataset.y_test, predictions)
        with open(f'results/{self.get_title_preffix()}Classifier performance'.lower(), 'w') as f:
            f.write(f'Default Classifier Accuracy: {self.default_accuracy}\n')
            f.write(f'Tuned Classifier Accuracy: {self.tuned_accuracy}\n')
            f.write(f'Training time (sec): {self.training_time}\n')
            f.write(f'Query time (sec): {self.query_time}\n')
        self.logger.info(f'Comparing classifier performance for dataset {self.dataset.name}. Done!')


class DecisionTreeExperiment(Experiment):
    def __init__(self, dataset, default_classifier_parameters={}):
        super().__init__(
            classifier_name='Decision Tree',
            classifier_class=sklearn.tree.DecisionTreeClassifier,
            default_classifier_parameters=default_classifier_parameters,
            dataset=dataset)

    def get_parameters_for_tuning(self,
                                  max_depth_start: int = 1,
                                  max_depth_end: int = 150,
                                  min_samples_leaf_min_percentage: float = 0.005,
                                  min_samples_leaf_max_percentage: float = 0.05,
                                  min_samples_split_start: int = 2,
                                  min_samples_split_end: int = 5,
                                  ccp_alpha_min: float = 0,
                                  ccp_alpha_max: float = 0.03):
        max_depth = np.arange(max_depth_start, max_depth_end)
        ccp_alpha = np.linspace(ccp_alpha_min, ccp_alpha_max, 10)
        min_samples_leaf = np.linspace(
            round(min_samples_leaf_min_percentage * len(self.dataset.x_train)),
            round(min_samples_leaf_max_percentage * len(self.dataset.x_train)),
            20
        ).round().astype('int')
        min_samples_split = np.arange(min_samples_split_start, min_samples_split_end)
        return {
            'criterion': ['gini', 'entropy'],
            'max_depth': max_depth,
            # 'ccp_alpha': ccp_alpha,
            # 'min_samples_leaf': min_samples_leaf,
            # 'min_samples_split': min_samples_split,
        }

    def model_complexity_analysis(self):
        self.explore_hyperparameter_tuning(
            'max_depth',
            range(1, 31),
            f'{self.classifier_name} ({self.dataset.name} dataset) - Model Complexity')


class KNeighborsExperiment(Experiment):
    def __init__(self, dataset):
        default_classifier_parameters = {'random_state': None}
        super().__init__(
            classifier_name='kNN',
            classifier_class=sklearn.neighbors.KNeighborsClassifier,
            default_classifier_parameters=default_classifier_parameters,
            dataset=dataset)

    def get_parameters_for_tuning(self,
                                  n_neighbors_min: int = 1,
                                  n_neighbors_max: int = 50):
        metric = ['euclidean', 'manhattan']
        n_neighbors = np.arange(n_neighbors_min, n_neighbors_max)
        return {
            'metric': metric,
            'n_neighbors': n_neighbors,
        }

    def model_complexity_analysis(self):
        return self.explore_hyperparameter_tuning(
            'n_neighbors',
            range(1, 101),
            f'{self.classifier_name} ({self.dataset.name} dataset) - Model Complexity')

    def learning_curve(self):
        self.tuned_hyperparameters.pop('random_state')
        super().learning_curve()


class BoostingExperiment(Experiment):
    def __init__(
            self,
            dataset,
            child_classifier_class=sklearn.tree.DecisionTreeClassifier,
            default_classifier_parameters={},
            default_child_classifier_parameters={}):
        self.child_classifier_class = child_classifier_class
        self.default_child_classifier_parameters = default_child_classifier_parameters
        # default_classifier_parameters['base_estimator'] = self.child_classifier_class(
        #     **self.default_child_classifier_parameters)
        super().__init__(
            classifier_name='Boosting',
            classifier_class=sklearn.ensemble.AdaBoostClassifier,
            default_classifier_parameters=default_classifier_parameters,
            dataset=dataset)

    def get_parameters_for_tuning(self,
                                  n_estimators_min: int = 1,
                                  n_estimators_max: int = 100,
                                  max_depth_min: int = 1,
                                  max_depth_max: int = 20):
        n_estimators = np.arange(n_estimators_min, n_estimators_max)
        return {
            'n_estimators': n_estimators,
            'max_depth': np.arange(max_depth_min, max_depth_max)
        }

    def model_complexity_analysis(self):
        return self.explore_hyperparameter_tuning(
            'n_estimators',
            np.arange(1, 1001, 10),
            f'{self.classifier_name} ({self.dataset.name} dataset) - Model Complexity')

    def tune_hyperparameters(self, parameters):
        max_depth_range = parameters.pop('max_depth')
        best_score = 0
        best_model = None
        classifier_parameters_backup = dict(self.default_classifier_parameters)
        for max_depth in max_depth_range:
            child_classifier_parameters = dict(self.default_child_classifier_parameters)
            child_classifier_parameters['max_depth'] = max_depth
            self.default_classifier_parameters['base_estimator'] = self.child_classifier_class(
                **child_classifier_parameters)
            super().tune_hyperparameters(parameters, save_to_file=False)
            if best_model is None or best_model.best_score_ < self.tuned_model.best_score_:
                best_model = self.tuned_model
        self.tuned_model = best_model
        self.tuned_hyperparameters = best_model.best_params_
        self.default_classifier_parameters = classifier_parameters_backup
        save_hyperparameters(
            self.get_tuned_hyperparameters_filename(),
            self.tuned_hyperparameters)
        return self.tuned_hyperparameters


class NeuralNetworkExperiment(Experiment):
    def __init__(self, dataset):
        default_classifier_parameters = {
            'max_iter': 10000,
            'early_stopping': False,
            'hidden_layer_sizes': (10, 5)
        }
        super().__init__(
            classifier_name='ANN',
            classifier_class=sklearn.neural_network.MLPClassifier,
            default_classifier_parameters=default_classifier_parameters,
            dataset=dataset)

    def get_parameters_for_tuning(self):
        alphas = np.logspace(-3, 1, 10)
        learning_rates = np.logspace(-5, 0, 6)
        activations = ['logistic', 'tanh', 'identity', 'relu']
        return {
            'alpha': alphas,
            'learning_rate_init': learning_rates,
            # 'activation': activations
        }

    def model_complexity_analysis(self):
        neurons_per_layer = np.arange(2, 21)
        hidden_layer_sizes = [(npl,) for npl in neurons_per_layer]
        self.explore_hyperparameter_tuning(
            'hidden_layer_sizes',
            hidden_layer_sizes,
            f'{self.get_title_preffix()}Model Complexity - One Layer')

        hidden_layer_sizes = [(npl, npl // 2) for npl in neurons_per_layer]
        self.explore_hyperparameter_tuning(
            'hidden_layer_sizes',
            hidden_layer_sizes,
            f'{self.get_title_preffix()}Model Complexity - Two Layers (2x, 1x)')

        hidden_layer_sizes = [(npl, npl * 2) for npl in neurons_per_layer]
        self.explore_hyperparameter_tuning(
            'hidden_layer_sizes',
            hidden_layer_sizes,
            f'{self.get_title_preffix()}Model Complexity - Two Layers (1x, 2x)')

    def alpha_analysis(self):
        alpha_range = np.logspace(-3, 3, 20)
        self.explore_hyperparameter_tuning(
            'alpha',
            alpha_range,
            f'{self.classifier_name} ({self.dataset.name} dataset) - Alpha', log_scale=True)

    def learning_rate_analysis(self):
        learning_rate_range = np.logspace(-5, 0, 6)
        self.explore_hyperparameter_tuning(
            'learning_rate_init',
            learning_rate_range,
            f'{self.classifier_name} ({self.dataset.name} dataset) - Learning Rate', log_scale=True)

    def epoch_analysis(self):
        number_of_epochs = np.arange(1, 10000)
        parameters = self.default_classifier_parameters
        parameters['warm_start'] = True
        parameters['max_iter'] = 1
        classifier = self.classifier_class(**parameters)
        train_loss = []
        train_scores = []
        test_scores = []
        for i in number_of_epochs:
            print(i)
            classifier.fit(self.dataset.x_train, self.dataset.y_train)
            train_loss.append(classifier.loss_)
            train_scores.append(
                sklearn.metrics.accuracy_score(
                    self.dataset.y_train,
                    classifier.predict(
                        self.dataset.x_train)))
            test_scores.append(
                sklearn.metrics.accuracy_score(
                    self.dataset.y_test,
                    classifier.predict(
                        self.dataset.x_test)))
        ultra_simple_plot(
            number_of_epochs,
            'Epochs',
            train_loss,
            'Train Loss',
            f'{self.get_title_preffix()}Training Loss')
        plot_without_erros(number_of_epochs, 'Epochs', train_scores, test_scores,
                           f'{self.get_title_preffix()}Training and Cross-validation Scores')


class SVMExperiment(Experiment):
    def __init__(self, dataset):
        super().__init__(
            classifier_name='SVM',
            classifier_class=sklearn.svm.SVC,
            default_classifier_parameters={'kernel': 'linear'},
            dataset=dataset)

    def get_parameters_for_tuning(self):
        pass

    def model_complexity_analysis(self):
        pass
