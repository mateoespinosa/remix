"""
Runs a grid search over hyper-parameters of our model to try and find the
best hyper-parameterization.
"""

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import itertools
import json
import logging
import numpy as np
import sklearn
import tensorflow as tf

from .train import model_fn


def deserialize_best_params(best_params_file):
    # Helper function to deserialize the set of best parameters obtained by
    # a previous grid search.
    indicator = " using "
    with open(best_params_file, 'r') as f:
        best_params_line = f.readline()
        indicator_ind = best_params_line.find(indicator)
        best_params_serialized = \
            best_params_line[indicator_ind + len(indicator):]
        return json.loads(best_params_serialized)


def serialize_best_params(grid_result, best_params_file):
    # Helper function to serialize the set of best parameters we found in our
    # grid search.
    with open(best_params_file, 'w') as file:
        file.write(
            f"Best: {grid_result.best_score_} using "
            f"{json.dumps(grid_result.best_params_)}\n"
        )

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            file.write(f"{mean} ({stdev}) with: {json.dumps(param)}\n")

    # And return the best parameters we found
    return grid_result.best_params_


class CustomMetricKerasClassifier(KerasClassifier):
    """
    Helper class to wrap a Keras model and turn it into a sklearn classifier
    whose scoring function can be given by any metric in the Keras model.
    """

    def __init__(self, build_fn=None, metric_name='accuracy', **sk_params):
        """
        metric_name represents the name of a valid metric in the given model.
        """

        super(CustomMetricKerasClassifier, self).__init__(
            build_fn=build_fn,
            **sk_params
        )
        self.metric_name = metric_name

    def get_params(self, **params):  # pylint: disable=unused-argument
        """
        Gets parameters for this estimator.
        """
        res = super(CustomMetricKerasClassifier, self).get_params(**params)
        res.update({'metric_name': self.metric_name})
        return res

    def score(self, x, y, **kwargs):
        """
        Returns the requested metric on the given test data and labels.
        """
        y = np.searchsorted(self.classes_, y)
        kwargs = self.filter_sk_params(
            tf.python.keras.models.Sequential.evaluate,
            kwargs
        )
        outputs = self.model.evaluate(x, y, **kwargs)
        if not isinstance(outputs, list):
            outputs = [outputs]

        for name, output in zip(self.model.metrics_names, outputs):
            if name == self.metric_name:
                return output
        raise ValueError(
            'The model is not configured to compute metric with name '
            f'{self.metric_name}. All available metrics are '
            f'{self.model.metrics_names}.'
        )


def grid_search(X, y, manager=None, X_val=None, y_val=None, num_outputs=2):
    """
    Performs a grid search over the hyper-parameters of our model using
    training dataset X with labels y.
    """
    logging.warning(
        'Performing grid search over hyper parameters from scratch. '
        'This will take a while...'
    )
    if manager and manager.GRID_SEARCH_PARAMS:
        batch_sizes = manager.GRID_SEARCH_PARAMS.get("batch_sizes", [16, 32])
        epochs = manager.GRID_SEARCH_PARAMS.get(
            "epochs",
            [50, 100, 150],
        )
        learning_rates = manager.GRID_SEARCH_PARAMS.get(
            "learning_rates",
            [1e-3, 1e-4],
        )
        layer_sizes = manager.GRID_SEARCH_PARAMS.get(
            "layer_sizes",
            ([128, 64, 32], [64, 32]),
        )
        activations = manager.GRID_SEARCH_PARAMS.get(
            "activations",
            ["tanh", "elu"]
        )
        loss_functions = manager.GRID_SEARCH_PARAMS.get(
            "loss_functions",
            ["softmax_xentr", "sigmoid_xentr"]
        )
        dropout_rates = manager.GRID_SEARCH_PARAMS.get(
            "dropout_rates",
            [0, 0.2]
        )
    else:
        batch_sizes = [16, 32]
        epochs = [50, 100, 150, 200]
        learning_rates = [1e-3]
        layer_sizes = (
            [128, 64, 32],
            [64, 32],
        )
        activations = ["tanh", "elu", "relu"]
        loss_functions = ["softmax_xentr", "sigmoid_xentr"]
        dropout_rates = [0, 0.2]

    param_grid = dict(
        input_features=[X.shape[-1]],
        num_outputs=[num_outputs],
        batch_size=batch_sizes,
        epochs=epochs,
        layer_units=list(itertools.product(*layer_sizes)),
        activation=activations,
        last_activation=[None],
        loss_function=loss_functions,
        learning_rate=learning_rates,
        dropout_rate=dropout_rates,
    )

    model = CustomMetricKerasClassifier(
        build_fn=model_fn,
        # Given class imbalance, we will score our fits based on AUC rather
        # than plain accuracy.
        metric_name=manager.GRID_SEARCH_PARAMS.get(
            "metric_name",
            "accuracy",
        ) if manager else "accuracy",
        verbose=0,
    )

    grid = sklearn.model_selection.GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=-1,
        cv=3,
        verbose=(
            int(logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        ),
    )
    # We will weight our classes according to their distribution when fitting
    # our model to compensate for the class imbalance in this graph
    class_weights = dict(enumerate(
        sklearn.utils.class_weight.compute_class_weight(
            'balanced',
            np.unique(y),
            y
        )
    ))
    grid_result = grid.fit(
        X,
        tf.keras.utils.to_categorical(y),
        class_weight=class_weights,
    )

    logging.debug('Grid Search for hyper parameters complete.')
    logging.debug(
        f"Best: {grid_result.best_score_} using {grid_result.best_params_}"
    )
    return grid_result
