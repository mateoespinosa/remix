"""
Very minimal test that verifies that we can call all of our rule extractors
using a simple XOR dataset with 10 features of which 8 are pure noise.
"""

import numpy as np
import pandas as pd
import pytest
import sklearn
import tensorflow as tf

from remix import deep_red_c5
from remix import eclaire
from remix import pedagogical
from remix import rem_d
from remix import rem_t
from remix.utils.data_handling import stratified_k_fold_split

################################################################################
## Global Variables
################################################################################

XOR_DATA_FILE = "tests/xor.csv"

################################################################################
## Model Function Definition
################################################################################


def model_fn(
    input_features,
    layer_units,
    num_outputs,
    activation="tanh",
    optimizer=None,
    last_activation="softmax",
    loss_function="softmax_xentr",
    learning_rate=0.001,
    dropout_rate=0,
    skip_freq=0,
    regression=False,
    decay_rate=1,
    decay_steps=None,
    staircase=False,
):
    """
    Model function to construct our TF model for learning our given task.

    :param int input_features: the number of features our network consumes.
    :param List[int] layer_units: The number of units in each hidden layer.
    :param int num_outputs: The number of outputs in our network.
    :param [str | function] activation: Valid keras activation function name or
        actual function to use as activation between hidden layers.
    :param [str | tf.keras.Optimizer] optimizer: Optimizer to be used for
        training.
    :param str last_activation: valid keras activation to be used
        as our last layer's activation. For now, we only support "softmax" or
        "sigmoid".
    :param str loss_function: valid keras loss function to be used
        after our last layer. This will define the used loss. For
        now, we only support "softmax_xentr" or "sigmoid_xentr".

    :returns tf.keras.Model: Compiled model for training and evaluation.
    """

    # Input layer.
    # NOTE: it is very important to have an explicit Input layer for now rather
    # than using a Sequential model. Otherwise, we will not be able to pick
    # it up correctly during rule generation.
    input_layer = tf.keras.layers.Input(shape=(input_features,))

    # And build our intermediate dense layers
    net = input_layer
    prev_output = None
    for i, units in enumerate(layer_units, start=1):
        if units == 0:
            # Then this is a no-op layer so we will skip it. This is useful
            # for grid searching
            continue

        if skip_freq and ((i % skip_freq) == 0):
            # Then let's add this result with the output of the previous
            # skip block
            net = prev_output + net
            prev_output = None

        net = tf.keras.layers.Dense(
            units,
            activation=activation,
            name=f"dense_{i}",
        )(net)

        if prev_output is None:
            # Then this is the first layer
            prev_output = net

        if ((i % 2) == 0) and dropout_rate:
            # Then let's add a dropout layer in here
            net = tf.keras.layers.Dropout(
                dropout_rate,
                name=f"dropout_{i//2}",
            )(net)

    if (loss_function is None) and last_activation:
        if last_activation == "sigmoid":
            loss_function = "sigmoid_xentr"
        elif last_activation == "softmax":
            loss_function = "softmax_xentr"
    if (last_activation is None) and (not regression):
        if loss_function is None:
            raise ValueError(
                "We were not provided with a loss function or last activation "
                "function in our model."
            )
        elif "_" in loss_function:
            last_activation = loss_function[:loss_function.find("_")]
        else:
            # Default to empty so that we can do substring search
            last_activation = ""

    # And our output layer map
    net = tf.keras.layers.Dense(
        num_outputs,
        name="output_dense",
        # If the last activation is a part of the loss, then we will go ahead
        # and merge them for numerical stability. Else, let's explicitly mark
        # it here.
        activation=(
            None if (last_activation in loss_function) else last_activation
        ) if last_activation else None,
    )(net)

    # Compile Model
    model = tf.keras.models.Model(inputs=input_layer, outputs=net)
    optimizer = optimizer or tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=(decay_steps or 1),
            decay_rate=decay_rate,
            staircase=staircase,

        )
    )
    if loss_function == "softmax_xentr":
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=(last_activation in loss_function),
        )
    elif loss_function == "sigmoid_xentr":
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=(last_activation in loss_function),
        )
    elif loss_function == "mse":
        loss = "mse"
    else:
        raise ValueError(
            f"Unsupported loss {loss_function}. We currently only support "
            "softmax_xentr and softmax_xentr"
        )

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[
            'accuracy',
        ] if (not regression) else [],
    )
    return model


################################################################################
## Tests
################################################################################


@pytest.mark.parametrize(
    'method,name,min_cases,kwargs',
    [
        (eclaire.extract_rules, "eclaire", 2, {}),
        (rem_d.extract_rules, "rem_d", 33, {}),
        (pedagogical.extract_rules, "pedagogical", 2, {}),
        (rem_t.extract_rules, "rem_t", 2, {}),
        (deep_red_c5.extract_rules, "deep_red_c5", 30, {}),
    ]
)
def test_rule_extraction(
    method,
    name,
    min_cases,
    kwargs
):
    # First we load our data
    data = pd.read_csv(XOR_DATA_FILE, sep=',')
    X = data.drop([data.columns[-1]], axis=1).values
    y = data[data.columns[-1]].values
    [(train_indices, test_indices)] = stratified_k_fold_split(
        X=X,
        y=y,
        random_state=42,
        test_size=0.2,
        n_folds=1,
    )
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Compute class weights
    unique_elems = np.unique(y_train)
    class_weights = dict(enumerate(
        sklearn.utils.class_weight.compute_class_weight(
            'balanced',
            unique_elems,
            y_train
        )
    ))

    # Now instantiate the model we will train
    model = model_fn(
        input_features=X_train.shape[-1],
        layer_units=[64, 32, 16],
        num_outputs=2,
        last_activation="softmax",
        loss_function="softmax_xentr",
        activation="tanh",
    )

    # Train it
    model.fit(
        X_train,
        tf.keras.utils.to_categorical(
            y_train,
            num_classes=2,
        ),
        class_weight=class_weights,
        epochs=150,
        batch_size=16,
        verbose=0,
    )

    # Compute its test accuracy
    nn_accuracy = sklearn.metrics.roc_auc_score(
        y_test,
        np.argmax(model.predict(X_test), axis=-1),
    )

    # And time to extract rules from it
    ruleset = method(
        model=model,
        train_data=X_train,
        train_labels=y_train,
        min_cases=min_cases,
        last_activation="softmax",
        feature_names=[f"feat_{i}" for i in range(10)],
        output_class_names=["0", "1"],
        num_workers=4,
    )

    # Serialize for inspection
    ruleset.to_file(f"xor_{name}.rules")

    # Finally, predict its testing accuracy
    ruleset_acc = sklearn.metrics.accuracy_score(
        y_test,
        ruleset.predict(
            X_test,
            num_workers=2,
        )
    )

    print(
        f"For method {name} the DNN accuracy was {nn_accuracy} compared to "
        f"the rule set accuracy {ruleset_acc} which uses "
        f"{ruleset.num_clauses()} total clauses."
    )
