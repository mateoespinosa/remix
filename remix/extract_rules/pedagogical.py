"""
Baseline implementation of an algorithm to extract rules from a DNN using a
simple pedagogical algorithm: we extract a decision tree that maps input
features with the model's outputs.
"""

import numpy as np
import pandas as pd


from remix.logic_manipulator.merge import merge
from remix.rules.C5 import C5
from remix.rules.cart import cart_rules, random_forest_rules
from remix.rules.ruleset import Ruleset
from remix.utils.data_handling import stratified_k_fold_split

################################################################################
## Exposed Methods
################################################################################


def extract_rules(
    model,
    train_data,
    winnow=True,
    threshold_decimals=None,
    min_cases=15,
    feature_names=None,
    output_class_names=None,
    max_number_of_samples=None,
    tree_extraction_algorithm_name="C5.0",
    trials=1,
    tree_max_depth=None,
    ccp_prune=True,
    estimators=30,
    regression=False,
    **kwargs,
):
    """
    Extracts a set of rules which imitates given the provided model in a
    pedagogical manner using C5 on the outputs and inputs of the network.

    :param keras.Model model: An input instantiated Keras Model object from
        which we will extract rules from.
    :param np.ndarray train_data: A tensor of shape [N, m] with N training
        samples which have m features each.
    :param logging.VerbosityLevel verbosity: The verbosity level to use for this
        function.
    :param int threshold_decimals: The maximum number of decimals a threshold in
        the generated ruleset may have. If None, then we impose no limit.
    :param bool winnow: Whether or not we use winnowing when using
        C5.0 for rule extraction
    :param int min_cases: The minimum number of samples we must have to perform
        a split in a decision tree.
    :param List[str] feature_names: List of feature names to be used for
        generating our rule set. If None, then we will assume all input features
        are named `h_0_0`, `h_0_1`, `h_0_2`, etc.
    :param List[str] output_class_names: List of output class names to be used
        for generating our rule set. If None, then we will assume all output
        are named `h_{d+1}_0`, `h_{d+1}_1`, `h_{d+1}_2`, etc where `d` is the
        number of hidden layers in the network.
    :param Or[int, float] max_number_of_samples: The maximum number of samples
        to use from the training data. This corresponds to how much we will
        subsample the input training data before using it to construct
        rules. If given as a number in [0, 1], then this represents the fraction
        of the input set which will be used during rule extraction. If None,
        then we will use the entire training set as given.
    :param str tree_extraction_algorithm_name: One of ["C5.0", "CART",
        "random_forest"] indicating which rule extraction algorithm to use for
        extracting rules.
    :param int trials: The number of sampling trials to use when using bagging
        for C5.0 in rule extraction.
    :param int tree_max_depth: max tree depth when using CART or random_forest
        for rule set extraction.
    :param bool ccp_prune: whether or not we do post-hoc CCP prune to CART trees
        if CART is used for rule induction.
    :param int estimators: The number of trees to use if using random_forest
        for rule extraction.
    :param bool regression: whether or not we are dealing with a regression
        task or a classification task.
    :param Dict[str, Any] kwargs: The keywords arguments used for easier
        integration with other rule extraction methods.

    :returns Ruleset: the set of rules extracted from the given model.
    """

    if tree_extraction_algorithm_name.lower() in ["c5.0", "c5", "see5"]:
        algo_call = C5
        algo_kwargs = dict(
            winnow=winnow,
            threshold_decimals=threshold_decimals,
            trials=trials,
        )
    elif tree_extraction_algorithm_name.lower() == "cart":
        algo_call = cart_rules
        algo_kwargs = dict(
            threshold_decimals=threshold_decimals,
            ccp_prune=ccp_prune,
            class_weight="balanced",
            max_depth=tree_max_depth,
            regression=regression,
        )
    elif tree_extraction_algorithm_name.lower() == "random_forest":
        algo_call = random_forest_rules
        algo_kwargs = dict(
            threshold_decimals=threshold_decimals,
            estimators=estimators,
            regression=regression,
        )
    else:
        raise ValueError(
            f'Unsupported tree extraction algorithm '
            f'{tree_extraction_algorithm_name}. Supported algorithms are '
            '"C5.0", "CART", and "random_forest".'
        )

    # Determine whether we want to subsample our training dataset to make it
    # more scalable or not
    sample_fraction = 0
    if max_number_of_samples is not None:
        if max_number_of_samples < 1:
            sample_fraction = max_number_of_samples
        elif max_number_of_samples < train_data.shape[0]:
            sample_fraction = max_number_of_samples / train_data.shape[0]

    if sample_fraction:
        [(new_indices, _)] = stratified_k_fold_split(
            X=train_data,
            n_folds=1,
            test_size=(1 - sample_fraction),
            random_state=42,
        )
        train_data = train_data[new_indices, :]

    # y = output classifications of neural network. C5 requires y to be a
    # pd.Series
    nn_model_predictions = model.predict(train_data)
    if regression:
        nn_model_predictions = np.squeeze(nn_model_predictions, axis=-1)
    else:
        nn_model_predictions = np.argmax(nn_model_predictions, axis=-1)
    y = pd.Series(nn_model_predictions)

    assert len(train_data) == len(y), \
        'Unequal number of data instances and predictions'

    # We can extract the number of output classes from the model itself
    num_classes = model.layers[-1].output_shape[-1]

    # Use C5 to extract rules using only input and output values of the network
    # C5 returns disjunctive rules with conjunctive terms
    train_data = pd.DataFrame(
        data=train_data,
        columns=[
            feature_names[i] if feature_names is not None else f"h_0_{i}"
            for i in range(train_data.shape[-1])
        ],
    )

    if regression:
        rule_conclusion_map = None
    else:
        rule_conclusion_map = {}
        for i in range(num_classes):
            if output_class_names is not None:
                rule_conclusion_map[i] = output_class_names[i]
            else:
                rule_conclusion_map[i] = i

    rules = algo_call(
        x=train_data,
        y=y,
        rule_conclusion_map=rule_conclusion_map,
        prior_rule_confidence=1,
        min_cases=min_cases,
        **algo_kwargs,
    )

    # Merge rules so that they are in Disjunctive Normal Form
    # Now there should be only 1 rule per rule conclusion
    # Ruleset is encapsulated/represented by a DNF rule
    # dnf_rules is a set of rules
    dnf_rules = merge(rules)

    return Ruleset(
        dnf_rules,
        feature_names=feature_names,
        output_class_names=output_class_names,
        regression=regression,
    )
