"""
Baseline implementation of an algorithm to extract rules while ignoring the
given DNN's predictions. It simply uses a vanilla decision tree learner
and extracts rules from it.
"""

import numpy as np
import pandas as pd

from remix.logic_manipulator.merge import merge
from remix.rules.C5 import C5
from remix.rules.cart import cart_rules, random_forest_rules
from remix.rules.ruleset import Ruleset

################################################################################
## Exposed Methods
################################################################################


def extract_rules(
    model,
    train_data,
    train_labels,
    winnow=True,
    threshold_decimals=None,
    min_cases=15,
    feature_names=None,
    output_class_names=None,
    tree_extraction_algorithm_name="C5.0",
    ccp_prune=True,
    estimators=30,
    regression=False,
    tree_max_depth=None,
    **kwargs,
):
    """
    Extracts a set of rules using the requested tree extraction algorithm and
    IGNORES the provided model.

    :param keras.Model model: An input instantiated Keras Model object from
        which we will extract rules from.
    :param np.ndarray train_data: A tensor of shape [N, m] with N training
        samples which have m features each.
    :param np.ndarray train_labels: A 1-D tensor of shape [N] with the labels
        corresponding to the input training samples in `train_data`.
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
    :param str tree_extraction_algorithm_name: One of ["C5.0", "CART",
        "random_forest"] indicating which rule extraction algorithm to use for
        extracting rules.
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
    # C5 requires y to be a pd.Series
    y = pd.Series(train_labels)

    if isinstance(tree_extraction_algorithm_name, str):
        if tree_extraction_algorithm_name.lower() in ["c5.0", "c5", "see5"]:
            tree_extraction_algorithm = C5
            algo_kwargs = dict(
                prior_rule_confidence=1,
                winnow=winnow,
                threshold_decimals=threshold_decimals,
                min_cases=min_cases,
            )
        elif tree_extraction_algorithm_name.lower() == "cart":
            tree_extraction_algorithm = cart_rules
            algo_kwargs = dict(
                threshold_decimals=threshold_decimals,
                min_cases=min_cases,
                ccp_prune=ccp_prune,
                regression=regression,
                max_depth=tree_max_depth,
            )
        elif tree_extraction_algorithm_name.lower() == "random_forest":
            tree_extraction_algorithm = random_forest_rules
            algo_kwargs = dict(
                threshold_decimals=threshold_decimals,
                min_cases=min_cases,
                estimators=estimators,
                regression=regression,
                max_depth=tree_max_depth,
            )
        else:
            raise ValueError(
                f'Unsupported tree extraction algorithm '
                f'{tree_extraction_algorithm_name}. Supported algorithms are '
                '"C5.0", "CART", and "random_forest".'
            )

    assert len(train_data) == len(y), \
        'Unequal number of data instances and predictions'

    # We can extract the number of output classes from the model itself
    num_classes = model.layers[-1].output_shape[-1]

    train_data = pd.DataFrame(
        data=train_data,
        columns=[
            feature_names[i] if feature_names is not None else f"h_0_{i}"
            for i in range(train_data.shape[-1])
        ],
    )
    rule_conclusion_map = {}
    for i in range(num_classes):
        if output_class_names is not None:
            rule_conclusion_map[i] = output_class_names[i]
        else:
            rule_conclusion_map[i] = i

    rules = tree_extraction_algorithm(
        x=train_data,
        y=y,
        rule_conclusion_map=rule_conclusion_map,
        **algo_kwargs
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
