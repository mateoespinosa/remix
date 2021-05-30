"""
Module for learning and extracting rules using CART decision trees.
"""

from remix.logic_manipulator.merge import merge
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np

from .C5 import truncate
from .rule import Rule
from .term import Term


def tree_to_ruleset(
    tree,
    threshold_decimals=None,
    feature_names=None,
    multilabel=False,
    rule_conclusion_map=None,
    scalar_confidence=True,
    prior_rule_confidence=1,
    regression=False,
):
    """
    Helper function that turns a sklearn decision tree into a set of rules.
    """
    rules_set = set()
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value
    rule_conclusion_map = rule_conclusion_map or {}

    def _recurse(node_id, previous_premise):
        if children_left[node_id] != children_right[node_id]:
            # Then this is a split node!
            if feature_names is not None:
                var = feature_names[feature[node_id]]
            else:
                var = feature[node_id]
            thresh = threshold[node_id]
            if threshold_decimals is not None:
                thresh = truncate(
                    thresh,
                    threshold_decimals,
                )
            new_term = Term(
                variable=var,
                operator="<=",
                threshold=thresh,
            )
            lhs_premise = previous_premise + [new_term]

            thresh = threshold[node_id]
            if threshold_decimals is not None:
                thresh = truncate(
                    thresh,
                    threshold_decimals,
                )
            new_term = Term(
                variable=var,
                operator=">",
                threshold=thresh,
            )
            rhs_premise = previous_premise + [new_term]

            # And recurse using this node's children
            _recurse(children_left[node_id], lhs_premise)
            _recurse(children_right[node_id], rhs_premise)
        else:
            # Else this is a leaf node so let's go ahead and add its conclusion
            # as a rule using the previously given premise
            if multilabel:
                # Then this is the multi-label setting
                conclusion = []
                confidence = []
                for i, binary_counts in enumerate(value[node_id]):
                    false_count, true_count = binary_counts
                    if false_count > true_count:
                        # Then we classify this as false
                        conclusion.append(
                            rule_conclusion_map.get((i, False), False)
                        )
                        class_count = false_count
                    else:
                        # Otherwise we classify this as true
                        conclusion.append(
                            rule_conclusion_map.get((i, True), True)
                        )
                        class_count = true_count

                    confidence.append(
                        prior_rule_confidence * (
                            class_count / (true_count + false_count)
                        )
                    )
                if scalar_confidence:
                    confidence = np.mean(confidence)
                else:
                    confidence = tuple(confidence)
                conclusion = tuple(conclusion)
            elif regression:
                conclusion = value[node_id][0][0]
                # We cannot guarantee any confidence in here so we will leave it
                # as None
                confidence = 0
            else:
                # Else this is the multi-class setting
                assert len(value[node_id]) == 1, \
                    "In multiclass setting expected leafs to have a single val."
                conclusion = None
                max_count = 0
                total_count = 0
                for label_id, count in enumerate(value[node_id][0]):
                    total_count += count
                    if count >= max_count:
                        max_count = count
                        conclusion = rule_conclusion_map.get(label_id, label_id)
                confidence = prior_rule_confidence * (
                    max_count / (total_count)
                )

            rules_set.add(
                Rule.from_term_set(
                    premise=previous_premise,
                    conclusion=conclusion,
                    confidence=confidence,
                )
            )

    # Call our method from the root
    _recurse(0, [])

    # And return the ruleset after merging them
    return merge(rules_set)


def cart_rules(
    x,
    y,
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_cases=None,
    max_features=None,
    seed=42,
    max_leaf_nodes=None,
    class_weight=None,
    threshold_decimals=None,
    ccp_prune=True,
    rule_conclusion_map=None,
    prior_rule_confidence=1,
    regression=False,
):
    """
    Extracts a ruleset from learning a CART decision tree that maps datapoints
    in x to labels in y.

    :param 2D np.ndarray x: The training samples used to construct the decision
    tree from which we will extract rules from.
    :param 1D/2D np.ndarray y: The labels to be used in our training data. If it
        is given as a 2D array, then we expect this to be a multi-label
        classification problem.
    :param "gini" or "entropy" criterion: The criterion used for splitting on a
        given feature.
    :param int max_depth: The maximum depth allowed for the decision tree.
    :param int, float  min_cases: The minimum number of samples each leaf node
        should contain given as an explicit value or a percent of the entire
        training dataset.
    :param int, float or {“auto”, “sqrt”, “log2”} max_features: The max number
        of features to be used for each split. Can be an explicit number, a
        fraction, or a function of the available number of features.
    :param int seed: The random seed to be used.
    :param int max_leaf_nodes: The maximum number of leaf nodes allowed to be
        used for the decision tree.
    :param Dict[y, float] class_weight: Optional dictionary to be passed for
        weighting different classes differently.
    :param int threshold_decimals: Max number of decimals to be used in a given
        threshold. If None, then no limit is applied.
    :param True ccp_prune: Whether or not we perform post-growing CCP pruning.
    :param Dict[y, Any] rule_conclusion_map: A map between possible output
        labels and what they imply. If not given, or if a label is missing, we
        will simply use the plain label as the conclusion.

    :returns Set[Rule]: a set of rules extracted from the decision tree we grew
        using the training data.
    """
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if regression:
        dt_class = DecisionTreeRegressor
        extra_params = {
            "criterion": "mse",
        }
    else:
        dt_class = DecisionTreeClassifier
        extra_params = {
            "class_weight": class_weight,
            "criterion": criterion,
        }

    dt = dt_class(
        max_depth=max_depth,
        min_samples_leaf=min_cases,
        max_features=max_features,
        splitter=splitter,
        random_state=seed,
        max_leaf_nodes=max_leaf_nodes,
        **extra_params,
    )
    if ccp_prune:
        path = dt.cost_complexity_pruning_path(x, y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        dt = dt_class(
            max_depth=max_depth,
            min_samples_leaf=min_cases,
            max_features=max_features,
            splitter=splitter,
            random_state=seed,
            max_leaf_nodes=max_leaf_nodes,
            ccp_alpha=ccp_alphas[len(ccp_alphas)//2 - 1],
            **extra_params,
        )

    dt.fit(x, y)

    return tree_to_ruleset(
        dt.tree_,
        threshold_decimals=threshold_decimals,
        feature_names=x.columns,
        multilabel=(len(y.shape) == 2),
        rule_conclusion_map=rule_conclusion_map,
        prior_rule_confidence=prior_rule_confidence,
        regression=regression,
    )


def random_forest_rules(
    x,
    y,
    criterion="gini",
    max_depth=None,
    min_cases=15,
    max_features=None,
    seed=42,
    max_leaf_nodes=None,
    class_weight=None,
    threshold_decimals=None,
    estimators=30,
    rule_conclusion_map=None,
    bootstrap=True,
    prior_rule_confidence=1,
    regression=False,
):
    """
    Extracts a ruleset from learning a random forest that maps datapoints in x
    to labels in y.

    :param 2D np.ndarray x: The training samples used to construct the random
        forest from which we will extract rules from.
    :param 1D/2D np.ndarray y: The labels to be used in our training data. If it
        is given as a 2D array, then we expect this to be a multi-label
        classification problem.
    :param "gini" or "entropy" criterion: The criterion used for splitting on a
        given feature.
    :param int max_depth: The maximum depth allowed for each tree we grow in our
        forest.
    :param int, float  min_cases: The minimum number of samples each leaf node
        should contain given as an explicit value or a percent of the entire
        training dataset.
    :param int, float or {“auto”, “sqrt”, “log2”} max_features: The max number
        of features to be used for each split. Can be an explicit number, a
        fraction, or a function of the available number of features.
    :param int seed: The random seed to be used.
    :param int max_leaf_nodes: The maximum number of leaf nodes allowed to be
        used for any given tree in the forest.
    :param Dict[y, float] class_weight: Optional dictionary to be passed for
        weighting different classes differently.
    :param int threshold_decimals: Max number of decimals to be used in a given
        threshold. If None, then no limit is applied.
    :param int estimators: The number of trees to grow in our random forest.
    :param Dict[y, Any] rule_conclusion_map: A map between possible output
        labels and what they imply. If not given, or if a label is missing, we
        will simply use the plain label as the conclusion.
    :param bool bootstrap: Whether bootstrapping is performed when growing the
        random forest.

    :returns Set[Rule]: a set of rules extracted from the random forest we grew
        using the training data.
    """
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if regression:
        rf_class = RandomForestRegressor
        extra_params = {
            "criterion": "mse",
        }
    else:
        rf_class = RandomForestClassifier
        extra_params = {
            "class_weight": class_weight,
            "criterion": criterion,
        }

    dt = rf_class(
        n_estimators=estimators,
        max_depth=max_depth,
        min_samples_leaf=min_cases,
        max_features=max_features,
        random_state=seed,
        max_leaf_nodes=max_leaf_nodes,
        bootstrap=bootstrap,
        **extra_params,
    )

    dt.fit(x, y)
    result_rules = set()
    for tree in dt.estimators_:
        result_rules.update(
            tree_to_ruleset(
                tree.tree_,
                threshold_decimals=threshold_decimals,
                feature_names=x.columns,
                multilabel=len(y.shape) == 2,
                rule_conclusion_map=rule_conclusion_map,
                prior_rule_confidence=prior_rule_confidence,
                regression=regression,
            )
        )
    return result_rules
