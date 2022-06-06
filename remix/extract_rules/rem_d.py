"""
Main implementation of the vanilla REM-D rule extraction algorithm for DNNs.
"""
import dill
import logging
import numpy as np

from remix.logic_manipulator.substitute_rules import substitute
from remix.rules.C5 import C5
from remix.rules.cart import cart_rules, random_forest_rules
from remix.rules.rule import Rule
from remix.rules.ruleset import Ruleset
from remix.utils.data_handling import stratified_k_fold_split
from remix.utils.parallelism import serialized_function_execute
from multiprocessing import Pool
from tqdm import tqdm  # Loading bar for rule generation

from .utils import ModelCache


################################################################################
## Exposed Methods
################################################################################

def extract_rules(
    model,
    train_data,
    verbosity=logging.INFO,
    last_activation=None,
    threshold_decimals=None,
    winnow_intermediate=True,
    winnow_features=True,
    min_cases=15,
    num_workers=1,  # 1 for original
    feature_names=None,
    output_class_names=None,
    trials=1,  # 1 for original
    block_size=1,  # 1 for original
    merge_repeated_terms=False,  # False for original
    max_number_of_samples=None,
    intermediate_algorithm_name="C5.0",
    estimators=30,
    ccp_prune=True,
    balance_classes=False,
    intermediate_tree_max_depth=None,
    **kwargs,
):
    """
    Extracts a set of rules which imitates given the provided model using the
    algorithm described in the paper.

    :param keras.Model model: An input instantiated Keras Model object from
        which we will extract rules from.
    :param np.ndarray train_data: A tensor of shape [N, m] with N training
        samples which have m features each.
    :param logging.VerbosityLevel verbosity: The verbosity level to use for this
        function.
    :param str last_activation: Either "softmax" or "sigmoid" indicating which
        activation function should be applied to the last layer of the given
        model if last function is fused with loss. If None, then no activation
        function is applied.
    :param int threshold_decimals: The maximum number of decimals a threshold in
        the generated ruleset may have. If None, then we impose no limit.
    :param bool winnow_intermediate: Whether or not we use winnowing when using
        C5.0 for intermediate hidden layers.
    :param bool winnow_features: Whether or not we use winnowing when extracting
        rules in the features layer.
    :param int min_cases: The minimum number of samples we must have to perform
        a split in a decision tree.
    :param int num_workers: Maximum number of working processes to be spanned
        when extracting rules.
    :param List[str] feature_names: List of feature names to be used for
        generating our rule set. If None, then we will assume all input features
        are named `h_0_0`, `h_0_1`, `h_0_2`, etc.
    :param List[str] output_class_names: List of output class names to be used
        for generating our rule set. If None, then we will assume all output
        are named `h_{d+1}_0`, `h_{d+1}_1`, `h_{d+1}_2`, etc where `d` is the
        number of hidden layers in the network.
    :param int trials: The number of sampling trials to use when using bagging
        for C5.0 rule extraction.
    :param int block_size: The hidden layer sampling frequency. That is, how
        often will we use a hidden layer in the input network to extract an
        intermediate rule set from it.
    :param bool merge_repeated_terms: If set, we will only extract rules to
        approximate a term but not its negation. That way its negation's
        substitution would come from using that tree's negative class.
    :param Or[int, float] max_number_of_samples: The maximum number of samples
        to use from the training data. This corresponds to how much we will
        subsample the input training data before using it to construct
        intermediate and clause-wise rules. If given as a number in [0, 1], then
        this represents the fraction of the input set which will be used during
        rule extraction. If None, then we will use the entire training set as
        given.
    :param Dict[str, Any] kwargs: The keywords arguments used for easier
        integration with other rule extraction methods.

    :returns Ruleset: the set of rules extracted from the given model.
    """

    # First find out which algorithm to use for rule extraction
    if intermediate_algorithm_name.lower() in ["c5.0", "c5", "see5"]:
        intermediate_algo_call = C5
        intermediate_algo_kwargs = dict(
            winnow=winnow_intermediate,
            threshold_decimals=threshold_decimals,
            trials=trials,
        )
    elif intermediate_algorithm_name.lower() == "cart":
        intermediate_algo_call = cart_rules
        intermediate_algo_kwargs = dict(
            threshold_decimals=threshold_decimals,
            ccp_prune=ccp_prune,
            max_depth=intermediate_tree_max_depth,
        )
        if balance_classes:
            intermediate_algo_kwargs["class_weight"] = "balanced"
    elif intermediate_algorithm_name.lower() == "random_forest":
        intermediate_algo_call = random_forest_rules
        intermediate_algo_kwargs = dict(
            threshold_decimals=threshold_decimals,
            estimators=estimators,
            max_depth=intermediate_tree_max_depth,
        )
    else:
        raise ValueError(
            f'Unsupported tree extraction algorithm '
            f'{intermediate_algorithm_name}. Supported algorithms are '
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

    # First we will instantiate a cache of our given keras model to obtain all
    # intermediate activations
    cache_model = ModelCache(
        keras_model=model,
        train_data=train_data,
        last_activation=last_activation,
        feature_names=feature_names,
        output_class_names=output_class_names,
    )

    # Now time to actually extract our set of rules
    dnf_rules = set()

    # Compute our total looping space for purposes of logging our progress
    output_layer = len(model.layers) - 1
    input_hidden_acts = list(range(0, output_layer, block_size))
    output_hidden_acts = input_hidden_acts[1:] + [output_layer]

    num_classes = model.layers[-1].output_shape[-1]
    total_loop_volume = num_classes * (len(input_hidden_acts) - 1)

    with tqdm(
        total=total_loop_volume,
        disable=(verbosity == logging.WARNING),
    ) as pbar:
        for output_class_idx in range(num_classes):
            if output_class_names:
                output_class_name = output_class_names[output_class_idx]
            else:
                output_class_name = str(output_class_idx)

            # Initial output layer rule
            class_rule = Rule.initial_rule(
                output_class=output_class_name,
                # If we use sigmoid cross-entropy loss, then this threshold
                # becomes 0.5 and does not depend on the number of classes.
                # Also if activation function is not provided, we will default
                # to using 0.5 thresholds.
                threshold=(
                    (1 / num_classes) if (last_activation == "softmax") else 0.5
                ),
            )

            # Extract layer-wise rules
            for hidden_layer, next_hidden_layer in zip(
                reversed(input_hidden_acts),
                reversed(output_hidden_acts),
            ):
                # Obtain our cached predictions
                predictors = cache_model.get_layer_activations(
                    layer_index=hidden_layer,
                )

                # We will generate an intermediate ruleset for this layer
                intermediate_rules = Ruleset(
                    feature_names=list(predictors.columns)
                )

                # And time to call C5.0 for each term
                term_confidences = \
                    class_rule.get_terms_with_conf_from_rule_premises()
                partial_terms = list(term_confidences.keys())
                # And get rid of terms that are negations of each other
                if merge_repeated_terms:
                    terms = set()
                    for term in partial_terms:
                        if term.negate() in terms:
                            # Then no need to add this guy
                            continue
                        terms.add(term)
                    terms = list(terms)
                else:
                    terms = partial_terms

                num_terms = len(terms)

                # We preemptively extract all the activations of the next layer
                # so that we can serialize the function below using dill.
                # Otherwise, we will hit issues due to Pandas dataframes not
                # being compatible with dill/pickle
                next_layer_activations = cache_model.get_layer_activations(
                    layer_index=next_hidden_layer,
                )

                # Helper method to extract rules from the terms coming from a
                # hidden layer and a given label. We encapsulate it as an
                # anonymous function for it to be able to be used in a
                # multi-process fashion.
                def _extract_rules_from_term(term, i=None, pbar=None):
                    if pbar and (i is not None):
                        pbar.set_description(
                            f'Extracting rules for term {i}/'
                            f'{num_terms} {term} of layer '
                            f'{hidden_layer} for class {output_class_name}'
                        )

                    #  y1', y2', ...ym' = t(h(x1)), t(h(x2)), ..., t(h(xm))
                    target = term.apply(
                        next_layer_activations[str(term.variable)]
                    )
                    logging.debug(
                        f"\tA total of {np.count_nonzero(target)}/"
                        f"{len(target)} training samples satisfied {term}."
                    )

                    prior_rule_confidence = term_confidences[term]
                    rule_conclusion_map = {
                        True: term,
                        False: term.negate(),
                    }
                    new_rules = intermediate_algo_call(
                        x=predictors,
                        y=target,
                        rule_conclusion_map=rule_conclusion_map,
                        prior_rule_confidence=prior_rule_confidence,
                        min_cases=min_cases,
                        **intermediate_algo_kwargs,
                    )

                    if pbar:
                        pbar.update(1/num_terms)
                    return new_rules

                # Now compute the effective number of workers we've got as
                # it can be less than the provided ones if we have less terms
                effective_workers = min(num_workers, num_terms)
                if effective_workers > 1:
                    # Them time to do this the multi-process way
                    pbar.set_description(
                        f"Extracting rules for layer {hidden_layer} of with "
                        f"output class {output_class_name} using "
                        f"{effective_workers} new processes for {num_terms} "
                        f"terms"
                    )
                    with Pool(processes=effective_workers) as pool:
                        # Now time to do a multiprocess map call. Because this
                        # needs to operate only on serializable objects, what
                        # we will do is the following: we will serialize each
                        # partition bound and the function we are applying
                        # into a tuple using dill and then the map operation
                        # will deserialize each entry using dill and execute
                        # the provided method
                        serialized_terms = [None for _ in range(len(terms))]
                        for j, term in enumerate(sorted(terms, key=str)):
                            # Let's serialize our (function, args) tuple
                            serialized_terms[j] = dill.dumps(
                                (_extract_rules_from_term, (term,))
                            )

                        # And do the multi-process pooling call
                        new_rulesets = pool.map(
                            serialized_function_execute,
                            serialized_terms,
                        )

                    # And update our bar with only one step as we do not have
                    # the granularity we do in the non-multi-process way
                    pbar.update(1)
                else:
                    # Else we will do it in this same process in one jump
                    new_rulesets = list(map(
                        lambda x: _extract_rules_from_term(
                            term=x[1],
                            i=x[0],
                            pbar=pbar
                        ),
                        enumerate(sorted(terms, key=str), start=1),
                    ))

                # Time to do our simple reduction from our map above by
                # accumulating all the generated rules into a single ruleset
                for ruleset in new_rulesets:
                    intermediate_rules.add_rules(ruleset)
                logging.debug(
                    f'\tGenerated intermediate ruleset for layer '
                    f'{hidden_layer} and output class {output_class_name} has '
                    f'{intermediate_rules.num_clauses()} rules and '
                    f'{intermediate_rules.num_terms()} different terms in it.'
                )

                # Merge rules with current accumulation
                pbar.set_description(
                    f"Substituting rules for layer {hidden_layer} with output "
                    f"class {output_class_name}"
                )
                class_rule = substitute(
                    total_rule=class_rule,
                    intermediate_rules=intermediate_rules,
                )

                if not len(class_rule.premise):
                    pbar.write(
                        f"[WARNING] Found rule with empty premise of for "
                        f"class {output_class_name}."
                    )

            # Finally add this class rule to our solution ruleset
            dnf_rules.add(class_rule)

        pbar.set_description("Done extracting rules from neural network")

    return Ruleset(
        rules=dnf_rules,
        feature_names=feature_names,
        output_class_names=output_class_names,
    )
