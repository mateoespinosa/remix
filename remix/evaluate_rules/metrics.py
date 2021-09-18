"""
Metrics for evaluation of a given ruleset that was extracted from a network.
"""

################################################################################
## Exposed Methods
################################################################################


def fidelity(predicted_labels, network_labels):
    """
    Evaluate fidelity of rules generated i.e. how well do they mimic the
    performance of the Neural Network.

    :param np.array predicted_labels:  The predicted labels from our rule set.
    :param np.array network_labels:    The labels as predicted by our original
        neural network.

    :returns float: How many labels were predicted in our rule set as they
        were predicted in the original NN model.
    """
    assert (len(predicted_labels) == len(network_labels)), \
        "Error: number of labels inconsistent !"

    return sum(predicted_labels == network_labels) / len(predicted_labels)


def comprehensibility(rules):
    """
    Computes a dictionary containing statistics on the lengths and composition
    of the rules provided.

    The number of rules per class is defined as the number of conjunctive
    clauses in a class' DNF.

    :param Iterable[Rule] rules: The rules whose compressibility we want to
        analyze.
    :returns Dictionary[str, object]: Returns a dictionary with statistics
        of the given set of rules.
    """
    all_ruleset_info = []
    all_terms = set()
    output_classes = set()
    for class_ruleset in rules:
        class_encoding = class_ruleset.conclusion

        # Number of rules in that class
        n_rules_in_class = len(class_ruleset.premise)

        #  Get min max average number of terms in a clause
        min_n_terms = float('inf')
        max_n_terms = -min_n_terms
        total_n_terms = 0
        for clause in class_ruleset.premise:
            # Number of terms in the clause
            n_clause_terms = len(clause.terms)
            min_n_terms = min(n_clause_terms, min_n_terms)
            max_n_terms = max(n_clause_terms, max_n_terms)
            total_n_terms += n_clause_terms
            for term in clause.terms:
                all_terms.add(term)

        av_n_terms_per_rule = (
            (total_n_terms / n_rules_in_class) if n_rules_in_class else 0
        )

        class_ruleset_info = [
            class_encoding,
            n_rules_in_class,
            min_n_terms,
            max_n_terms,
            av_n_terms_per_rule,
        ]

        all_ruleset_info.append(class_ruleset_info)

    output_classes, n_rules, min_n_terms, max_n_terms, av_n_terms_per_rule = zip(
        *all_ruleset_info
    )
    return dict(
        output_classes=output_classes,
        n_rules_per_class=n_rules,
        min_n_terms=min_n_terms,
        max_n_terms=max_n_terms,
        av_n_terms_per_rule=av_n_terms_per_rule,
        n_unique_terms=len(all_terms),
    )
