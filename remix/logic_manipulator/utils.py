"""
Helper functions for our logic manipulators.
"""
from collections import defaultdict
from ..rules.term import TermOperator


def terms_set_to_variable_dict(terms):
    """
    Converts a set of terms into a dictionary mapping each variable in each term
    to a map between operators ('>' and '<=') and thresholds used in those
    variables with those operators.

    :param Iterable[Terms] terms: The terms which will source our dictionary.

    :returns Dict[str, Dict[TermOperator, Set[float]]]: the corresponding
        dictionary of variable values.
    """
    # Convert set of conditions into dictionary
    variable_conditions = defaultdict(lambda: {
        TermOp: set()
        for TermOp in TermOperator
    })

    for term in terms:
        variable_conditions[term.variable][term.operator].add(term.threshold)

    # Returns {variable_name (str): {TermOperator: Set[Float]}}
    return variable_conditions
