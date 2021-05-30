"""
Methods for satisfiability checks.
"""

from .utils import terms_set_to_variable_dict
from ..rules.term import TermOperator


def is_satisfiable(clause):
    """
    Determines whether or not the clause is satisfiable.
    Unsatisfiable two or more terms are mutually exclusive i.e. they restrict
    a variable into

    :param ConjuctiveClause clause: The clause whose satisfiability we want to
        determine.

    :returns boolean: True if the specified clause is satisfiable, False
        otherwise.
    """

    # Empty Clause is always satisfiable
    if len(clause.terms) == 0:
        return True

    # Check if variables min value >= max value
    variable_conds = terms_set_to_variable_dict(clause.terms)
    for var in variable_conds.keys():
        # Look at this variable's lower and upper bounds
        upper_bounds = variable_conds[var][TermOperator.LessThanEq]
        lower_bounds = variable_conds[var][TermOperator.GreaterThan]

        if upper_bounds and lower_bounds and (
            min(lower_bounds) >= max(upper_bounds)
        ):
            # Then we have reached a situation where both conditions are
            # mutually exclusive
            return False

    # If we reached this point, then all variables are satisfiable
    return True


def remove_unsatisfiable_clauses(clauses):
    """
    Removes all unsatisfiable clauses from the given set of clauses.
    Returns a new set (rather than modifying the provided one.)
    :param Set[ConjuctiveClause] clauses: The set of clauses we want to filter.

    :returns Set[ConjuctiveClause]: Equivalent set of clauses with all
        unsatisfiable clauses removed.
    """
    return set([
        clause for clause in clauses
        if is_satisfiable(clause)
    ])
