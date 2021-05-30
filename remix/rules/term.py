"""
Represent components that make up a rule. All immutable and hashable.
"""

from enum import Enum
import numpy as np


class TermOperator(Enum):
    """
    Simple operator enum for comparing two real numbers.
    """
    GreaterThan = '>'
    LessThanEq = '<='

    def __str__(self):
        return self.value

    def negate(self):
        # Negate term
        if self is self.GreaterThan:
            return self.LessThanEq
        if self is self.LessThanEq:
            return self.GreaterThan

    def eval(self):
        # Return evaluation operation for term operator
        if self is self.GreaterThan:
            return lambda x, y: x > y
        if self is self.LessThanEq:
            return lambda x, y: np.logical_or(np.isclose(x, y), (x < y))

    def most_general_value(self, values):
        # Given a list of values, return the most general depending on the
        # operator
        if self is self.GreaterThan:
            return max(values)
        if self is self.LessThanEq:
            return min(values)


class Term(object):
    """
    Represent a condition indicating if activation value of variable is
    above/below a threshold.

    Immutable and Hashable.
    """
    def __init__(self, variable, operator, threshold):
        self.variable = variable
        self.threshold = threshold
        self.operator = TermOperator(operator)

    def __str__(self):
        return f'({self.variable} {self.operator} {round(self.threshold, 4)})'

    def to_cat_str(self, dataset=None, var_transform=lambda x: x, limit=10):
        """
        Pretty-prints this term into a string by taking into account the
        possibility than some of the given features may be categorical in
        nature.

        :param DatasetDescriptor dataset: A dataset descriptor object
            containing possibly useful information for different features that
            may be used in this term.
        :param Function[(str), str] var_transform: A function mapping the
            variable name of this term to something else for pretty printing
            purposes.
        :param int limit: The limit of elements we will print as list
            membership if this term partitions on a categorical feature.
        """
        if dataset is None:
            return (
                f'({var_transform(self.variable)} '
                f'{self.operator} {round(self.threshold, 4)})'
            )

        units = dataset.get_units(self.variable) or ""
        if units:
            units = " " + units
        if not dataset.is_discrete(self.variable):
            return (
                f'({var_transform(self.variable)} '
                f'{self.operator} {round(self.threshold, 4)}{units})'
            )

        allowed_vals = dataset.get_allowed_values(self.variable)
        # Else time to see which of these variables is satisfied
        satisfied_vals = []
        for val in allowed_vals:
            val = dataset.transform_to_numeric(self.variable, val)
            if self.apply(val):
                satisfied_vals.append(
                    dataset.transform_from_numeric(self.variable, val)
                )

        if (len(allowed_vals) == 0) or (len(allowed_vals) > limit) or (
            len(satisfied_vals) == 0
        ):
            return (
                f'({var_transform(self.variable)} '
                f'{self.operator} {round(self.threshold, 4)}{units})'
            )

        # Else let's make it clear which values are the ones that satisfy
        # the clause
        if len(satisfied_vals) > 1:
            return (
                f'({var_transform(self.variable)} '
                f'in {{{", ".join(map(str, satisfied_vals))}}}{units})'
            )
        return (
                f'({var_transform(self.variable)} '
                f'= {str(satisfied_vals[0])}{units})'
            )

    def __eq__(self, other):
        return (
            isinstance(other, Term) and
            (self.variable == other.variable) and
            (self.operator == other.operator) and
            (np.isclose(self.threshold, other.threshold))
        )

    def __hash__(self):
        return hash((self.variable, self.operator, self.threshold))

    def negate(self):
        """
        Return term with opposite sign
        """
        return Term(
            self.variable,
            str(self.operator.negate()),
            self.threshold
        )

    def apply(self, value):
        """
        Apply condition to a value
        """
        return self.operator.eval()(value, self.threshold)

    def to_json(self):
        result = {}
        result["variable"] = self.variable
        result["threshold"] = self.threshold
        result["operator"] = str(self.operator)
        return result
