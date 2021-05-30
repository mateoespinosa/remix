"""
Python wrapper implementation around R's C5.0 package.
"""
import math
import numpy as np

from .term import Term
from .helpers import parse_variable_str_to_dict
from .rule import Rule

# Interface to R running embedded in a Python process
from rpy2.robjects.packages import importr
from rpy2 import robjects
from rpy2.robjects import pandas2ri

# activate Pandas conversion between R objects and Python objects
pandas2ri.activate()

# C50 R package is interface to C5.0 classification model
C50 = importr('C50')
C5_0 = robjects.r('C5.0')


def truncate(x, decimals):
    """
    Truncates a given number x to contain at most `decimals`.

    :param float x: Number which we want to truncate.
    :param int decimals: Maximum number of decimals that the result will
        contain.

    :returns float: Truncated result.
    """
    power = 10**decimals
    return math.trunc(power * x) / power


def _parse_C5_rule_str(
    rule_str,
    y,
    rule_conclusion_map,
    prior_rule_confidence,
    threshold_decimals=None,
):
    """
    Helper function for extracting rules from the generated string text of
    R's C5.0 output.

    :param str rule_str: The result of running R's C5.0 algorithm in raw string
        form.
    :param Dict[X, Y] rule_conclusion_map: A map between all possible output
        labels of our rules and their corresponding conclusions.
    :param Dict prior_rule_confidence: The prior confidence levels of each
        term in our given ruleset.
    :param int threshold_decimals: If provided, the number of decimals to
        truncate our thresholds when generating new rules.

    :returns Set[Rule]: A set of rules representing the ones extracted from
        the given output of C5.0.
    """
    rules_set = set()
    rule_str_lines = rule_str.split('\n')
    line_index = 2

    metadata_variables = parse_variable_str_to_dict(
        rule_str_lines[line_index]
    )
    n_rules = metadata_variables['rules']

    for _ in range(n_rules):
        line_index += 1

        rule_data_variables = parse_variable_str_to_dict(
            rule_str_lines[line_index]
        )
        n_rule_terms = rule_data_variables['conds']
        rule_conclusion = rule_conclusion_map[rule_data_variables['class']]

        # C5 rule confidence = (
        #     number of training cases correctly classified + 1
        # )/(total training cases covered  + 2)
        rule_confidence = \
            (rule_data_variables['ok'] + 1) / (rule_data_variables['cover'] + 2)
        # Weight rule confidence by confidence of previous rule
        rule_confidence = rule_confidence * prior_rule_confidence

        rule_terms = set()
        for _ in range(n_rule_terms):
            line_index += 1

            term_variables = parse_variable_str_to_dict(
                rule_str_lines[line_index]
            )

            term_operator = (
                '<=' if term_variables['result'] == '<' else '>'
            )  # In C5, < -> <=, > -> >
            threshold = term_variables['cut']
            if threshold_decimals is not None:
                if term_operator == "<=":
                    # Then we will truncate it in a way that we can be sure
                    # any element that was less to this one before this
                    # operation, is still  kept that way
                    threshold = truncate(
                        round(threshold, threshold_decimals + 1),
                        threshold_decimals
                    )
                else:
                    threshold = truncate(threshold, threshold_decimals)
            rule_terms.add(Term(
                variable=term_variables['att'],
                operator=term_operator,
                threshold=threshold,
            ))

        rules_set.add(Rule.from_term_set(
            premise=rule_terms,
            conclusion=rule_conclusion,
            confidence=rule_confidence,
        ))
    if n_rules == 0:
        # Then we will an empty rule that always output the default class
        y = np.array(y)
        default_class = metadata_variables['default']
        default_val = rule_conclusion_map[default_class]
        default_class_percent = np.sum(y == default_class) / len(y)
        rules_set.add(Rule.from_term_set(
            premise=[],
            conclusion=default_val,
            confidence=prior_rule_confidence * default_class_percent,
        ))

    return rules_set


def C5(
    x,
    y,
    rule_conclusion_map,
    prior_rule_confidence,
    winnow=True,
    min_cases=15,
    threshold_decimals=None,
    fuzzy_threshold=False,
    seed=42,
    sample_fraction=0,
    trials=1,
    case_weights=1,
):
    y = robjects.vectors.FactorVector(
        y.map(str),
        levels=robjects.vectors.FactorVector(
            list(map(str, rule_conclusion_map.keys()))
        ),
    )
    if not isinstance(min_cases, int):
        # Then this can be a fraction of training points to be including
        # at a minimum as part of each leaf
        if not isinstance(min_cases, float):
            raise ValueError(
                f"min_cases need to be an integer greater than or equal to 1 "
                f"or a float in [0, 1]. Instead we got {min_cases}"
            )
        if int(min_cases) == float(min_cases):
            min_cases = int(min_cases)
        else:
            # Then let's take a fraction from the total number of points
            # in our training data
            min_cases = int(np.ceil(len(y) * min_cases))

    C5_model = C50.C5_0(
        x=x,
        y=y,
        rules=True,
        # weights=(case_weights or 1),
        control=C50.C5_0Control(
            winnow=winnow,
            minCases=min_cases,
            seed=seed,
            fuzzyThreshold=fuzzy_threshold,
            sample=sample_fraction,
            earlyStopping=True,  # Make bagging stop if it is
                                 # not helpful
        ),
        trials=trials,

    )
    C5_rules_str = C5_model.rx2('rules')[0]
    C5_rules = _parse_C5_rule_str(
        rule_str=C5_rules_str,
        y=y,
        rule_conclusion_map=rule_conclusion_map,
        prior_rule_confidence=prior_rule_confidence,
        threshold_decimals=threshold_decimals,
    )
    return C5_rules
