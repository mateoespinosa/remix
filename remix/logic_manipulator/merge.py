"""
Merge multiple rules into Disjunctive Normal Form rules

e.g.
    if x>1 AND y<3 AND z<1 THEN 1
    if x>4 THEN 2
    if y<0.4 THEN 2
        ->
    if (x>1 AND y<3 AND z<1) THEN 1
    if (x>4) OR (y<0.4) THEN 2
"""
from ..rules.rule import Rule


def merge(rules):
    """
    Given a disjunctive set of rules (rules must be made up of only conjunctive
    terms), this method will return an equivalent rule set in DNF.

    :param Set[Rule] rules: A set of rules we are merging.
    :returns Set[Rule]: An equivalent set of rules to the provided one in DNF.
    """

    # Build Dictionary mapping rule conclusions to premises(= a set of
    # ConjunctiveClauses)
    conclusion_map = {}
    for rule in rules:
        if rule.conclusion in conclusion_map:
            # Seen conclusion - add rule premise to set of premises for that
            #                   conclusion
            conclusion_map[rule.conclusion] = \
                conclusion_map[rule.conclusion].union(
                    rule.premise
                )
        else:
            # Unseen conclusion - initialize dictionary entry with a set of 1
            # conjunctive clauses
            conclusion_map[rule.conclusion] = rule.premise

    # Convert this dictionary into a set of rules where each conclusion occurs
    # only once, i.e. all rules are in DNF
    DNF_rules = set()
    for conclusion, premise in conclusion_map.items():
        DNF_rules.add(Rule(
            premise=premise,
            conclusion=conclusion
        ))

    return DNF_rules
