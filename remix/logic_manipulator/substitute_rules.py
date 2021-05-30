"""
Methods for making rule substitution.
"""

import itertools
import logging
import numpy as np

from ..rules.clause import ConjunctiveClause
from ..rules.rule import Rule


def substitute(total_rule, intermediate_rules, conf_threshold=0):
    """
    Substitute the intermediate rules from the previous layer into the total
    rule.

    :param Rule total_rule: The receiver rule of the beta reduction we are about
        to make.
    :param Ruleset intermediate_rules: The set of intermediate rules which we
        want to substitute in the given total rule.

    :returns Rule: a new rule equivalent to making the substitution of the given
        intermediate rules into total_rule.
    """
    new_premise_clauses = set()
    # for each clause in the total rule
    logging.debug(
        f"Performing substitution with class rule with "
        f"{len(total_rule.premise)} clauses in it."
    )
    for i, old_premise_clause in enumerate(total_rule.premise):
        # list of sets of conjunctive clauses that are all conjunctive
        conj_new_premise_clauses = []
        for old_premise_term in old_premise_clause.terms:
            clauses_to_append = \
                intermediate_rules.get_rule_premises_by_conclusion(
                    old_premise_term
                )
            if clauses_to_append:
                conj_new_premise_clauses.append([
                    clause for clause in clauses_to_append
                    if clause.confidence >= conf_threshold
                ])

        # When combined into a Cartesian product, get all possible conjunctive
        # clauses for merged rule
        # Itertools implementation does not build up intermediate results in
        # memory
        new_combos = itertools.product(
            *tuple(conj_new_premise_clauses)
        )
        logging.debug(
            f"\tAbout to perform a Cartesian product "
            f"{i + 1}/{len(total_rule.premise)} with sets of size "
            f"{list(map(len, conj_new_premise_clauses))}"
        )

        # given tuples of ConjunctiveClauses that are all now conjunctions,
        # union terms into a single clause
        for premise_clause_tuple in new_combos:
            new_terms = set()
            total_confidence = 0
            for premise_clause in premise_clause_tuple:
                # new_clause = new_clause.union(premise_clause)
                total_confidence += premise_clause.confidence
                new_terms = new_terms.union(premise_clause.terms)
            new_clause = ConjunctiveClause(
                terms=new_terms,
                confidence=(
                    (total_confidence / len(premise_clause_tuple))
                    if premise_clause_tuple else 0
                ),
            )
            new_premise_clauses.add(new_clause)
    return Rule(
        premise=new_premise_clauses,
        conclusion=total_rule.conclusion,
    )


def multilabel_substitute(total_rule, multi_label_rules, term_mapping):
    """
    Performs a multi-label substitution in the given total rule using a ruleset
    whose rules all have multi-label conclusions.

    Each clause in the total rule is replaced with the set of rules whose
    conclusion imply the activation of all terms in the given clause.

    :param Rule total_rule: The total rule where we will perform our
        substitution.
    :param Ruleset multi_label_rules: The set of intermediate rules with
        multi-label conclusions, one for each term in the given total rule.
    :param Dict[Term, Tuple[int, bool]] term_mapping: A dictionary mapping a
        term with its corresponding index in the multi-label conclusion vector
        as well as to what that index should be set to (e.g., True or False) for
        this clause to be matched by that label.

    :returns Rule: The result of the aforementioned substitution.
    """
    new_premise_clauses = []
    for clause in total_rule.premise:
        # Let's see which terms will need to be activated for this
        required_terms = []
        for term in clause.terms:
            term_id, term_activated = term_mapping[term]
            required_terms.append((term_id, term_activated))

        # Now let's construct our new premise by iterating over the multi-label
        # rules and adding up all of them whose conclusion satisfy all the
        # required terms
        for rule in multi_label_rules:
            include_rule = True
            for term_id, term_activated in required_terms:
                if rule.conclusion[term_id] != term_activated:
                    # Then this is not a premise to include in our resulting
                    # clause
                    include_rule = False
                    break
            if not include_rule:
                # Then time to go to the next one
                continue

            # Add this rule's premise into our new total premise
            new_premise_clauses += rule.premise

    # We now have our full substitution so let's pack it up into a rule
    return Rule(
        premise=new_premise_clauses,
        conclusion=total_rule.conclusion,
    )


def conditional_substitute(
    total_rule,
    intermediate_rules,
    independent_intermediate_rules,
    term_mapping,
    extra_feature_names,
):
    """
    Substitutes a set of rulesets mapping activations to truth values of terms
    into a total rule that uses those terms in its clauses.

    Contrary to vanilla substitution as done above, we expect the given ruleset
    to contain rules that are conditional on the truth value of all other terms
    than the target term.


    :param Rule total_rule: The total rule where we will perform our
        substitution.
    :param Ruleset intermediate_rules: And intermediate ruleset mapping input
        features conditioned on the truth value of all other terms to the
        true value of a term that we did not conditioned on.
    :param Ruleset ndependent_intermediate_rules: An intermediate ruleset
        mapping input activations to the true values of different terms,
        independently of each other.
    :param Dict[Term, Tuple[int, bool]] term_mapping: A dictionary mapping a
        term with its corresponding conditional variable name as well as to what
        that variable should be set to (i.e., True or False) for this for this
        variable to satisfy that term.
    :param Set[str] extra_feature_names: A set of strings containing the names
        of all the extra feature variables used for conditioning.

    :returns Rule: The result of the aforementioned substitution.
    """

    new_premise_clauses = set()
    # for each clause in the total rule
    for old_premise_clause in total_rule.premise:
        # list of sets of conjunctive clauses that are all conjunctive
        conj_new_premise_clauses = []
        required_condition = {}
        for old_premise_term in old_premise_clause.terms:
            if not required_condition:
                # Then for the first entry of our clause we will use its
                # independent ruleset
                used_ruleset = independent_intermediate_rules
            else:
                used_ruleset = intermediate_rules
            clauses_to_append = \
                used_ruleset.get_conditional_rule_premises_by_conclusion(
                    old_premise_term,
                    condition=required_condition,
                    condition_variables=extra_feature_names,
                )

            # Now add this term's true value as a condition for the follow up
            # term rule substitution
            (var_name, value) = term_mapping[old_premise_term]
            required_condition[var_name] = value

            if clauses_to_append:
                conj_new_premise_clauses.append(clauses_to_append)

        # When combined into a Cartesian product, get all possible conjunctive
        # clauses for merged rule
        # Itertools implementation does not build up intermediate results in
        # memory
        new_combos = itertools.product(
            *tuple(conj_new_premise_clauses)
        )

        # given tuples of ConjunctiveClauses that are all now conjunctions,
        # union terms into a single clause
        for premise_clause_tuple in new_combos:
            new_terms = set()
            total_confidence = 0
            for premise_clause in premise_clause_tuple:
                # new_clause = new_clause.union(premise_clause)
                total_confidence += premise_clause.confidence
                new_terms = new_terms.union(premise_clause.terms)
            new_clause = ConjunctiveClause(
                terms=new_terms,
                confidence=(total_confidence / len(premise_clause_tuple)),
            )
            new_premise_clauses.add(new_clause)

    return Rule(
        premise=new_premise_clauses,
        conclusion=total_rule.conclusion,
    )


def clausewise_substitute(total_rule, intermediate_rules):
    """
    Substitutes the intermediate rules from the previous layer into the total
    rule in a clause-wise fashion.
    What this means is that each rule in the given intermediate set of rules has
    as a conclusion the truth value of a given clause. We then simply replace
    every clause in the total rule with the set of rules in the intermediate
    rules that imply the activation of that clause.

    :param Rule total_rule: The total rule where we will perform our
        substitution.
    :param Ruleset intemediate_rules: A set of rules whose conclusions are all
        full clauses in the given total rule.

    :returns Rule: The result of the aforementioned substitution.
    """
    new_premise_clauses = set()
    # for each clause in the total rule
    logging.debug(
        f"Performing a clause-wise substitution "
        f"using {len(intermediate_rules)} intermediate rules into a rule "
        f"with {len(total_rule.premise)} clauses in it."
    )
    for old_premise_clause in total_rule.premise:
        # list of sets of conjunctive clauses that are all conjunctive
        conj_new_premise_clauses = \
            intermediate_rules.get_rule_premises_by_conclusion(
                old_premise_clause
            )
        new_premise_clauses.update(conj_new_premise_clauses)

    return Rule(
        premise=new_premise_clauses,
        conclusion=total_rule.conclusion,
    )


