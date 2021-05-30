"""
Represent a rule with a premise in Disjunctive Normal Form (DNF) and conclusion
of another term or class conclusion
"""

from collections import defaultdict

from enum import Enum
from .clause import ConjunctiveClause
from .term import Term
from remix.logic_manipulator.satisfiability import \
    remove_unsatisfiable_clauses


class RulePredictMechanism(Enum):
    """
    This class encapsulates the different rule prediction mechanisms we have
    given a set of scores for each rule,
    """

    # Max Prediction: returns the rule with the maximum score. If multiple,
    #                 then ties are break arbitrarily
    Max = 0
    # Min Prediction: returns the rule with the minimum score. If multiple,
    #                 then ties are break arbitrarily
    Min = 1
    # Aggregate Prediction: returns the rule with the maximum aggregated score
    #                       of all activated rules.
    Aggregate = 2
    # Aggregate Average Prediction: same as above but all scores of activating
    #                               rules are averaged.
    AggregateAvg = 3
    # Count Prediction: simply returns counts of all activated rules (equivalent
    #                   to aggregating rules when they all have scores 1)
    Count = 4


class Rule(object):
    """
    Represents a rule in DNF form i.e.
        (t1 AND t2 AND ..) OR ... OR  ( tk AND tk2 AND ... )  -> <conclusion>

    Immutable and Hashable.
    """

    def __init__(self, premise, conclusion, remove_unsatisfiable=True):
        self.premise = premise
        if remove_unsatisfiable:
            self.premise = remove_unsatisfiable_clauses(clauses=self.premise)
        self.conclusion = conclusion

    def remove_unsatisfiable_clauses(self):
        self.premise = remove_unsatisfiable_clauses(clauses=self.premise)

    def __eq__(self, other):
        return (
            isinstance(other, Rule) and
            (self.premise == other.premise) and
            (self.conclusion == other.conclusion)
        )

    def __hash__(self):
        return hash(self.conclusion)

    def __str__(self):
        premise_str = [
            (str(clause)) for clause in sorted(self.premise, key=str)
        ]
        return f"IF {' OR '.join(premise_str)} THEN {self.conclusion}"

    def evaluate_score(self, data):
        """
        Given a list of input activations and their values, return the combined
        proportion of clauses that satisfy the rule
        """
        total = len(self.premise)
        total_correct_score = 0
        for clause in self.premise:
            if clause.evaluate(data):
                total_correct_score += clause.score

        # Be careful with the always true clause (i.e. empty). In that case, the
        # average score is always 1.
        return total_correct_score/total if total else 0

    def count_activated_clauses(self, data):
        result = 0
        for clause in self.premise:
            if clause.evaluate(data):
                result += 1
        return result

    def evaluate_score_and_explain(
        self,
        data,
        use_confidence=False,
        aggregator=RulePredictMechanism.AggregateAvg,
    ):
        """
        Given a list of input activations and their values, return the combined
        score of clauses that satisfy the rule and a list with individual
        rules that this data point satisfied
        """
        total = len(self.premise)
        total_correct_score = 0
        if aggregator == RulePredictMechanism.Min:
            total_correct_score = float("inf")
        explanation = []
        for clause in self.premise:
            if clause.evaluate(data):
                weight = clause.confidence if use_confidence else clause.score
                if (aggregator == RulePredictMechanism.Max) and (
                    total_correct_score < weight
                ):
                    total_correct_score = weight
                    explanation = [Rule(
                        premise=[clause],
                        conclusion=self.conclusion,
                    )]
                elif (aggregator == RulePredictMechanism.Min) and (
                    total_correct_score > weight
                ):
                    total_correct_score = weight
                    explanation = [Rule(
                        premise=[clause],
                        conclusion=self.conclusion,
                    )]
                elif aggregator in [
                    RulePredictMechanism.Aggregate,
                    RulePredictMechanism.AggregateAvg,
                ]:
                    explanation.append(
                        Rule(premise=[clause], conclusion=self.conclusion)
                    )
                    total_correct_score += clause.score
                elif aggregator == RulePredictMechanism.Count:
                    explanation.append(
                        Rule(premise=[clause], conclusion=self.conclusion)
                    )
                    total_correct_score += clause.score

        explanation.sort(
            key=lambda x: list(x.premise)[0].score,
        )
        if aggregator == RulePredictMechanism.AggregateAvg:
            # Be careful with the always true clause (i.e. empty). In that case,
            # the average score is always 1.
            total_correct_score = total_correct_score/total if total else 0
        return total_correct_score, explanation

    @classmethod
    def from_term_set(cls, premise, conclusion, confidence):
        """
        Construct Rule given a single clause as a set of terms and a conclusion
        """
        rule_premise = {
            ConjunctiveClause(terms=premise, confidence=confidence)
        }
        return cls(premise=rule_premise, conclusion=conclusion)

    @classmethod
    def initial_rule(cls, output_class, threshold):
        """
        Construct Initial Rule given parameters with default confidence value
        of 1
        """
        rule_premise = ConjunctiveClause(
            terms={Term(
                variable=output_class,
                operator='>',
                threshold=threshold,
            )},
            confidence=1
        )
        return cls(premise={rule_premise}, conclusion=output_class)

    def get_terms_with_conf_from_rule_premises(self):
        """
        Return all the terms present in the bodies of all the rules in the
        ruleset with their max confidence
        """
        # Every term will be initialized to have a confidence of 1. We will
        # select the minimum across all clauses that use the same term
        term_confidences = defaultdict(lambda: 1)

        for clause in self.premise:
            for term in clause.terms:
                term_confidences[term] = min(
                    term_confidences[term],
                    clause.confidence
                )

        return term_confidences

    def to_json(self):
        result = {}
        result["premise"] = []
        for clause in sorted(self.premise, key=str):
            result["premise"].append(clause.to_json())
        result["conclusion"] = self.conclusion
        return result
