"""
Represents a ruleset made up of rules
"""
from collections import defaultdict
from enum import Enum
from multiprocessing import Pool
import dill
import logging
import numpy as np
import pickle
import random

from .clause import ConjunctiveClause
from .rule import Rule, RulePredictMechanism
from remix.logic_manipulator.merge import merge
from remix.utils.parallelism import serialized_function_execute

################################################################################
## Exposed Classes
################################################################################


class RuleScoreMechanism(Enum):
    """
    This class encapsulates the different rule score mechanisms/algorithms
    we support during rule evaluation.
    """
    # Majority algorithm: every rule has score 1
    Majority = 0
    # Accuracy algorithm: every rule has a score equal to
    # (samples covered with same label as this rule)/(total samples covered)
    Accuracy = 1
    # Augmented Hill Climbing algorithm: same as used in paper.
    HillClimb = 2
    # Confidence algorithm: every rule has a score equal to its confidence
    # level
    Confidence = 3

    @staticmethod
    def from_string(mechanism_name):
        search_name = mechanism_name.lower()
        for enum_entry in RuleScoreMechanism:
            if enum_entry.name.lower() == search_name:
                return enum_entry
        raise ValueError(
            f'We do not support score mode "{mechanism_name}" as a rule '
            f'scoring mechanism. We support the following rule scoring '
            f'mechanisms: {list(map(lambda x: x.name, RuleScoreMechanism))}'
        )


class Ruleset(object):
    """
    Represents a set of disjunctive rules, each one with its own conclusion.
    """

    def __init__(
        self,
        rules=None,
        feature_names=None,
        output_class_names=None,
        default_class=None,
        regression=False,
    ):
        self.rules = rules or set()
        self.feature_names = feature_names
        self.output_class_map = dict(zip(
            output_class_names or [],
            range(len(output_class_names or [])),
        ))
        self.default_class = default_class
        self.regression = regression

    def copy(self):
        # Perform a deep copy using pickle serialization
        pickled = pickle.dumps(self)
        return pickle.loads(pickled)

    def merge(self, other_ruleset):
        self.rules.update(other_ruleset.rules)
        self.rules = merge(self.rules)

    def output_class_names(self):
        return sorted(
            list(self.output_class_map.keys()),
            key=lambda x: self.output_class_map[x],
        )

    def __iter__(self):
        # If we iterate over this guy, it is the same as iterating over
        # its rules
        return self.rules.__iter__()

    def __len__(self):
        # The size of our ruleset will be how many rules we have inside it
        return len(self.rules)

    def _get_named_dictionary(self, instance):
        variable_to_value_map = {}
        for i in range(len(instance)):
            variable_name = f'h_0_{i}'
            if self.feature_names is not None:
                variable_name = self.feature_names[i]
            variable_to_value_map[variable_name] = instance[i]
        return variable_to_value_map

    def predict(
        self,
        X,
        use_label_names=False,
        num_workers=1,
    ):
        """
        Predicts the labels corresponding to unseen data points X given a set of
        rules.

        :param np.array X: 2D matrix of data points of which we want to obtain a
            prediction for.
        :param bool use_label_names: If true, then it will output the string
            representations of the predictions. Otherwise it will output their
            encodings by default.
        :param int num_workers: Number of parallel processes we can span to
            evaluate this prediction more efficiently.

        :returns np.array: 1D vector with as many entries as data points in X
            containing our predicted results.
        """
        X = np.atleast_2d(X)
        if use_label_names:
            y = []
        else:
            y = np.array([])

        if len(X.shape) != 2:
            raise ValueError(
                "Expected provided data to be 2D but got "
                "shape {X.shape} instead."
            )

        def _predict_samples(block_id, instances):
            result = []
            for i, instance in enumerate(instances):
                # Map of variable names to values from input data
                variable_to_value_map = self._get_named_dictionary(instance)

                # Each output class given a score based on how many rules x
                # satisfies
                class_ruleset_scores = {}
                for rule_id, class_rules in enumerate(self.rules):
                    score = class_rules.evaluate_score(
                        variable_to_value_map
                    )
                    class_ruleset_scores[class_rules] = score

                # Output class with max score decides the classification of
                # instance. If a tie happens, then choose randomly
                max_rule = max(
                    class_ruleset_scores,
                    key=class_ruleset_scores.get
                )
                max_score = class_ruleset_scores[max_rule]
                if (max_score == 0) and (self.default_class is not None):
                    # Then time to use the default class as the output given
                    # that no other rule got activated
                    max_class = self.default_class
                else:
                    max_set = [
                        rule for (rule, score) in class_ruleset_scores.items()
                        if score == max_score
                    ]
                    if len(max_set) > 1:
                        # Then select one at random
                        max_rule = random.choice(max_set)
                    max_class = max_rule.conclusion

                # Output class encoding is index out output variable
                if not use_label_names:
                    # Then turn this into its encoding
                    max_class = self.output_class_map.get(max_class, max_class)
                result.append(max_class)
            return result

        def _regress_samples(block_id, instances):
            result = []
            for i, instance in enumerate(instances):
                # Map of variable names to values from input data
                variable_to_value_map = self._get_named_dictionary(instance)

                # Each output class given a score based on how many rules x
                # satisfies
                total_activated = 0
                total_sum = 0
                for rule_id, rule in enumerate(self.rules):
                    num_activated = rule.count_activated_clauses(
                        variable_to_value_map
                    )
                    total_activated += num_activated
                    total_sum += num_activated * rule.conclusion

                if not total_activated:
                    # Then return the default value
                    instance_pred = self.default_class
                else:
                    instance_pred = total_sum / total_activated

                result.append(instance_pred)
            return result

        effective_workers = min(num_workers, len(X))
        logging.info(
            f"Performing prediction over {len(X)} test samples "
            f"using {self.num_clauses()} rules and {effective_workers} "
            f"effective processes."
        )
        if effective_workers > 1:
            # Them time to do this the multi-process way
            with Pool(processes=effective_workers) as pool:
                serialized_terms = []
                block_size = len(X) // effective_workers
                current_ind = 0
                blocks = []
                while current_ind < len(X):
                    next_ind = min(current_ind + block_size, len(X))
                    blocks.append(X[current_ind:  next_ind])
                    current_ind = next_ind

                for block_instances in enumerate(blocks):
                    # Let's serialize our (function, args) tuple
                    serialized_terms.append(dill.dumps((
                        _regress_samples
                        if self.regression else _predict_samples,
                        (*block_instances,)
                    )))

                # And do the multi-process pooling call
                all_lists = pool.map(
                    serialized_function_execute,
                    serialized_terms,
                )
                result = []
                for partial in all_lists:
                    result.extend(partial)
                return np.array(result)
        # Else we will do it in this same process in one jump
        return np.array(_regress_samples(1, X)) if self.regression else (
            np.array(_predict_samples(1, X))
        )

    def predict_and_explain(
        self,
        X,
        use_label_names=False,
        only_positive=False,
        use_confidence=False,
        aggregator=RulePredictMechanism.AggregateAvg,
    ):
        """
        Predicts the labels corresponding to unseen data points X given a set of
        rules. It also provides a list of rules that were activated for the
        given input (sorted in terms of their scores) that explain the
        prediction.

        :param np.array X: 2D matrix of data points of which we want to obtain a
            prediction for.
        :param bool use_label_names: If true, then it will output the string
            representations of the predictions. Otherwise it will output their
            encodings by default.
        :param bool only_positive: if True, then we will only produce as an
            explanations activated rules whose conclusion is the output
            predicted class.
        :param bool use_confidence: if set, then we will use confidence values
            for individual rules to compute a single best predicting rule.
        :param RulePredictMechanism aggregator: mechanism used to aggregate
            scores/confidences of activated rules in this rule set.

        :returns Tuple[np.array, List[List[Rule]], List[float]]: A tuple
            containing (1) a 1D vector with as many entries as data points in X
            with the predicted results, (2) a list with as many entries as
            datapoints in X with each entry being a list of Rules that were
            activated by that datapoint, and (3) a vector of scores that
            resulted in the predictions provided in this tuple.
        """
        X = np.atleast_2d(X)
        if use_label_names:
            y = []
        else:
            y = np.array([])
        if len(X.shape) != 2:
            raise ValueError(
                "Expected provided data to be 2D but got "
                "shape {X.shape} instead."
            )

        explanations = [
            [] for _ in range(X.shape[0])
        ]
        scores = [None for _ in range(X.shape[0])]

        for i, instance in enumerate(X):
            # Map of variable names to values from input data
            variable_to_value_map = self._get_named_dictionary(instance)

            # Each output class given a score based on how many rules x
            # satisfies
            class_ruleset_scores = {}
            for class_rules in self.rules:
                score, activated_rules = class_rules.evaluate_score_and_explain(
                    variable_to_value_map,
                    aggregator=aggregator,
                    use_confidence=use_confidence,
                )
                class_ruleset_scores[class_rules] = (score, activated_rules)

            # Output class with max score decides the classification of
            # instance. If a tie happens, then choose randomly
            max_rule = max(
                class_ruleset_scores,
                key=lambda x: class_ruleset_scores[x][0]
            )
            max_score = class_ruleset_scores[max_rule][0]
            if (max_score == 0) and (self.default_class is not None):
                # Then time to use the default class as the output given
                # that no other rule got activated
                max_class = self.default_class
            else:
                max_set = [
                    rule for (score, rule) in class_ruleset_scores.items()
                    if score == max_score
                ]
                if len(max_set) > 1:
                    # Then select one at random
                    max_rule = random.choice(max_set)
                max_class = max_rule.conclusion

            # Output class encoding is index out output variable
            if not use_label_names:
                # Then turn this into its encoding
                max_class = self.output_class_map.get(max_class, max_class)
                y = np.append(y, max_class)
            else:
                y.append(max_class)
            # And add the explanation as well. We will include all the negative
            # explanations as well as the positive ones
            if (max_score != 0) and (
                only_positive or (
                    aggregator in [
                        RulePredictMechanism.Max,
                        RulePredictMechanism.Min,
                        RulePredictMechanism.Count,
                    ]
                )
            ):
                # Then we only sow the positive classes as those are the only
                # valid explanations for the result when no aggregation happens
                explanations[i] = class_ruleset_scores[max_rule][1]
            else:
                for rule in self.rules:
                    explanations[i].extend(class_ruleset_scores[rule][1])
            scores[i] = max_score
        return y, explanations, scores

    def rank_rules(
        self,
        X,
        y,
        score_mechanism=RuleScoreMechanism.Majority,
        use_label_names=False,
        k=4,
    ):
        """
        Assigns scores to all rules in this dataset given training data (X, y)
        and using the provided score mechanism to score each rule.

        :param np.darray X: 2D np.ndarray containing the training data for our
            scoring. First dimension is the sample dimension and second
            dimension is the feature dimension.
        :param np.darray y: 1D np.ndarray containing the training labels for
            datapoints in X.
            scoring.
        :param RuleScoreMechanism score_mechanism: The mechanism to use for
            our scoring function in the rules.
        :param int k: k parameter used in the augmented HillClimbing rule
            scoring mechanism.
        """

        # We will cache a map between sample IDs and their corresponding
        # maps of variables to values to avoid computing this at all times.
        # We only do this when we use the training data for scoring our rules.
        if score_mechanism in [
            RuleScoreMechanism.Accuracy,
            RuleScoreMechanism.HillClimb,
        ]:
            variable_map_cache = [None] * len(X)
        for class_rule in self.rules:
            # Each run of rule extraction return a DNF rule for each output
            # class
            rule_output = class_rule.conclusion
            if not use_label_names:
                # Then turn this into its encoding
                rule_output = self.output_class_map.get(
                    rule_output,
                    rule_output,
                )

            # Each clause in the DBF rule is considered a rule for this output
            # class
            for i, clause in enumerate(class_rule.premise):
                correct = incorrect = 0

                if score_mechanism == RuleScoreMechanism.Majority:
                    # Then every single rule will have score of 1 as we will
                    # only consider the majority class in there
                    clause.score = 1
                elif score_mechanism == RuleScoreMechanism.Confidence:
                    # We will use the confidence of this clause as its scoring
                    # function as well
                    clause.score = clause.confidence
                else:
                    # Else we will score it based on
                    # Iterate over all items in the training data
                    for sample_id, (sample, label) in enumerate(zip(X, y)):
                        # Map of variable names to values from input data. This
                        # is the form of data a rule expects
                        if variable_map_cache[sample_id] is None:
                            # Then we are populating our cache for the first
                            # time
                            variable_map_cache[sample_id] = \
                                self._get_named_dictionary(sample)
                        variable_to_value_map = variable_map_cache[sample_id]

                        # if rule predicts the correct output class
                        if clause.evaluate(data=variable_to_value_map):
                            if rule_output == label:
                                correct += 1
                            else:
                                incorrect += 1

                    # Compute score that we will use for this clause
                    if correct + incorrect == 0:
                        clause.score = 0
                    else:
                        if score_mechanism == RuleScoreMechanism.Accuracy:
                            # Then we only use the plain accuracy to score
                            # this method
                            clause.score = correct / (correct + incorrect)
                        elif score_mechanism == RuleScoreMechanism.HillClimb:
                            clause.score = (
                                (correct - incorrect) / (correct + incorrect)
                            )
                            clause.score += correct / (incorrect + k)
                            # Then we include the extra correction term from
                            # the length of the ruleset
                            clause.score += correct / len(clause.terms)
                        else:
                            raise ValueError(
                                f"Unsupported scoring mechanism "
                                f"{score_mechanism}"
                            )

    def break_rules(self):
        new_rules = set()
        for rule in self.rules:
            for clause in rule.premise:
                new_rules.add(Rule(
                    premise=set([clause]),
                    conclusion=rule.conclusion,
                ))
        return Ruleset(
            rules=rules,
            feature_names=self.feature_names,
            output_class_names=self.output_class_names(),
            default_class=self.default_class,
        )

    def num_clauses(self):
        """
        Computes the number of unique rules with a single conjunctive clause as
        their premise in this rule set.

        :return int: number of unique rules with single conjunctive clause.
        """
        result = 0
        for rule in self.rules:
            result += len(rule.premise)
        return result

    def all_clauses(self):
        """
        Computes a set with the clauses in the premises of all rules in this
        rule set.

        :return Set[ConjunctiveClause]: set of all premises for all rules.
        """
        result = set()
        for rule in self.rules:
            for clause in rule.premise:
                result.add(clause)
        return result

    def num_terms(self):
        """
        Computes the number of unique terms in this rule set.

        :return int: number of unique terms.
        """

        seen_terms = set()
        for rule in self.rules:
            for clause in rule.premise:
                for term in clause.terms:
                    seen_terms.add(term)
        return len(seen_terms)

    def add_rules(self, rules):
        """
        Adds the given rules into this rule set.

        :param Set[Rule] rules: A set of rules to be added.
        """
        self.rules = self.rules.union(rules)

    def eliminate_rules(
        self,
        percent,
        max_num=float("inf"),
        by_confidence=False,
        per_class=False,
    ):
        """
        Eliminates the lowest scoring `percent` of clauses for each rule in this
        ruleset.

        :param float percent: A value in [0, 1] indicating the percent of
            rules we will drop.
        :param max_num: if given, then this is the maximum number of rules we
            will allow in this rule set. If rule set has more rules, then we
            will drop the lowest ranked ones until we hit this target.
        :param bool by_confidence: If True, then we will drop rules by
            confidence rather than by their given scores.
        :param bool per_class: If per_class is set to True, then rule
            elimination is done on a class-basis. Otherwise, it is done on a
            global basis.
        """
        if percent == 0:
            # Then nothing to see here
            return

        # Otherwise, let's do a quick sanity check
        if (percent > 1) or (percent < 0):
            raise ValueError(
                f'Expected given percentage to be a real number between 0 and '
                f'1 but got {percent} instead.'
            )

        # Time to actually do the elimination
        if per_class:
            self.rules = merge(self.rules)
            for class_rule in self.rules:
                # For each set of rules of a given class, we will sort all all the
                # individual clauses in it by their score and drop the lowest
                # `percent` of them.

                # 1. Sort all rules based on their score
                premise = sorted(
                    list(class_rule.premise),
                    key=lambda x: x.confidence if by_confidence else x.score
                )

                # 2. Eliminate the lowest `percent` percent of the rules
                # PS: make sure we leave at least one rule in here....
                to_eliminate = min(
                    int(round(len(premise) * percent)),
                    len(premise) - 1,
                    max_num
                )
                if to_eliminate:
                    premise = premise[to_eliminate:]

                # 3. And update this class's premise
                class_rule.premise = set(premise)
        else:
            # Otherwise we will consider all rules at the same time
            candidates = []
            for rule in self.rules:
                for clause in rule.premise:
                    candidates.append(clause, rule.conclusion)

            # 1. Sort all rules as a whole based on their score
            candidates = sorted(
                candidates,
                key=lambda x: x[0].confidence if by_confidence else x[0].score
            )

            # 2. Eliminate the lowest `percent` percent of the rules
            # PS: make sure we leave at least one rule in here....
            to_eliminate = min(
                int(round(len(candidates) * percent)),
                len(candidates) - 1,
                max_num
            )
            if to_eliminate:
                candidates = candidates[to_eliminate:]

            # 3. And time to reconstruct our rule set
            self.rules = set()
            for (clause, conclusion) in candidates:
                self.rules.add(Rule(
                    premise=set([clause]),
                    conclusion=conclusion,
                ))
            self.rules = merge(self.rules)

    def get_rule_premises_by_conclusion(self, conclusion):
        """
        Return a set of conjunctive clauses that all imply a given conclusion
        """
        premises = set()
        for rule in self.rules:
            if conclusion == rule.conclusion:
                premises = premises.union(rule.premise)

        return premises

    def get_conditional_rule_premises_by_conclusion(
        self,
        conclusion,
        condition,
        condition_variables,
    ):
        """
        Return a set of conjunctive clauses that all imply a given conclusion
        while conditioned on the truth value of the given condition variable.
        """

        premises = set()
        condition = condition or {}
        for rule in self.rules:
            if conclusion != rule.conclusion:
                # Then we are not even interested in this guy
                continue
            # Else, let's only add the clauses that satisfy the given condition
            # while also deleting those terms with extra features which we are
            # not conditioning on
            new_clauses = []
            for clause in rule.premise:
                new_terms = set()
                to_include = True
                for term in clause.terms:
                    if term.variable in condition_variables:
                        # Then this is one of the variable we can condition on
                        # Let's check whether or not we are conditioning
                        # on it
                        if term.variable in condition:
                            if not term.apply(
                                condition[term.variable]
                            ):
                                # Then our condition is not satisfied in here!
                                to_include = False
                                break

                        # Otherwise we will simply ignore it as we are currently
                        # marginalizing it
                    else:
                        # Else this term goes as is in our clause
                        new_terms.add(term)
                if to_include and new_terms:
                    new_clauses.append(ConjunctiveClause(
                        terms=new_terms,
                        confidence=clause.confidence,
                        score=clause.score,
                    ))

            premises.update(new_clauses)

        return premises

    def get_terms_with_conf_from_rule_premises(self):
        """
        Return all the terms present in the bodies of all the rules in the
        ruleset with their max confidence
        """
        term_confidences = {}

        for rule in self.rules:
            for clause in rule.premise:
                clause_confidence = clause.confidence
                for term in clause.terms:
                    if term in term_confidences:
                        term_confidences[term] = max(
                            term_confidences[term],
                            clause_confidence
                        )
                    else:
                        term_confidences[term] = clause_confidence

        return term_confidences

    def get_terms_from_rule_premises(self):
        """
        Return all the terms present in the bodies of all the rules in the
        ruleset
        """
        terms = set()
        for rule in self.rules:
            for clause in rule.premise:
                terms = terms.union(clause.terms)
        return terms

    def __str__(self):
        """
        Pretty print representation of this rule set.
        """

        ruleset_str = '\n'
        for rule in sorted(self.rules, key=str):
            ruleset_str += str(rule) + '\n'

        return ruleset_str

    def to_file(self, path):
        """
        Serializes this rule set into a file with the given path.
        """

        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return self

    def remove_rule(self, delete_rule):
        """
        Removes the rule given by `delete_rule`. If not in the rule set, then
        this acts as a NoOp.

        :param Rule delete_rule: The rule we want to delete.
        """
        for rule in self.rules:
            if rule.conclusion != delete_rule.conclusion:
                continue
            rule.premise = rule.premise.difference(delete_rule.premise)

    def from_file(self, path):
        """
        Deserializes Ruleset from file which was generated by serializing it
        with `to_file(...)`

        :param str path: A valid path containing a serialized Ruleset object.
        """
        with open(path, 'rb') as f:
            deserialized = pickle.load(f)
        self._load_from_deserialized(deserialized)
        return self

    def _load_from_deserialized(self, deserialized):
        """
        Helper function to load Ruleset from deserialized file.

        :param Any deserialized: Deserialized object from pickle load.
        """
        if isinstance(deserialized, tuple):
            # This is the case when we run experiments as we also save the
            # resources needed to generate this rule set.
            deserialized = deserialized[0]
        self.rules = deserialized.rules
        self.feature_names = deserialized.feature_names
        # For backwards compatibility
        self.default_class = getattr(deserialized, "default_class", None)
        self.output_class_map = deserialized.output_class_map

    def from_binary_blob(self, blob):
        """
        Loads rule set from a binary blob representing it.

        :param Bytes blob: A binary blob representing a Ruleset object.
        """
        deserialized = pickle.loads(blob)
        self._load_from_deserialized(deserialized)
        return self

    def to_json(self, **kwargs):
        """
        Serializes rule set to JSON object.
        """
        result = {}
        result["rules"] = []
        for rule in self.rules:
            result["rules"].append(rule.to_json())
        result["feature_names"] = list(self.feature_names)
        result["output_class_map"] = dict(self.output_class_map)
        result["default_class"] = self.default_class
        return result
