"""
Widgets to visualize rule sets in their list representation with all rules
listed in a scrollable pane. It also allows inserting editing options to each
rule for more user interaction.
"""

from remix.rules.rule import Rule
from remix.rules.ruleset import Ruleset
from flexx import flx, ui

from .gui_window import RemixWindow
from .ruleset_loader import RulesetUploader

################################################################################
## Helper Functions
################################################################################


def _clause_to_str(clause, dataset=None):
    """
    Simple helper function to turn a given clause into an readable HTML string.

    :param ConjunctiveClause clause: The Clause we are interested in
        transforming in to an HTML representation.
    :param DatasetDescriptor dataset: An optional dataset descriptor providing
        annotations for the features used in the given clause.
    :type       dataset:  { type_description }
    """
    result = ""
    terms = []
    for term in clause.terms:
        new_term_str = term.to_cat_str(
            dataset=dataset,
            var_transform=lambda x: (
                f"<span style='font-weight: bold; color: #ff6666;'>"
                f"{x}"
                f"</span>"
            ),
        )
        new_term_str = new_term_str.replace("<=", "&leq;")
        terms.append(new_term_str)
    return " <span style='color: #758a7d;'>AND</span> ".join(terms)

################################################################################
## Flexx Helper Widgets
################################################################################


class RuleView(flx.Widget):
    """
    Lower-level widget for displaying a single rule in a horizontal text
    representation.
    """

    CSS = """
    .flx-RuleView {
        border-style: dashed;
        border-color: black;
        border-width: thin;
        margin: 1px;
    }
    .flx-RuleView:hover {
        background-color: #eefafe;
    }"""

    # Property: the precedent/antecedent of this rule
    precedent = flx.StringProp(settable=True)
    # Property: the conclusion of this rule
    conclusion = flx.StringProp(settable=True)
    # Property: the score of this rule
    score = flx.FloatProp(settable=True)
    # Property: the confidence of this rule
    confidence = flx.FloatProp(settable=True)

    # Property: unique index given to the clause of this rule
    clause_idx = flx.IntProp(settable=True)
    # Property: unique index given to this rule
    rule_idx = flx.IntProp(settable=True)
    # Property: true index in list of all rules of parent rule set.
    true_idx = flx.IntProp(settable=True)
    # Property: whether or not this rule will be editable.
    editable = flx.BoolProp(True, settable=True)

    # Property: button used for deleting this rule.
    delete_button = flx.ComponentProp(settable=True)
    # Property: button used for editing this rule (TODO: implement this)
    edit_button = flx.ComponentProp(settable=True)
    # Property: button used for ranking this rule (TODO: implement this)
    rank_button = flx.ComponentProp(settable=True)

    def init(self):
        with flx.HFix(style="margin: auto;"):
            self.index_label = flx.Label(
                flex=0.03,
                css_class="rule-list-counter",
                html=f'{self.true_idx + 1}.',
            )
            self.rule_text = flx.Label(
                flex=0.9,
                css_class="rule-list-rule-text"
            )
            self.rule_text.set_html(
                "<span class='camviz_tooltip'>"
                "<b style=\"font-family: 'Source Code Pro', monospace;\"> IF</b>"
                f'<span style="font-family:\'Inconsolata\';">'
                f' {self.precedent}</span> '
                f"<b style=\"font-family: 'Source Code Pro', monospace;\">THEN"
                f'</b> <i>{self.conclusion}</i>'
                f'<span class="camviz_tooltiptext">Score: {self.score}</span>'
                "</span>"
            )
            if self.editable:
                # TODO: In future work, we will allow direct editing of a rule
                #       and its conditions.
                # self._mutate_edit_button(flx.Button(
                #     text="edit",
                #     flex=0.07,
                #     css_class="rule-list-delete-button",
                #     style="padding: 0;",
                # ))
                self._mutate_delete_button(flx.Button(
                    text="delete",
                    flex=0.07,
                    css_class="rule-list-delete-button",
                    style="padding: 0;",
                ))

    @flx.emitter
    def rule_removed(self):
        """
        Emitter produced when this rule was removed.
        """

        return {
            'conclusion': self.conclusion,
            'clause_idx': self.clause_idx,
            'rule_idx': self.rule_idx,
            'true_idx': self.true_idx,
            'precedent': self.precedent,
            'score:': self.score,
        }

    @flx.reaction('!delete_button.pointer_click')
    def delete_rule(self, *events):
        """
        Reaction to delete button which will trigger the removal of this rule.
        """

        self.rule_removed()
        self.set_parent(None)

    @flx.reaction('true_idx')
    def _update_labels(self, *events):
        """
        Reaction to changes in the true_idx which will force its index label to
        be corrected accordingly.
        """

        self.index_label.set_html(
            f'{self.true_idx + 1}.'
        )

    @flx.emitter
    def pointer_click(self, e):
        """
        Emitter produced when the user clicks on this rule visualization.
        """
        return e

    @flx.reaction('rule_text.pointer_click')
    def __on_pointer_click(self, *events):
        """
        Reaction to when the text containing the rule itself is clicked,
        """
        self.rule_text.node.blur()
        for event in events:
            self.pointer_click({
                "isTrusted": event.isTrusted,
            })


class ClassRuleList(flx.PyWidget):
    """
    Core simple widget showing a list of rules and allowing for possible editing
    of the same. Each rule will be listed into a scrollable pane.
    """

    # Property: the list of rule visualizations. Needs to be a property so that
    #           we can attach listeners to emissions from individual rules.
    rules = flx.ListProp([], settable=True)

    # Property: whether or not this object is editable and its rules can be
    #           modified.
    editable = flx.BoolProp(True, settable=True)

    def init(self, ruleset):
        self.ruleset = ruleset
        self.rule_objs = sorted(
            list(self.ruleset.rules),
            key=lambda x: x.conclusion,
        )
        self.clause_orderings = []
        with ui.Widget(
            css_class='scrollable_group',
        ) as self.container:
            for rule_idx, rule_obj in enumerate(self.rule_objs):
                new_ordering = []
                for clause_idx, clause in enumerate(sorted(
                    rule_obj.premise,
                    key=lambda x: x.score,
                )):
                    new_ordering.append(clause)
                    self.add_rule(
                        rule_idx,
                        clause_idx,
                        clause,
                        rule_obj.conclusion,
                        clause.score,
                        clause.confidence,
                    )
                self.clause_orderings.append(new_ordering)

    def add_rule(
        self,
        rule_idx,
        clause_idx,
        clause,
        conclusion,
        score,
        confidence,
    ):
        """
        Helper function to add a single given rule widget into this rule list.
        """

        # Generate the widget visualization of this rule
        new_rule = RuleView(
            rule_idx=rule_idx,
            clause_idx=clause_idx,
            true_idx=len(self.rules),
            precedent=_clause_to_str(
                clause=clause,
                dataset=self.root.state.dataset,
            ),
            conclusion=conclusion,
            score=score,
            confidence=confidence,
            editable=self.editable,
        )

        # And insert it
        self._mutate_rules(
            [new_rule],
            'insert',
            len(self.rules),
        )

    @flx.emitter
    def ruleset_update(self, rule_idx):
        """
        Update the rule set as rule with index rule_idx has been
        modified/removed.
        """
        return {
            "rule_idx": rule_idx
        }

    @flx.reaction('rules*.rule_text.pointer_click')
    def _clicked_rule(self, *events):
        """
        Reaction to clicking on a single rule in this list.
        """
        for e in events:
            rule = e["source"]
            self.pointer_click({
                "class_idx": e.class_idx,
                "rule_idx": e.rule_idx,
                "true_idx": e.true_idx,
                "precedent": e.precedent,
                "conclusion": e.conclusion,
                "score": e.score,
                "confidence": e.confidence,
            })

    @flx.emitter
    def pointer_click(self, e):
        """
        Emitter when someone clicks on the list. Used for other applications
        which may want to know this.
        """
        return e

    @flx.reaction('rules*.rule_removed')
    def remove_rule(self, *events):
        """
        Removes rule with index event["rule_idx"] and clause with index
        event["clause_idx"] from this list. This is a reaction to the "delete"
        button in each rule.
        """

        # Time to remove the rule from our ruleset
        for event in events:
            rule_idx = event["rule_idx"]
            clause_idx = event["clause_idx"]
            true_idx = event["true_idx"]
            self.root.state.ruleset.remove_rule(
                Rule(
                    premise=set([self.clause_orderings[rule_idx][clause_idx]]),
                    conclusion=self.rule_objs[rule_idx].conclusion,
                )
            )

            # Remove it from our ordering of the different rules
            self.clause_orderings[rule_idx].pop(clause_idx)

            # And update the IDs of all entries that came after this one
            for i, rule in enumerate(self.rules):
                if i <= true_idx:
                    continue
                # Otherwise, time to update it
                rule.set_true_idx(rule.true_idx - 1)
                if rule.rule_idx == rule_idx:
                    # Then also need to decrease the clause number here
                    rule.set_clause_idx(rule.clause_idx - 1)

            # Remove the rule entry from our list of rules
            self.rules.pop(true_idx)

            # Finally, emit an even that will tell all other windows to update
            # as needed
            self.ruleset_update(rule_idx)

    @flx.action
    def reset(self):
        """
        Resets the whole rule list widget to be updated given the
        new rule set.
        """
        old_rules = self.rules[:]
        self._mutate_rules([])
        self.clause_orderings = []
        with self:
            with self.container:
                for rule_idx, rule_obj in enumerate(self.rule_objs):
                    new_ordering = []
                    for clause_idx, clause in enumerate(sorted(
                        rule_obj.premise,
                        key=lambda x: x.score,
                    )):
                        new_ordering.append(clause)
                        self.add_rule(
                            rule_idx,
                            clause_idx,
                            clause,
                            rule_obj.conclusion,
                            clause.score,
                            clause.confidence,
                        )
                    self.clause_orderings.append(new_ordering)
        for rule in old_rules:
            rule.set_parent(None)

    @flx.action
    def clear(self):
        """
        Clears the entire rule list.
        """

        # Detach every rule from its parent
        for rule in self.rules:
            rule.set_parent(None)

    @flx.reaction('editable')
    def update_list(self, *events):
        """
        Updates this list if the editable property changes.
        """
        self.reset()

    @flx.action
    def set_ruleset(self, ruleset):
        """
        Sets the ruleset to be used in this visualization.
        """
        self.ruleset = ruleset
        self.rule_objs = sorted(
            list(self.ruleset.rules),
            key=lambda x: x.conclusion,
        )
        self.reset()




################################################################################
## Main REMIX Widget
################################################################################


class RuleListComponent(RemixWindow):
    """
    This class describes a simple Flexx widget for listing and visualizing
    the list of rules in the given rule set while also allowing for editing.

    In future iterations, we expect this widget to provide capabilities for
    introducing new rules as well as exporting and merging rule sets.
    """

    def init(self, ruleset):
        self.ruleset = ruleset
        self.current_rule_idx = 0
        self.rules = list(sorted(
            self.ruleset.rules,
            key=lambda x: x.conclusion
        ))
        with ui.HBox(title="Rule Editor") as tab:
            with ui.VBox(flex=0.075) as self.box_pannel:
                ui.Widget(flex=1)  # Filler
                with ui.GroupWidget(
                    title="Class",
                    css_class="file-edit-group",
                    style="padding-bottom: 20%;",
                ):
                    with ui.VBox(flex=1):
                        self.class_selector = ui.ComboBox(
                            options=[
                                rule.conclusion for rule in self.rules
                            ],
                            selected_index=0,
                            css_class='class-selector-box',
                            flex=1,
                        )
                ui.Widget(flex=0.25)  # Filler
                with ui.GroupWidget(
                    title="File",
                    css_class="file-edit-group",
                    flex=0.25,
                    style="padding-bottom: 30%;",
                ):
                    with ui.VBox(flex=1):
                        # TODO: In future support rule set exporting
                        # self.export_button = ui.Button(
                        #     text="Export Ruleset",
                        #     css_class='tool-bar-button',
                        # )

                        # TODO: In future support rule set merging
                        # self.merge_button = RulesetUploader(
                        #     text="Merge Ruleset",
                        #     css_class='tool-bar-button',
                        # )
                        self.reset_button = ui.Button(
                            text="Reset Ruleset",
                            css_class='tool-bar-button',
                        )

                # TODO: In future support for rule addition
                # ui.Widget(flex=0.25)  # Filler
                # with ui.GroupWidget(
                #     title="Ruleset",
                #     css_class="file-edit-group",
                #     flex=0.25,
                # ):
                #     with ui.VBox(flex=1):
                #         self.add_rule_button = ui.Button(
                #             text="Add Rule",
                #             css_class='tool-bar-button',
                #         )
                ui.Widget(flex=1)  # Filler

            first_rule = self.rules[self.current_rule_idx]
            with ui.GroupWidget(
                title=(
                    f'Class: {first_rule.conclusion} '
                    f'({len(first_rule.premise)} '
                    f'rule{"" if len(first_rule.premise) == 1 else "s"})'
                ),
                style='overflow-y: scroll;',
                flex=0.925,
            ) as self.class_group:
                self.class_ruleset = ClassRuleList(
                    Ruleset(
                        rules=(
                            [first_rule]
                            if self.rules else []
                        ),
                        feature_names=self.ruleset.feature_names,
                        output_class_names=(
                            self.ruleset.output_class_names()
                        ),
                    ),
                    flex=1,
                )

    @flx.reaction('class_selector.user_selected')
    def _current_view(self, *events):
        """
        Handler for the class view selector change.
        """

        # Simple change all the rules to the ones corresponding to the
        # selected class
        rule = self.rules[events[-1]['index']]
        self.current_rule_idx = events[-1]['index']
        self.class_ruleset.set_ruleset(
            Ruleset(
                rules=[rule],
                feature_names=self.ruleset.feature_names,
                output_class_names=(
                    self.ruleset.output_class_names()
                ),
            )
        )
        self.class_group.set_title(
            f'Class: {rule.conclusion} '
            f'({len(rule.premise)} '
            f'rule{"" if len(rule.premise) == 1 else "s"})'
        )

    @flx.reaction('reset_button.pointer_click')
    def _reset_ruleset(self, *events):
        """
        Handler for reset button.
        """

        self.root.state.reset_ruleset()
        # And propagate this reset everywhere
        self.ruleset_update({"source_id": self.id})
        self.reset()

    # TODO: In future, add merge button to allow the merging of two or more
    #       rule sets.
    # @flx.reaction('merge_button.ruleset_load_ended')
    # def _merge_ruleset(self, *events):
    #     for event in events:
    #         self.root.state.merge_ruleset(event['ruleset'])
    #     self.ruleset_update({"source_id": None})
    #     self.reset()

    @flx.reaction('class_ruleset.ruleset_update')
    def bypass_update(self, *events):
        """
        Bypasses an update done in one of the rules to the parent so that
        it can be communicated across windows.
        """
        for event in events:
            event = event.copy()
            event["source_id"] = self.id
            rule = self.rules[self.current_rule_idx]
            # And the group title
            self.class_group.set_title(
                f'Class: {rule.conclusion} '
                f'({len(rule.premise)} '
                f'rule{"" if len(rule.premise) == 1 else "s"})'
            )

            self.ruleset_update(event)

    @flx.action
    def reset(self):
        """
        Resets the entire rule list widget using the newly updated rule set
        object.
        """

        # Reset the class ruleset itself
        rule = self.rules[self.current_rule_idx]
        self.class_ruleset.set_ruleset(
            Ruleset(
                rules=[rule],
                feature_names=self.ruleset.feature_names,
                output_class_names=(
                    self.ruleset.output_class_names()
                ),
            )
        )
        # And the group title
        self.class_group.set_title(
            f'Class: {rule.conclusion} '
            f'({len(rule.premise)} '
            f'rule{"" if len(rule.premise) == 1 else "s"})'
        )
