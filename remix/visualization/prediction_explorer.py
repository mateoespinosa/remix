"""
File containing widgets for making a new prediction with a given ruleset.
"""

import numpy as np
import pandas as pd

from remix.rules.rule import RulePredictMechanism
from remix.rules.ruleset import Ruleset
from flexx import flx, ui
from io import StringIO
from pscript.stubs import window, Infinity, Math

from .gui_window import RemixWindow
from .rule_explorer import HierarchicalTreeViz, ruleset_hierarchy_tree
from .rule_list import ClassRuleList
from .rule_statistics import _CLASS_PALETTE
from .uploader import FileUploader


###############################################################################
## Helper Methods
###############################################################################

def _get_activated_ruleset(ruleset, activations, mode="weighted majority"):
    """
    Given a ruleset and a set of input feature activations, this method produces
    a new rule set that contains only the rules that are activated by the given
    sample. It does this while also providing a new prediction together with
    its score.

    :param Ruleset ruleset: The ruleset we will use to make the prediction.
    :param Dict[str, float] activations: A dictionary mapping feature names
        to their values for the sample of interest.
    :param str mode: One of ["weighted majority", "majority class",
        "highest confidence", "highest score"] indicating how to make the
        prediction and compute its score.

    :return Tuple[str, Ruleset, float]: A tuple containing the prediction, a
        Ruleset describing only the rules that were activated by the given
        activations, and the score of this prediction as computed using the
        given mode.
    """
    vector = np.zeros([len(ruleset.feature_names)])
    for i, feature_name in enumerate(ruleset.feature_names):
        vector[i] = activations.get(feature_name, 0)
    if mode == "weighted majority":
        aggregator = RulePredictMechanism.Aggregate
        use_confidence = False
    elif mode == "majority class":
        aggregator = RulePredictMechanism.Count
        use_confidence = False
    elif mode == "highest confidence":
        aggregator = RulePredictMechanism.Max
        use_confidence = True
    elif mode == "highest score":
        aggregator = RulePredictMechanism.Max
        use_confidence = False
    [prediction], [activated_rules], [score] = ruleset.predict_and_explain(
        X=vector,
        use_label_names=True,
        use_confidence=use_confidence,
        aggregator=aggregator,
    )

    explanation = Ruleset(
        rules=activated_rules,
        feature_names=ruleset.feature_names,
        output_class_names=ruleset.output_class_names(),
    )
    return prediction, explanation, score


def _filter_ruleset(ruleset, cls_name):
    rules = []
    for rule in ruleset.rules:
        if rule.conclusion == cls_name:
            rules.append(rule)
    return Ruleset(
        rules=rules,
        feature_names=ruleset.feature_names,
        output_class_names=ruleset.output_class_names(),
    )


def _get_prediction_confidence(ruleset, cls_name):
    result = 0
    tot_sum = 0
    for rule in ruleset.rules:
        if rule.conclusion == cls_name:
            for clause in rule.premise:
                tot_sum += clause.score
                result += clause.score * clause.confidence
    return result / tot_sum if tot_sum else 0

###############################################################################
## Graphical Path Visualization Component
###############################################################################


class PredictionPathComponent(flx.PyWidget):
    """
    Simple widget container for the visualization tree explanation together
    with some simple buttons to allow for easier panning of the tree.
    """

    # Property: the name of the current predicted class from our prediction
    #           result panel.
    predicted_val = flx.StringProp("", settable=True)

    def init(self, ruleset):
        self.ruleset = ruleset

        # Make sure we filter the ruleset to only include rules that are from
        # the predicted class
        used_ruleset = _filter_ruleset(
            ruleset=self.ruleset,
            cls_name=self.predicted_val,
        )
        with ui.VSplit(flex=1, css_class='prediction-path-container'):
            # Add the tree visualization
            self.tree_view = HierarchicalTreeViz(
                data=ruleset_hierarchy_tree(
                    ruleset=used_ruleset,
                    dataset=self.root.state.dataset,
                    merge=self.root.state.merge_branches,
                ),
                fixed_node_radius=5,
                class_names=self.ruleset.output_class_names(),
                flex=0.75,
                branch_separator=1,
            )

            # And also include a control panel with options to collapse, expand,
            # and include more branches in the visualized tree.
            with ui.HBox(
                css_class="prediction-result-tool-bar",
                flex=0.07,
            ):
                ui.Widget(flex=1)  # Filler
                self.collapse_button = flx.Button(
                    text="Collapse Tree",
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=0.25)
                self.expand_button = flx.Button(
                    text="Expand Tree",
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=0.25)
                self.fit_button = flx.Button(
                    text="Fit to Screen",
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=0.25)
                self.only_positive = flx.CheckBox(
                    text="Only Predicted Class",
                    css_class='tool-bar-checkbox',
                    checked=True,
                )
                ui.Widget(flex=1)  # Filler
            ui.Widget(flex=0.15)  # Filler

    @flx.action
    def set_ruleset(self, ruleset):
        """
        Action to change this widget's visualized ruleset.
        """
        self.ruleset = ruleset
        self._update()

    @flx.action
    def zoom_fit(self):
        """
        Action to move the view so that the entire tree fits in its provided
        space.
        """
        self.tree_view.zoom_fit()

    def _update(self, checked=None):
        """
        Updates the visualization with the current state of the ruleset it
        represents.

        :param bool checked:  Whether or not we want to show activated rules
            that correspond only to the predicted class. If None, then we will
            default to the state of the `only_positive` checkbox in this widget.
        """
        if checked is None:
            checked = self.only_positive.checked
        if checked:
            ruleset = _filter_ruleset(
                ruleset=self.ruleset,
                cls_name=self.predicted_val,
            )
        else:
            ruleset = self.ruleset
        self.tree_view.set_data(
            ruleset_hierarchy_tree(
                ruleset=ruleset,
                dataset=self.root.state.dataset,
                merge=self.root.state.merge_branches,
            )
        )

    @flx.reaction('expand_button.pointer_click')
    def _expand_tree(self, *events):
        """
        Reaction to the expand button being clicked.
        """
        self.tree_view.expand_tree()
        self.tree_view.zoom_fit()

    @flx.reaction('fit_button.pointer_click')
    def _fit_tree(self, *events):
        """
        Reaction to the fit button being clicked.
        """
        self.tree_view.zoom_fit()

    @flx.reaction('collapse_button.pointer_click')
    def _collapse_tree(self, *events):
        """
        Reaction to the collapse button being clicked.
        """
        self.tree_view.collapse_tree()
        self.tree_view.zoom_fit()

    @flx.reaction('only_positive.user_checked')
    def _check_positive(self, *events):
        """
        Reaction to the state of the only_positive checkbox changing.
        """
        self._update(checked=events[-1]["new_value"])


###############################################################################
## Rule List Visualization Component
###############################################################################

class NumberEdit(flx.Widget):
    """
    Pretty much the same as Flexx's default LineEdit but it makes sure that
    the provided text is a number.

    This is then inspired by the code in https://github.com/flexxui/flexx/blob/master/flexx/ui/widgets/_lineedit.py
    """

    DEFAULT_MIN_SIZE = 100, 28

    CSS = """
    .flx-LineEdit {
        color: #333;
        padding: 0.2em 0.4em;
        border-radius: 3px;
        border: 1px solid #aaa;
        margin: 2px;
    }
    .flx-LineEdit:focus  {
        outline: none;
        box-shadow: 0px 0px 3px 1px rgba(0, 100, 200, 0.7);
    }

    input::-webkit-outer-spin-button,
    input::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    """

    ## Properties

    num = flx.FloatProp(
        settable=True,
        doc="""
        The current num of the line edit. Settable. If this is an empty
        string, the placeholder_num is displayed instead.
        """
    )

    password_mode = flx.BoolProp(
        False,
        settable=True,
        doc="""
        Whether the insered num should be hidden.
        """
    )

    placeholder_num = flx.StringProp(
        settable=True,
        doc="""
        The placeholder num (shown when the num is an empty string).
        """
    )

    autocomp = flx.TupleProp(
        settable=True,
        doc="""
        A tuple/list of strings for autocompletion. Might not work in all browsers.
        """
    )

    disabled = flx.BoolProp(
        False,
        settable=True,
        doc="""
        Whether the line edit is disabled.
        """
    )

    ## Methods, actions, emitters

    def _create_dom(self):
        global window

        # Create node element
        node = window.document.createElement('input')
        node.setAttribute('type', 'number')
        node.type = 'number'
        node.setAttribute('list', self.id)

        self._autocomp = window.document.createElement('datalist')
        self._autocomp.id = self.id
        node.appendChild(self._autocomp)

        f1 = lambda: self.user_num(self.node.value)
        self._addEventListener(node, 'input', f1, False)
        self._addEventListener(node, 'blur', self.user_done, False)
        #if IE10:
        #    self._addEventListener(self.node, 'change', f1, False)
        return node

    @flx.emitter
    def user_num(self, num):
        """ Event emitted when the user edits the num. Has ``old_value``
        and ``new_value`` attributes.
        """
        d = {'old_value': self.num, 'new_value': num}
        self.set_num(num)
        return d

    @flx.emitter
    def user_done(self):
        """ Event emitted when the user is done editing the num, either by
        moving the focus elsewhere, or by hitting enter.
        Has ``old_value`` and ``new_value`` attributes (which are the same).
        """
        d = {'old_value': self.num, 'new_value': self.num}
        return d

    @flx.emitter
    def submit(self):
        """ Event emitted when the user strikes the enter or return key
        (but not when losing focus). Has ``old_value`` and ``new_value``
        attributes (which are the same).
        """
        self.user_done()
        d = {'old_value': self.num, 'new_value': self.num}
        return d

    @flx.emitter
    def key_down(self, e):
        # Prevent propagating the key
        ev = super().key_down(e)
        pkeys = 'Escape',  # keys to propagate
        if (ev.modifiers and ev.modifiers != ('Shift', )) or ev.key in pkeys:
            pass
        else:
            e.stopPropagation()
        if ev.key in ('Enter', 'Return'):
            self.submit()
            # Nice to blur on mobile, since it hides keyboard, but less nice on desktop
            # self.node.blur()
        elif ev.key == 'Escape':
            self.node.blur()
        return ev

    ## Reactions

    @flx.reaction
    def __num_changed(self):
        self.node.value = self.num

    @flx.reaction
    def __password_mode_changed(self):
        self.node.type = ['number', 'password'][int(bool(self.password_mode))]

    @flx.reaction
    def __placeholder_num_changed(self):
        self.node.placeholder = self.placeholder_num

    # note: this works in the browser but not in e.g. firefox-app
    @flx.reaction
    def __autocomp_changed(self):
        global window
        autocomp = self.autocomp
        # Clear
        for op in self._autocomp:
            self._autocomp.removeChild(op)
        # Add new options
        for option in autocomp:
            op = window.document.createElement('option')
            op.value = option
            self._autocomp.appendChild(op)

    @flx.reaction
    def __disabled_changed(self):
        if self.disabled:
            self.node.setAttribute("disabled", "disabled")
        else:
            self.node.removeAttribute("disabled")


class FeatureSelectorBox(flx.Widget):
    """
    Simple widget to aid with the selection of a single feature in our feature
    selection pane.
    """
    # Describe a minimum size so that it fits nicely.
    DEFAULT_MIN_SIZE = 370, 60

    # Property: the name of the feature
    name = flx.StringProp("", settable=True)

    # Property: an optional unit to use for this feature
    units = flx.StringProp("", settable=True)

    # Property: optional tuple (min, max) indicating the range that this
    #           feature can take.
    limits = flx.TupleProp(settable=True)

    # Property: optional list of discrete values this feature can take if it is
    #           a discrete feature. Otherwise this can be None/[].
    discrete_vals = flx.ListProp([], settable=True)

    # Property: the current value for this feature. =
    value = flx.AnyProp(settable=True)

    # Property: the slider's label for this feature if we will set it using a
    #           slider.
    slider_label = flx.ComponentProp(settable=True)

    def init(self):
        with ui.HFix(css_class='feature-selector-container'):
            # Start with the label for its name
            self.label = flx.Label(
                html=(
                    f"<p>{self.name} ({self.units})</p>" if self.units
                    else f"<p>{self.name}</p>"
                ),
                css_class='feature-selector-label',
                flex=1.5,
            )
            with ui.VBox(flex=1):
                ui.Widget(flex=1)  # filler
                if self.discrete_vals:
                    # Then we will use a combo box here to allow selection
                    index = 0
                    while (
                        (self.discrete_vals[index] != self.value) and
                        (index < len(self.discrete_vals))
                    ):
                        index += 1
                    self.feature_selector = ui.ComboBox(
                        options=self.discrete_vals,
                        selected_index=index,
                        css_class='feature-selector-box',
                    )
                else:
                    # Then its continuous so we may use a slider or an actual
                    # input box as last resource
                    (self.low_limit, self.high_limit) = self.limits
                    if (self.low_limit not in [Infinity, -Infinity]) and (
                        (self.high_limit not in [Infinity, -Infinity])
                    ):
                        # Then we can use a slider in here!
                        with ui.VBox(css_class='slider-group', flex=1):
                            self._mutate_slider_label(NumberEdit(
                                num=Math.round(
                                    min(
                                        max(self.value, self.low_limit),
                                        self.high_limit
                                    ) * 10000
                                ) / 10000,
                                css_class='slider-value-label',
                                placeholder_num=str(self.value),
                                flex=1,
                            ))
                            self.feature_selector = flx.Slider(
                                min=self.low_limit,
                                max=self.high_limit,
                                value=self.value,
                                css_class="feature-selector-slider",
                                flex=1,
                            )
                    else:
                        # Else we will simply use a numeric input box
                        self.feature_selector = NumberEdit(
                            num=self.value,
                            placeholder_num=str(self.value),
                            css_class="feature-selector-edit",
                        )
                ui.Widget(flex=1)  # filler

    @flx.reaction('!feature_selector.user_value')
    def _update_slider_value(self, *events):
        """
        Reaction a change in the feature selector when this is a combo box.
        """
        new_value = events[-1]['new_value']
        new_value = Math.round(
            min(max(new_value, self.low_limit), self.high_limit) * 10000
        ) / 10000
        self.slider_label.set_num(new_value)
        self.feature_selector.set_value(new_value)
        self.set_value(new_value)

    @flx.reaction('!feature_selector.user_done', '!feature_selector.submit')
    def _update_edit_value_boc(self, *events):
        """
        Reaction a change in the feature selector when this is a text box.
        """
        self.set_value(events[-1]['new_value'])

    @flx.reaction('!slider_label.user_done', '!slider_label.submit')
    def _update_edit_value_slider_box(self, *events):
        """
        Reaction a change in the feature selector when this is a slider.
        """
        new_value = events[-1]['new_value']
        self.feature_selector.set_value(
            min(max(new_value, self.low_limit), self.high_limit)
        )
        self.set_value(new_value)

    @flx.reaction('!feature_selector.user_selected')
    def _update_combo_value(self, *events):
        """
        Reaction a change in the feature selector when this is a text box.
        """
        self.set_value(events[-1]['text'])

    @flx.action
    def set_feat_val(self, value):
        """
        Action to set this feature's value to the new given value.
        """

        if self.discrete_vals:
            self.feature_selector.set_selected_index(int(value))
        else:
            # Then its continuous so we may use a slider or an actual
            # input box as last resource
            if (self.low_limit not in [Infinity, -Infinity]) and (
                (self.high_limit not in [Infinity, -Infinity])
            ):
                value = Math.round(
                    min(
                        max(value, self.low_limit),
                        self.high_limit
                    ) * 10000
                ) / 10000
                self.slider_label.set_num(value)
                self.feature_selector.set_value(value)
            else:
                # Else we will simply use a numeric input box
                self.feature_selector.set_num(value)
        self._mutate_value(value)


class FeatureSelectorComponent(flx.PyWidget):
    """
    Widget with a simple scrollable pane to select features that can be
    categorical or continuous in nature.
    """

    # Property: list of all features we will allow the user to set.
    features = flx.ListProp([], settable=True)
    # Property: list of all feature widgets in this pane
    feature_boxes = flx.ListProp([], settable=True)

    def init(self):
        dataset = self.root.state.dataset
        self.feature_to_box_map = {}
        for feature in self.features:
            # Iterate over all features and use the provided dataset object, if
            # any, to determine what is the best way to allow the user to
            # provide its value

            # If feature is discrete, then make sure to provide this to
            # the handler
            discrete_vals = (
                dataset.get_allowed_values(feature) if dataset is not None
                else None
            )
            for i, val in enumerate(discrete_vals or []):
                # Don't forget to make it numerical values only
                discrete_vals[i] = \
                    self.root.state.dataset.transform_from_numeric(
                        feature,
                        val,
                    )

            # Now this is the actual widget that will allow the visualization
            # of this feature
            feat_box = FeatureSelectorBox(
                name=feature,
                limits=self.root.state.get_feature_range(
                    feature,
                    empirical=False,
                ),
                discrete_vals=(discrete_vals or []),
                value=(
                    dataset.get_default_value(feature) if dataset is not None
                    else 0
                ),
                units=(
                    (dataset.get_units(feature) or "") if dataset is not None
                    else ""
                ),
            )
            self.feature_to_box_map[feature] = feat_box
            self._mutate_feature_boxes(
                [feat_box],
                'insert',
                len(self.feature_boxes),
            )

    def is_feature(self, feat_name):
        """
        Checks if given feat_name is a valid feature in this feature selection
        widget.

        :param str feat_name: name of feature we want to check.

        :return bool: True if feat_name is an existing feature in this selector
            and False otherwise.
        """
        return feat_name in self.feature_to_box_map

    @flx.action
    def set_feature_val(self, feat_name, feat_val):
        """
        Action called to set the value of feature with name `feat_name` to be
        `feat_val`.

        If given feature is not handled by this selector, then this operation
        does nothing.

        :param str feat_name:  The name of the feature we are changing.
        :param float feat_val:   The new value for this feature.
        """
        if feat_name in self.feature_to_box_map:
            feat_box = self.feature_to_box_map[feat_name]
            dataset = self.root.state.dataset
            if dataset is not None:
                feat_val = dataset.transform_from_numeric(feat_name, feat_val)
                if dataset.is_discrete(feat_name) and (
                    not dataset.is_categorical(feat_name)
                ):
                    # Then we need to move this guy one tick to the left
                    # due to zero indexing in combo boxes
                    feat_val -= 1
            feat_box.set_feat_val(feat_val)

    def get_values(self):
        """
        Returns the values of all features in this selector in the same order
        as the features in self.features.

        :return List[float]: values of all all features in selector.
        """
        return [
            x.value for x in self.feature_boxes
        ]


################################################################################
## Main REMIX Widget
################################################################################

class PredictComponent(RemixWindow):
    """
    REMIX widget window for performing new predictions and obtaining visual
    explanations from them.
    It contains four main panels:
        1. A results panel where the result of the current prediction is
           displayed at all times.
        2. A feature selection panel in which different features can be set and
           uploaded from a file to make a new prediction based on those
           features.
        3. A tree visualization panel in which an explanation for the prediction
           is given in a tree form representing all the rules that were
           satisfied.
        4. A tree list panel where the explanation is instead provided in form
           of a list of activated rules rather than a visual tree.
    """

    def init(self, ruleset, show_score=False):

        # Initialize some trivial state
        self.show_score = show_score
        self.ruleset = ruleset
        self.true_label = None
        self.mode = "weighted majority"

        # Let's get all the features that are used in the ruleset to simplify
        # the features we allow users to change.
        self.all_features = set()
        for rule in self.ruleset.rules:
            for clause in rule.premise:
                for term in clause.terms:
                    self.all_features.add(term.variable)

        self.all_features = list(self.all_features)
        # Make sure we display most used rules first
        self.all_features = sorted(self.all_features)
        self.class_names = sorted(self.ruleset.output_class_map.keys())

        # Figure out the initial prediction for the default values
        init_vals = self._get_feature_map([
            self.root.state.dataset.get_default_value(feature)
            if self.root.state.dataset is not None else 0
            for feature in self.all_features
        ])
        self.predicted_val, activated_ruleset, score = _get_activated_ruleset(
            ruleset=self.ruleset,
            activations=init_vals,
            mode=self.mode,
        )
        self.confidence_level = _get_prediction_confidence(
            ruleset=activated_ruleset,
            cls_name=self.predicted_val,
        )
        self.score_val = score

        with ui.HSplit(title="Prediction Explorer", flex=1):
            with ui.VSplit(flex=1) as self.prediction_pane:
                # First pane will be the prediction pane showing the tree
                # visualizer
                self.graph_path = PredictionPathComponent(
                    activated_ruleset,
                    predicted_val=self.predicted_val,
                    flex=0.70
                )
                with ui.GroupWidget(
                    title="Triggered Rules",
                    css_class='prediction-pane-group big-group',
                    flex=0.30,
                    style='overflow-y: scroll;',
                ):
                    # Second pane will be the rule list explanation for the
                    # predictions
                    self.rule_list = ClassRuleList(
                        activated_ruleset,
                        editable=False,
                    )

            with ui.VSplit(
                css_class='feature-selector',
                flex=(0.35, 1),
                style='overflow-y: scroll;',
            ):
                with ui.GroupWidget(
                    title="Predicted Result",
                    css_class='prediction-pane-group big-group',
                    flex=0.15,
                    style=(
                        "overflow-y: scroll;"
                        f"background-color: "
                        f"{self._get_color(self.predicted_val)};"
                        "text-size: 125%;"
                    ),
                ) as self.prediction_container:
                    # Third pane is our prediction result pane
                    self.prediction_label = flx.Label(
                        css_class='prediction-result',
                        html=(
                            f"{self.predicted_val}"
                        ),
                        flex=1,
                    )
                    self.confidence_label = flx.Label(
                        css_class='prediction-result',
                        html=(
                            f"Confidence "
                            f"{round(self.confidence_level * 100, 2)}%"
                        ),
                        style="font-size: 80%;",
                        flex=0.25,
                    )
                    self.given_label_match = flx.Label(
                        css_class='prediction-result',
                        html="",
                        style="font-size: 80%;",
                        flex=0.15,
                    )

                    if self.show_score:
                        self.score_label = flx.Label(
                            css_class='prediction-result',
                            html=(
                                f"Score "
                                f"{round(self.score_val, 2)}"
                            ),
                            style="font-size: 80%;",
                            flex=0.25,
                        )
                with ui.HBox(
                    css_class='feature-selector-control-panel',
                    flex=0.01
                ):
                    # We also allow the user to upload a file to make a
                    # prediction for
                    self.predict_button = flx.Button(
                        text="Predict",
                        css_class='predict-button',
                        flex=1,
                    )
                    self.upload_data = FileUploader(
                        text="Upload Data",
                        css_class='upload-button',
                        flex=1,
                    )
                with ui.HBox(
                    css_class='feature-selector-control-panel',
                    flex=0.01
                ):
                    # Let them pick what kind of voting we also request from
                    # the input rule set
                    flx.Widget(flex=1)  # filler
                    flx.Label(
                        text="Mode",
                        style=(
                            "font-weight: bold;"
                            "font-size: 125%;"
                        ),
                    )
                    flx.Widget(flex=0.03)  # filler
                    self.prediction_mode = flx.ComboBox(
                        options=[
                            "majority class",
                            "highest confidence",
                            "weighted majority",
                            "highest score",
                        ],
                        selected_index=0,
                        css_class='feature-selector-box',
                    )
                    flx.Widget(flex=1)  # filler

                with ui.GroupWidget(
                    title="Features",
                    css_class='scrollable_group big-group',
                    flex=0.75,
                    style="margin-bottom: 10%;",
                ):
                    # Finally, the last pane will be the feature selection
                    # pane
                    self.feature_selection = FeatureSelectorComponent(
                        features=self.all_features,
                        flex=1,
                    )

    def _get_color(self, cls_name):
        """
        Helper function to get the color assigned to a given class for
        consistency purposes.

        :param str cls_name: A valid class name

        :return str: an HTML color name used to represent the given class.
        """
        cls_ind = self.root.state.ruleset.output_class_map[cls_name]
        return _CLASS_PALETTE[cls_ind % len(_CLASS_PALETTE)]

    def _get_feature_map(self, values=None):
        """
        Produces a map between features and their assigned values by the feature
        selection pane.

        :param List[Tuple[str, float]] values: An optional list of
            (feature, value) entries indicating the values that different
            features take. If not given, then we will use the values currently
            in the feature selection pane.

        :returns Dict[str, float]: a dictionary mapping feature names to their
            float values.
        """
        real_vector = {}
        values = values or self.feature_selection.get_values()
        for feature_name, val in zip(self.all_features, values):
            if self.root.state.dataset is not None:
                # Then make sure to turn it into its numeric counterpart as
                # this will be needed for thresholding.
                real_vector[feature_name] = \
                    self.root.state.dataset.transform_to_numeric(
                        feature_name,
                        val,
                    )
            else:
                real_vector[feature_name] = val
        return real_vector

    @flx.emitter
    def perform_prediction(self):
        """
        Emitter produced when a prediction has been performed with the values
        currently in the feature selection panel.
        """
        return {
            "values": self.get_values()
        }

    @flx.reaction('predict_button.pointer_click')
    def _predict_action(self, *events):
        """
        Reaction handler to a click in the predict button.
        """
        self._act_on_prediction(true_label=self.true_label)

    @flx.reaction('upload_data.file_loaded')
    def _open_file_path(self, *events):
        """
        Reaction handler to a click in the upload button and a successful load.
        """

        # Set every feature according to the given file by parsing it as a
        # CSV file
        # TODO (mateoespinosa): Add error handling/correction
        data_str = events[-1]['filedata']
        self.true_label = None
        dataset = self.root.state.dataset
        df = pd.read_csv(StringIO(data_str), sep=",")
        for feat_name in df.columns:
            feat_name = str(feat_name)
            if (dataset) and (feat_name == dataset.target_col):
                self.true_label = df[feat_name][0]
            if self.feature_selection.is_feature(feat_name):
                self.feature_selection.set_feature_val(
                    feat_name,
                    df[feat_name][0],
                )
        # And now perform a prediction
        self._act_on_prediction(true_label=self.true_label)

    @flx.reaction('prediction_mode.user_selected')
    def _selected_mode(self, *events):
        """
        Reaction handler to a change in the prediction mode. Make sure we update
        the current prediction and change the state of the widget so that the
        mode is recorded.
        """
        self.mode = events[-1]["text"]
        self._act_on_prediction(true_label=self.true_label)

    def _act_on_prediction(self, true_label=None):
        """
        Once a prediction has been made, this helper function will update all
        the different panels in this widget to correctly show the prediction
        in a consistent fashion.

        :param str true_label:  If given, then provided sample had a true known
            label so the prediction widget will use it to verify if our
            approximation produced the same result.
        """
        real_vector = self._get_feature_map()
        # Let's get the new prediction together with the score and the rule set
        # containing all the rules that have been activated
        self.predicted_val, activated_ruleset, score = _get_activated_ruleset(
            ruleset=self.ruleset,
            activations=real_vector,
        )

        # Update confidence levels as well as results
        used_val = self.predicted_val if true_label is None else true_label
        self.confidence_level = _get_prediction_confidence(
            ruleset=activated_ruleset,
            cls_name=used_val,
        )
        if true_label is not None:
            # Then filter ruleset to only contain rules that match this label
            activated_ruleset = _filter_ruleset(activated_ruleset, true_label)
        self.score_val = score
        self._update_result(true_label=true_label)
        # Finally, update the tree and list visualizations using the activated
        # rule set
        self.graph_path.set_predicted_val(used_val)
        self.graph_path.set_ruleset(activated_ruleset)
        self.rule_list.set_ruleset(activated_ruleset)

    @flx.action
    def reset(self):
        """
        Resets this widget to use the new rule set shared across the entire
        application.
        """
        self.ruleset = self.root.state.ruleset
        self.true_label = None
        # Don't forget to re-make a prediction here.
        self._act_on_prediction()

    def _update_result(self, true_label=None):
        """
        Update our prediction panel with the current prediction state.

        :param str true_label: If given, then this is the true label of the
            sample we just made a prediction from and we will use it to
            display possible information on whether our rule set predicted the
            expected value of it or not.
        """
        self.confidence_label.set_html(
            f"Confidence {round(self.confidence_level * 100, 2)}%"
        )
        if true_label is not None:
            self.prediction_label.set_html(
                f"True label: {true_label}"
            )
            self.given_label_match.set_html(
                f"Predicted result: {self.predicted_val}"
            )
        else:
            self.prediction_label.set_html(
                f"{self.predicted_val}"
            )
            self.given_label_match.set_html("")

        if self.show_score:
            self.score_label.set_html(
                f"Score {round(self.score_val, 2)}"
            )

        used_val = self.predicted_val if true_label is None else true_label
        self.prediction_container.apply_style(
            f"background-color: {self._get_color(used_val)};"
        )
