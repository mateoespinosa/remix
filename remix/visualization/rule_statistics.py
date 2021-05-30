"""
File containing widgets to extract global statistics from a given
rule set.
"""

import bokeh
import numpy as np

from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from bokeh.palettes import Pastel2
from bokeh.plotting import figure
from bokeh.transform import cumsum
from bokeh.transform import factor_cmap
from collections import defaultdict
from flexx import flx, ui

from .gui_window import RemixWindow

################################################################################
## Global Variables
################################################################################

_PLOT_WIDTH = 700
_PLOT_HEIGHT = 325
_CLASS_PALETTE = Pastel2[8]

################################################################################
## Helper Plot Constructors
################################################################################


def _plot_rule_distribution(ruleset, show_tools=True, add_labels=False):
    """
    Helper function to generate a Bokeh plot for the rule distribution in the
    given rule set.

    :param Ruleset ruleset: The ruleset whose rule distribution we want to plot.
    :param bool show_tools: Whether or not we want the Bokeh diagram to show a
        toolset.
    :param bool add_labels: Whether or not we want to add a label box to each
        wedge of the distribution plot.

    :returns Bokeh.Plot: the resulting Bokeh doughnut plot.
    """
    rules_per_class_map = defaultdict(int)
    for rule in ruleset.rules:
        for clause in rule.premise:
            rules_per_class_map[rule.conclusion] += 1

    output_classes = list(ruleset.output_class_map.keys())
    rules_per_class = [
        rules_per_class_map[cls_name] for cls_name in output_classes
    ]
    total_rules = sum(rules_per_class)
    source = ColumnDataSource(
        data={
            "Output Classes": output_classes,
            "Number of Rules": rules_per_class,
            'Angle': [
                num_rules/total_rules * 2*np.pi
                for num_rules in rules_per_class
            ],
            'Percent': [
                f'{100 * num_rules/total_rules:.2f}%'
                for num_rules in rules_per_class
            ],
        }
    )
    result_plot = figure(
        toolbar_location=None if (not show_tools) else "right",
        plot_width=_PLOT_WIDTH,
        plot_height=_PLOT_HEIGHT,
        background_fill_color="#fafafa",
        title=f"Rule counts per class (total number of rules {total_rules})",
    )
    result_plot.annular_wedge(
        x=0,
        y=1,
        inner_radius=0.2,
        outer_radius=0.4,
        start_angle=cumsum('Angle', include_zero=True),
        end_angle=cumsum('Angle'),
        line_color="white",
        fill_color=factor_cmap(
            'Output Classes',
            palette=_CLASS_PALETTE,
            factors=output_classes,
        ),
        legend_field='Output Classes',
        source=source,
    )
    if add_labels:
        percent_labels = LabelSet(
            x=0,
            y=1,
            text='Percent',
            angle=cumsum('Angle', include_zero=True),
            source=source,
            render_mode='canvas',
        )
        result_plot.add_layout(percent_labels)

    result_plot.axis.axis_label = None
    result_plot.axis.visible = False
    result_plot.grid.grid_line_color = None
    result_plot.legend.location = "center_right"

    result_plot.toolbar.logo = None
    for tool in result_plot.toolbar.tools:
        if isinstance(
            tool,
            (bokeh.models.tools.HelpTool)
        ):
            result_plot.toolbar.tools.remove(tool)
    hover = HoverTool(tooltips=[
        ('Class', '@{Output Classes}'),
        ('Count', '@{Number of Rules}'),
        ('Percent', '@{Percent}'),
    ])
    result_plot.add_tools(hover)
    return result_plot


def _plot_term_distribution(
    ruleset,
    show_tools=True,
    dataset=None,
):
    """
    Helper function to generate the term distribution from a given ruleset.
    Returns a function that, when given the number of requested entries,
    returns a Bokeh plot showing that many entries for the term
    distribution of the given rule set.

    :param Ruleset ruleset: The ruleset whose term distribution we want to plot.
    :param bool show_tools: Whether or not we want the Bokeh diagram to show a
        toolset.
    :param DatasetDescriptor dataset: An optional dataset descriptor used to
        decorate the output plot in case some of the features are categorical in
        nature.
    :returns Tuple[Fun[(int), Bokeh.Plot], int]: a tuple containing the
        generator function and the total number of terms found in the given
        ruleset.
    """

    num_used_rules_per_term_map = defaultdict(lambda: defaultdict(int))
    all_terms = set()
    class_names = sorted(ruleset.output_class_map.keys())
    for rule in ruleset.rules:
        for clause in rule.premise:
            for term in clause.terms:
                all_terms.add(term)
                term_str = (
                    term.to_cat_str(dataset=dataset) if dataset is not None
                    else str(term)
                )
                num_used_rules_per_term_map[term_str][rule.conclusion] += 1

    all_terms = list(all_terms)
    # Make sure we display most used rules first
    all_terms = sorted(
        all_terms,
        key=lambda x: -sum(num_used_rules_per_term_map[
            x.to_cat_str(dataset=dataset) if dataset is not None
            else str(x)
        ].values()),
    )

    def _update_rank(num_entries):
        if num_entries != float("inf"):
            used_terms = all_terms[:num_entries]
        else:
            used_terms = all_terms[:]

        # And we will pick only the requested top entries
        used_terms = list(map(
            lambda x: x.to_cat_str(dataset=dataset) if dataset is not None
            else str(x),
            used_terms
        ))
        data = defaultdict(list)
        data["Terms"] = used_terms
        for term in used_terms:
            class_per_term = num_used_rules_per_term_map[term]
            for cls_name in class_names:
                data[cls_name].append(class_per_term.get(cls_name, 0))

        source = ColumnDataSource(data=data)
        title = f"Top {min(num_entries, len(used_terms))} used terms"
        if len(used_terms) != len(all_terms):
            title += (
                f" (out of {len(all_terms)} unique terms used in all rules)"
            )
        result_plot = figure(
            x_range=used_terms,
            toolbar_location=None if (not show_tools) else "right",
            plot_width=_PLOT_WIDTH,
            plot_height=_PLOT_HEIGHT,
            background_fill_color="#fafafa",
            title=title,
        )
        result_plot.vbar_stack(
            stackers=class_names,
            x='Terms',
            width=0.9,
            source=source,
            line_color='white',
            color=_CLASS_PALETTE[:len(class_names)],
            legend_label=list(map(lambda x: "Class: " + x, class_names)),
        )
        result_plot.yaxis.axis_label = 'Count'

        num_used_rules_per_term = np.zeros([len(used_terms)], dtype=np.int32)
        for data_row, vals in data.items():
            if data_row == "Terms":
                continue
            num_used_rules_per_term += vals

        result_plot.xgrid.grid_line_color = None
        result_plot.y_range.start = 0
        result_plot.y_range.end = int(max(0, 0, *num_used_rules_per_term) * 1.1)
        result_plot.xaxis.major_label_orientation = 1.15
        result_plot.xaxis.axis_label = "Terms"
        result_plot.xaxis.axis_label_text_font_size = (
            f"{10 if num_entries <= 10 else 8}pt"
        )
        result_plot.toolbar.logo = None
        for tool in result_plot.toolbar.tools:
            if isinstance(
                tool,
                (bokeh.models.tools.HelpTool)
            ):
                result_plot.toolbar.tools.remove(tool)
        hover = HoverTool(tooltips=[
            ('Count', '@$name'),
            ('Term', '@{Terms}'),
            ('Class', '$name')
        ])
        result_plot.add_tools(hover)
        result_plot.margin = (0, 100, 0, 100)
        return result_plot

    return _update_rank, len(all_terms)


def _plot_feature_distribution(
    ruleset,
    show_tools=True,
):
    """
    Helper function to generate the feature distribution from a given ruleset as
    how much a given feature is used by terms in the given rule set.
    Returns a function that, when given the number of requested entries,
    returns a Bokeh plot showing that many terms use a given feature of the
    given rule set.

    :param Ruleset ruleset: The ruleset object whose feature term distribution
        we want to plot.
    :param bool show_tools: Whether or not we will show Bokeh tools when
        generating our plots.

    :returns Tuple[Fun[(int), Bokeh.Plot], int]: a tuple containing the
        generator function and the total number of features used by terms the
        given ruleset.
    """
    num_used_rules_per_feat_map = defaultdict(lambda: defaultdict(int))
    all_features = set()
    for rule in ruleset.rules:
        for clause in rule.premise:
            for term in clause.terms:
                all_features.add(term.variable)
                num_used_rules_per_feat_map[term.variable][rule.conclusion] += 1

    all_features = list(all_features)
    # Make sure we display most used rules first
    all_features = sorted(
        all_features,
        key=lambda x: -sum(num_used_rules_per_feat_map[x].values()),
    )
    class_names = sorted(ruleset.output_class_map.keys())

    def _update_rank(num_entries):
        if num_entries != float("inf"):
            used_features = all_features[:num_entries]
        else:
            used_features = all_features

        # And we will pick only the requested top entries
        used_features = list(map(str, used_features))
        data = defaultdict(list)
        data["Feature"] = used_features
        for feature in used_features:
            class_per_feat = num_used_rules_per_feat_map[feature]
            for cls_name in class_names:
                data[cls_name].append(class_per_feat.get(cls_name, 0))

        source = ColumnDataSource(data=data)
        title = f"Top {min(num_entries, len(used_features))} used features"
        if len(used_features) != len(all_features):
            title += (
                f" (out of {len(all_features)}/{len(ruleset.feature_names)} "
                f"features used in all the ruleset)"
            )
        result_plot = figure(
            x_range=used_features,
            toolbar_location=None if (not show_tools) else "right",
            plot_width=_PLOT_WIDTH,
            plot_height=_PLOT_HEIGHT,
            background_fill_color="#fafafa",
            title=title,
        )
        result_plot.vbar_stack(
            stackers=class_names,
            x='Feature',
            width=0.9,
            source=source,
            line_color='white',
            color=_CLASS_PALETTE[:len(class_names)],
            legend_label=list(map(lambda x: "Class: " + x, class_names)),
        )
        result_plot.yaxis.axis_label = 'Count'
        result_plot.xgrid.grid_line_color = None
        result_plot.y_range.start = 0
        num_used_rules_per_feat = np.zeros([len(used_features)], dtype=np.int32)
        for data_row, vals in data.items():
            if data_row == "Feature":
                continue
            num_used_rules_per_feat += vals
        result_plot.y_range.end = int(max(0, 0, *num_used_rules_per_feat) * 1.1)
        result_plot.xaxis.major_label_orientation = 1.15
        result_plot.xaxis.axis_label = "Features"
        result_plot.toolbar.logo = None
        for tool in result_plot.toolbar.tools:
            if isinstance(
                tool,
                (bokeh.models.tools.HelpTool)
            ):
                result_plot.toolbar.tools.remove(tool)
        hover = HoverTool(tooltips=[
            ('Count', '@$name'),
            ('Feature', '@{Feature}'),
            ('Class', '$name')
        ])
        result_plot.add_tools(hover)
        return result_plot
    return _update_rank, len(all_features)


def _plot_rule_length_distribution(
    ruleset,
    show_tools=True,
    num_bins=10,
):
    """
    Helper function that produces a distribution Bokeh plot of the length of
    rules in the given rule set.

    :param Ruleset ruleset: The ruleset whose rule length distribution (in
        number of terms) we want to plot.
    :param bool show_tools: Whether or not we will show Bokeh tools when
        generating our plots.
    :param int num_bins: The number bins to use for the distribution bar plot.

    :returns Bokeh.Plot: the resulting Bokeh bar plot.
    """
    class_rule_lengths = [
        [] for _ in ruleset.output_class_map
    ]
    output_classes = [
        cls_name for cls_name in ruleset.output_class_map.keys()
    ]
    output_classes.sort(key=lambda x: ruleset.output_class_map[x])
    for rule in ruleset.rules:
        for clause in rule.premise:
            class_rule_lengths[
                ruleset.output_class_map[rule.conclusion]
            ].append(len(clause.terms))

    result_plot = figure(
        toolbar_location=None if (not show_tools) else "right",
        plot_width=_PLOT_WIDTH,
        plot_height=_PLOT_HEIGHT,
        background_fill_color="#fafafa",
        title="Rule length distribution (click legend to hide classes)",
    )
    for cls_name, rule_lengths in zip(output_classes, class_rule_lengths):
        bins = min(num_bins, len(rule_lengths))
        if bins:
            hist, edges = np.histogram(
                rule_lengths,
                bins=bins,
            )
            result_plot.quad(
                top=hist,
                bottom=0,
                left=edges[:-1],
                right=edges[1:],
                fill_color=_CLASS_PALETTE[ruleset.output_class_map[cls_name]],
                line_color="black",
                alpha=0.5,
                legend_label=("Class: " + cls_name),
            )
    result_plot.y_range.start = 0
    result_plot.legend.location = "center_right"
    result_plot.legend.background_fill_color = "#fefefe"
    result_plot.xgrid.grid_line_color = None
    result_plot.xaxis.axis_label = 'Rule Length'
    result_plot.toolbar.logo = None
    for tool in result_plot.toolbar.tools:
        if isinstance(
            tool,
            (bokeh.models.tools.HelpTool)
        ):
            result_plot.toolbar.tools.remove(tool)
    result_plot.yaxis.axis_label = 'Count'
    result_plot.legend.click_policy = "hide"
    hover = HoverTool(tooltips=[
        ('Count', '@top'),
        ('Range', '(@left, @right)'),
    ])
    result_plot.add_tools(hover)
    return result_plot


################################################################################
## Main Widget Class
################################################################################


class RuleStatisticsComponent(RemixWindow):
    """
    Main widget for handling the global cohort-level view of a given ruleset.
    This view will simply provide general statistics and patterns that are
    immediately found in the main ruleset.
    """

    # We will use this property to handle groups of statistics
    groups = flx.ListProp(settable=True)
    # We will use this property to handle rows of groups
    rows = flx.ListProp(settable=True)
    # And we will use this property to keep track of all plots in the current
    # widget
    plots = flx.ListProp(settable=True)

    def init(self):
        self.ruleset = self.root.state.ruleset
        self.show_tools = self.root.state.show_tools
        with ui.VSplit(
            title="Cohort Summary",
            style=(
                'overflow-y: scroll;'
                'overflow-x: scroll;'
            )
        ) as self.container:
            self._mutate_rows(
                [
                    ui.HBox(
                        flex=1,
                        style=(
                            'overflow-y: scroll;'
                            'overflow-x: scroll;'
                        ),
                    )
                ],
                'insert',
                len(self.rows)
            )
            self._mutate_rows(
                [
                    ui.HBox(
                        flex=1,
                        style=(
                            'overflow-y: scroll;'
                            'overflow-x: scroll;'
                        ),
                    )
                ],
                'insert',
                len(self.rows)
            )
        self._construct_plots()

    @flx.action
    def add_plot(self, title, plot):
        """
        Adds a Bokeh plot with the given title to our grid of plots.
        """
        with ui.Widget(
            title=title,
            style=(
                'overflow-y: scroll;'
                'overflow-x: scroll;'
            ),
            flex=1,
        ) as new_group:
            self._mutate_plots(
                [ui.BokehWidget.from_plot(plot, flex=1)],
                'insert',
                len(self.plots)
            )
            self._mutate_groups(
                [new_group],
                'insert',
                len(self.groups),
            )

    @flx.action
    def _construct_plots(self):
        """
        Action to be called to construct all plots in our visualization.
        """

        with self.container:
            with self.rows[0]:
                # First row has [class distribution, rule length distribution]
                with ui.VBox():
                    flx.Label(
                        text="Rule Class Distribution",
                        flex=1,
                        style=(
                            'font-size: 150%;'
                            'font-weight: bold;'
                            'text-align: center;'
                        ),
                    )
                    self.add_plot(
                        "Rule Distribution",
                        _plot_rule_distribution(
                            ruleset=self.ruleset,
                            show_tools=self.show_tools,
                        ),
                    )
                with ui.VBox():
                    flx.Label(
                        text="Rule Length Distribution",
                        flex=1,
                        style=(
                            'font-size: 150%;'
                            'font-weight: bold;'
                            'text-align: center;'
                        ),
                    )
                    self.add_plot(
                        "Rule Length Distribution",
                        _plot_rule_length_distribution(
                            ruleset=self.ruleset,
                            show_tools=self.show_tools,
                        ),
                    )
            with self.rows[1]:
                # Second row has [feature distribution, term distribution]
                with ui.VBox():
                    new_activation, num_features = \
                        _plot_feature_distribution(
                            ruleset=self.ruleset,
                            show_tools=self.show_tools,
                        )
                    flx.Label(
                        text="Feature Distribution",
                        flex=1,
                        style=(
                            'font-size: 150%;'
                            'font-weight: bold;'
                            'text-align: center;'
                        ),
                    )
                    with ui.HSplit():
                        flx.Label(
                            text="Number of top features to consider:",
                            flex=0.95,
                            style="text-align: right;"
                        )
                        self.feature_combo = flx.ComboBox(
                            options=list(range(1, num_features + 1)),
                            selected_index=min(num_features - 1, 14),
                            style='width: 100%',
                            flex=0.05,
                        )
                    self.add_plot(
                        "Feature Distribution",
                        new_activation(min(num_features - 1, 14) + 1),
                    )
                    self._feature_redraw = new_activation

                with ui.VBox():
                    new_activation, num_terms = \
                        _plot_term_distribution(
                            ruleset=self.ruleset,
                            show_tools=self.show_tools,
                            dataset=self.root.state.dataset,
                        )
                    flx.Label(
                        text="Term Distribution",
                        flex=1,
                        style=(
                            'font-size: 150%;'
                            'font-weight: bold;'
                            'text-align: center;'
                        ),
                    )
                    with ui.HSplit():
                        flx.Label(
                            text="Number of top terms to consider:",
                            flex=0.95,
                            style="text-align: right;"
                        )
                        self.term_combo = flx.ComboBox(
                            options=list(range(1, num_terms + 1)),
                            selected_index=min(num_terms - 1, 14),
                            style='width: 100%',
                            flex=0.05,
                        )
                    self.add_plot(
                        "Term Distribution",
                        new_activation(min(num_terms - 1, 14) + 1),
                    )
                    self._term_redraw = new_activation

    @flx.action
    def _insert_plot(self, new_plot, ind):
        self._mutate_plots(
            [new_plot],
            'replace',
            ind,
        )

    @flx.reaction('term_combo.user_selected')
    def _update_term_distribution(self, *events):
        group_ind = 3
        for event in events:
            # Detaching old plot from parent!
            with self.groups[group_ind]:
                new_plot = ui.BokehWidget.from_plot(
                    self._term_redraw(event['index'] + 1),
                    flex=1,
                )
                old_plot = self.plots[group_ind]
                self._insert_plot(new_plot, group_ind)
                old_plot.set_parent(None)

    @flx.reaction('feature_combo.user_selected')
    def _update_feature_distribution(self, *events):
        group_ind = 2
        # Detaching old plot from parent!
        for event in events:
            with self.groups[group_ind]:
                new_plot = ui.BokehWidget.from_plot(
                    self._feature_redraw(event['index'] + 1),
                    flex=1,
                )
                old_plot = self.plots[group_ind]
                self._insert_plot(new_plot, group_ind)
                old_plot.set_parent(None)

    @flx.action
    def reset(self):
        """
        Resets the entire visualization to use the possibly updated new rule
        set instead.
        """
        for group in self.groups:
            group.set_parent(None)
        for row in self.rows:
            row.set_parent(None)

        self._mutate_groups([])
        self._mutate_rows([])
        with self:
            with self.container:
                self._mutate_rows(
                    [
                        ui.HBox(
                            flex=1,
                            style=(
                                'overflow-y: scroll;'
                                'overflow-x: scroll;'
                            ),
                        )
                    ],
                    'insert',
                    len(self.rows)
                )
                self._mutate_rows(
                    [
                        ui.HBox(
                            flex=1,
                            style=(
                                'overflow-y: scroll;'
                                'overflow-x: scroll;'
                            ),
                        )
                    ],
                    'insert',
                    len(self.rows)
                )
            self._construct_plots()
