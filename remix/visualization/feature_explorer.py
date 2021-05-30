"""
This file will contain widgets for analyzing and visualizing how each
feature is thresholded.
"""
import numpy as np

from collections import defaultdict
from remix.rules.term import TermOperator
from flexx import flx, ui
from pscript.stubs import d3, window, Math
from sklearn.neighbors import KernelDensity

from .gui_window import RemixWindow
from .rule_statistics import _CLASS_PALETTE

################################################################################
## Helper Functions
################################################################################


def _js_round(x, num=0):
    mult = 10 ** num
    return Math.round(x * mult) / mult


def _bound_str(d):
    return (
        ("[" if d["left_inclusive"] else "(") +
        str(_js_round(d["bounds"][0], 4)) +
        ", " +
        str(_js_round(d["bounds"][1], 4)) +
        ("]" if d["right_inclusive"] else ")")
    )

################################################################################
## Helper Widgets
################################################################################


class FeatureBoundView(flx.Widget):
    """
    Low-level widget view for showing a feature's bounds on top of its
    empirical distribution in the given dataset.
    """

    CSS = """
        path {
            fill: none;
            stroke: #aaa;
        }
    """
    DEFAULT_MIN_SIZE = 800, 500

    class_name = flx.StringProp("", settable=True)
    feature_name = flx.StringProp(settable=True)
    feature_limits = flx.TupleProp(settable=True)
    rule_bounds = flx.ListProp(settable=True)
    data = flx.ListProp([], settable=True)
    estimated_density = flx.ListProp([], settable=True)
    plot_density = flx.BoolProp(False, settable=True)
    num_bins = flx.IntProp(50, settable=True)
    class_color = flx.StringProp("black", settable=True)

    def init(self):
        self.node.id = self.id
        window.setTimeout(self.load_viz, 500)

    @flx.action
    def load_viz(self):
        self.width, self.height = self.DEFAULT_MIN_SIZE
        self.left_margin, self.right_margin = 50, 50
        self.top_margin, self.bottom_margin = 50, 50
        self.tooltip = d3.select("body").append("div").style(
            "position",
            "absolute",
        ).style(
            "z-index",
            "10",
        ).style(
            "visibility",
            "hidden",
        ).text("")

        x = d3.select(f'#{self.id}')
        self.svg = x.append("svg").attr(
            "width",
            self.width
        ).attr(
            "height",
            self.height,
        ).attr(
            "xmlns",
            "http://www.w3.org/2000/svg",
        ).attr(
            "version",
            "1.1"
        )

        ########################################################################
        ## Attach pattern definitions
        ########################################################################

        self.svg.append(
            "defs",
        ).append(
            "pattern",
        ).attr(
            "id",
            "crosshatch",
        ).attr(
            "patternUnits",
            "userSpaceOnUse",
        ).attr(
            "width",
            8,
        ).attr(
            "height",
            8,
        ).append(
            "img"
        ).attr(
            "xlink:href",
            "data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPSc4JyBoZWlnaHQ9JzgnPgogIDxyZWN0IHdpZHRoPSc4JyBoZWlnaHQ9JzgnIGZpbGw9JyNmZmYnLz4KICA8cGF0aCBkPSdNMCAwTDggOFpNOCAwTDAgOFonIHN0cm9rZS13aWR0aD0nMC41JyBzdHJva2U9JyNhYWEnLz4KPC9zdmc+Cg==",
        ).attr(
            "x",
            0,
        ).attr(
            "y",
            0,
        ).attr(
            "width",
            8,
        ).attr(
            "height",
            8,
        )

        self.svg.append(
            'defs'
        ).append(
            'pattern'
        ).attr(
            'id',
            'diagonalHatch',
        ).attr(
            'patternUnits',
            'userSpaceOnUse',
        ).attr(
            'width',
            4,
        ).attr(
            'height',
            4,
        ).append(
            'path'
        ).attr(
            'd',
            'M-1,1 l2,-2 M0,4 l4,-4 M3,5 l2,-2',
        ).attr(
            'stroke',
            '#000000',
        ).attr(
            'stroke-width',
            1
        )

        ########################################################################
        ## Add x-axis
        ########################################################################

        self.x_scale = d3.scaleLinear().domain(
            [self.feature_limits[0] * 0.9, self.feature_limits[1] * 1.1]
        ).range(
            [self.left_margin, self.width - self.right_margin]
        )
        x_axis_y_position = self.height - self.bottom_margin - 50
        self.x_axis = self.svg.append(
            "g"
        ).attr(
            "class",
            "x-axis",
        ).attr(
            "transform",
            f"translate({0}, {x_axis_y_position})"
        ).call(
            d3.axisBottom(self.x_scale).ticks(
                min(max(2, self.width // 15), 25)
            )
        )

        self.x_axis.selectAll("path").attr(
            "stroke",
            "black",
        ).attr(
            "stroke-width",
            3,
        ).attr(
            "stroke-linecap",
            "round"
        )

        self.x_axis.selectAll(".tick").attr(
            "stroke",
            "black",
        ).attr(
            "stroke-width",
            1,
        )

        self.x_axis.selectAll(".tick:first-of-type").attr(
            "stroke",
            "red",
        ).attr(
            "stroke-width",
            2,
        )
        self.x_axis.selectAll(".tick:last-of-type").attr(
            "stroke",
            "red",
        ).attr(
            "stroke-width",
            2,
        )

        label_x = (
            (self.width - self.right_margin - self.left_margin) // 2 +
            self.left_margin
        )
        self.x_label = self.x_axis.append(
            "text"
        ).attr(
            "x",
            label_x,
        ).attr(
            "y",
            45,
        ).attr(
            "text-anchor",
            "middle",
        ).attr(
            "fill",
            "currentColor"
        ).text(
            self.feature_name
        ).attr(
            "class",
            "axis-label",
        )

        self.x_label_clone = self.x_label.selectAll(
            "text"
        ).clone(
            True
        ).lower().attr(
            "fill",
            "none",
        ).attr(
            "stroke-width",
            5,
        ).attr(
            "stroke-linejoin",
            "round",
        ).attr(
            "stroke",
            "white",
        )

        max_y = self._compute_class_bins()

        ########################################################################
        ## Add y-axis
        ########################################################################

        self.y_scale = d3.scaleLinear().domain(
            [0, max_y]
        ).range(
            [x_axis_y_position, self.top_margin]
        )
        y_axis = self.svg.append(
            "g"
        ).attr(
            "class",
            "y-axis",
        ).attr(
            "transform",
            f"translate({self.left_margin}, {0})"
        ).call(
            d3.axisLeft(self.y_scale).ticks(
                min(max(2, self.height // 15), 20)
            )
        )
        y_axis.selectAll("path").attr(
            "stroke",
            "black",
        ).attr(
            "stroke-width",
            3,
        ).attr(
            "stroke-linecap",
            "round"
        )
        y_axis.selectAll(".tick").attr(
            "stroke",
            "black",
        ).attr(
            "stroke-width",
            1,
        )

        self.bar_group = self.svg.append("g").attr(
            "class",
            "bar-group",
        )

        self.interval_group = self.svg.append("g").attr(
            "class",
            "interval-group",
        )

        self.disribution_group = self.svg.append("g").attr(
            "class",
            "distribution-group",
        )

        # And draw everything
        self.update_plot()

    def _compute_class_bins(self):
        self.class_bins = []
        filtered_data = list(filter(
            lambda d: (
                (d["class"] == self.class_name) if self.class_name else True
            ),
            self.data
        ))
        histogram = d3.histogram().value(
            lambda d: d[self.feature_name]
        ).domain(
            self.x_scale.domain()
        ).thresholds(
            self.x_scale.ticks(min(self.num_bins, len(filtered_data)))
        )
        self.class_bins = histogram(filtered_data)
        normalize_factor = d3.sum(
            self.class_bins,
            lambda d: d.length
        )

        max_y = 0
        for bin_d in self.class_bins:
            bin_d.norm_val = bin_d.length / normalize_factor
            max_y = max(max_y, bin_d.norm_val)
        return max_y


    # A function to call whenever the selected class changes
    @flx.action
    def update_plot(self):

        ########################################################################
        ## X-Axis Update
        ########################################################################
        self.x_scale.domain(
            [self.feature_limits[0] * 0.9, self.feature_limits[1] * 1.1]
        )
        self.svg.select(
            ".x-axis"
        ).transition(
        ).duration(
            1000
        ).call(
            d3.axisBottom(self.x_scale).ticks(
                min(max(2, self.width // 15), 25)
            )
        )
        self.x_label.text(self.feature_name)
        self.x_label_clone.text(self.feature_name)

        ########################################################################
        ## Y-Axis Update
        ########################################################################

        # Set up the y-axis accordingly after recomputing this feature's class
        # bins
        max_y = self._compute_class_bins(self.y_scale.domain())
        self.y_scale.domain(
            [0, max_y]
        )
        self.svg.select(
            ".y-axis"
        ).transition(
        ).duration(
            1000
        ).call(
            d3.axisLeft(self.y_scale).ticks(
                min(max(2, self.height // 15), 20)
            )
        )

        ########################################################################
        ## Plot data density estimation
        ########################################################################

        # Recompute the density here
        if self.plot_density:
            density = self.estimated_density
            # Time to make a pretty area diagram here!
            distribution = self.disribution_group.selectAll(
                "path"
            ).data(
                [density],
            )

            distribution_enter = distribution.enter(
            ).append(
                "path",
            ).attr(
                "class",
                "mypath",
            ).attr(
                "fill",
                "#69b3a2",
            ).attr(
                "opacity",
                0.8,
            ).attr(
                "stroke",
                "black",
            ).attr(
                "stroke-width",
                3,
            ).attr(
                "stroke-linejoin",
                "round",
            ).attr(
                "d",
                d3.line().curve(
                    d3.curveBasis
                ).x(
                    lambda d: self.x_scale(d[0])
                ).y(
                    lambda d: self.y_scale(
                        d[1]
                    )
                )
            )

            distribution_update = distribution_enter.merge(distribution)
            distribution_update.transition(
            ).duration(
                1000,
            ).attr(
                "d",
                d3.line().curve(
                    d3.curveBasis
                ).x(
                    lambda d: self.x_scale(d[0])
                ).y(
                    lambda d: self.y_scale(d[1]),
                )
            )

        ########################################################################
        ## Empirical Distribution
        ########################################################################

        empirical_distr = self.bar_group.selectAll(
            "rect",
        ).data(
            self.class_bins,
        )

        empirical_enter = empirical_distr.enter(
        ).append(
            "rect",
        ).attr(
            "x",
            1,
        ).attr(
            "transform",
            lambda d: (
                f"translate({self.x_scale(d.x0)}, {self.y_scale(d.norm_val)})"
            )
        ).attr(
            "width",
            lambda d: max(self.x_scale(d.x1) - self.x_scale(d.x0) - 1, 0),
        ).attr(
            "height",
            lambda d: (
                self.height - self.top_margin - self.bottom_margin -
                self.y_scale(d.norm_val)
            ),
        ).style(
            "fill",
            self.class_color,
        ).attr(
            "opacity",
            0.5,
        )

        # And the transition bit
        empirical_update = empirical_enter.merge(empirical_distr)
        empirical_update.transition(
        ).duration(
            1000,
        ).attr(
            "transform",
            lambda d: (
                f"translate({self.x_scale(d.x0)}, {self.y_scale(d.norm_val)})"
            )
        ).attr(
            "width",
            lambda d: max(self.x_scale(d.x1) - self.x_scale(d.x0) - 1, 0)
        ).attr(
            "height",
            lambda d: (
                self.height - self.top_margin - self.bottom_margin -
                self.y_scale(d.norm_val)
            )
        ).style(
            "fill",
            self.class_color,
        )

        empirical_exit = empirical_distr.exit().transition(
        ).duration(
            1000,
        ).attr(
            "height",
            0
        ).attr(
            "width",
            0,
        ).remove()

        ########################################################################
        ## Plot thresholds
        ########################################################################

        # Finally, update the thresholds
        max_bin_width = d3.max(
            self.class_bins,
            lambda d: max(self.x_scale(d.x1) - self.x_scale(d.x0) - 1, 0),
        )
        total_score = (
            sum(map(lambda x: x["score"], self.rule_bounds))
            if self.rule_bounds else 0
        )
        base_y = self.y_scale(max_y)
        total_height = self.y_scale(0) - self.y_scale(max_y)
        heights = [
            total_height * x["score"] / total_score
            for x in self.rule_bounds
        ]
        cum_heights = [0]
        for i, val in enumerate(heights):
            cum_heights.append(cum_heights[i] + val)
        intervals = self.interval_group.selectAll("rect").data(
            self.rule_bounds,
        )

        interval_enter = intervals.enter().append(
            "rect"
        ).attr(
            "class",
            "threshold-shading"
        ).attr(
            "x",
            # Have it "grow" from the middle of the interval by first
            # placing it in the center and then moving it left while we also
            # increase its width
            lambda d: self.x_scale(d["bounds"][0]) + (
                self.x_scale(d["bounds"][1]) - self.x_scale(d["bounds"][0])
            ) / 2
        ).attr(
            "y",
            lambda d, i: base_y + cum_heights[i]
        ).attr(
            "width",
            0,
        ).attr(
            "height",
            lambda d, i: heights[i],
        ).attr(
            "opacity",
            0.4
        ).attr(
            "fill",
            lambda d: d["color"],
        ).on(
            "mouseover",
            lambda event, d: self.tooltip.style(
                "visibility",
                "visible"
            ).html(
                f"<b>Threshold</b>: {_bound_str(d)}<br>"
                f"<b>Class</b>: {d['class']}<br>"
                f"<b>Confidence</b>: {_js_round(d['confidence'], 4)}<br>"
                f"<b>Score</b>: {_js_round(d['score'], 4)}<br>"
            )
        ).on(
            "mousemove",
            lambda event, d: self.tooltip.style(
                "top",
                f"{(event.pageY - 10)}px"
            ).style(
                "left",
                f"{(event.pageX + 10)}px",
            )
        ).on(
            "mouseout",
            lambda event, d: self.tooltip.style("visibility", "hidden")
        )

        interval_update = interval_enter.merge(intervals)
        interval_update.transition().duration(
            1000,
        ).attr(
            "x",
            lambda d: self.x_scale(d["bounds"][0]),
        ).attr(
            "y",
            lambda d, i: base_y + cum_heights[i]
        ).attr(
            "width",
            lambda d: (
                max(
                    self.x_scale(d["bounds"][1]) - self.x_scale(d["bounds"][0]),
                    max_bin_width,
                ),
            ),
        ).attr(
            "height",
            lambda d, i: heights[i],
        ).attr(
            "fill",
            lambda d: d["color"],
        )

        interval_exit = intervals.exit().transition(
        ).duration(
            1000,
        ).attr(
            "x",
            lambda d: self.x_scale(d["bounds"][0]) + (
                self.x_scale(d["bounds"][1]) - self.x_scale(d["bounds"][0])
            ) / 2
        ).attr(
            "width",
            0,
        ).remove()


def _collapse_intervals(intervals, fuse_intervals=False):
    result = []
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x["bounds"][0])
    current_low, current_high = intervals[0]["bounds"]
    total_score = 0
    cls_name = intervals[0]["class"]  # For now assume they all have the same
                                      # class
    confidence = intervals[0]["confidence"]
    color = intervals[0]["color"]
    current_left_inclusive = intervals[0]["left_inclusive"]
    current_right_inclusive = intervals[0]["right_inclusive"]
    for interval in intervals:
        (next_low, next_high) = interval["bounds"]
        if ((next_low, next_high) == (current_low, current_high)) or (
            fuse_intervals and (
                (next_low < current_high) or
                (next_low == current_high) and (current_right_inclusive)
            )
        ):
            # Then merge these two!
            total_score += interval["score"]
            confidence = min(confidence, interval["confidence"])
            current_high = max(current_high, next_high)
        else:
            # Else time to add the current interval to our list and move on
            # to form the following one
            result.append({
                "class": cls_name,
                "score": total_score,
                "color": color,
                "confidence": confidence,
                "bounds": (current_low, current_high),
            })
            total_score = 0
            confidence = interval["confidence"]
            cls_name = interval["class"]
            color = interval["color"]
            current_low, current_high = next_low, next_high

    # And we need to add the current interval here
    result.append({
        "class": cls_name,
        "score": total_score,
        "color": color,
        "confidence": confidence,
        "bounds": (current_low, current_high),
        "left_inclusive": current_left_inclusive,
        "right_inclusive": current_right_inclusive,
    })
    result.sort(key=lambda x: (x["bounds"][1] - x["bounds"][0]))
    return result


class FeatureBoundComponent(flx.PyWidget):
    """
    Widget for describing the bounds for a given feature in the application-wide
    shared rule set.
    """
    feature = flx.AnyProp(settable=True)
    feature_limits = flx.TupleProp(settable=True)
    class_name = flx.StringProp(settable=True)
    fuse_intervals = flx.BoolProp(False, settable=True)

    def init(self):

        self.view = FeatureBoundView(
            feature_name=self.feature,
            feature_limits=self.feature_limits,
            class_name=self.class_name,
        )
        self.update_feature()

    @flx.action
    def update_feature(self):
        ruleset = self.root.state.ruleset
        self.interval_map = defaultdict(list)
        self.classes = list(ruleset.output_class_map.keys())
        for rule in ruleset:
            for clause in rule.premise:
                for term in clause.terms:
                    if term.variable != self.feature:
                        continue
                    # Then include this interval in its corresponding class
                    if term.operator == TermOperator.GreaterThan:
                        bound = (term.threshold, self.feature_limits[1])
                    else:
                        bound = (self.feature_limits[0], term.threshold)
                    self.interval_map[rule.conclusion].append({
                            "color": _CLASS_PALETTE[
                                self.classes.index(rule.conclusion) % len(
                                    _CLASS_PALETTE
                                )
                            ],
                            "bounds": bound,
                            "left_inclusive": False,
                            "right_inclusive": True,
                            "class": rule.conclusion,
                            "score": clause.score,
                            "confidence": clause.confidence,
                        }
                    )
        # Time to collapse intervals
        for cls_name, intervals in self.interval_map.items():
            self.interval_map[cls_name] = _collapse_intervals(
                intervals,
                fuse_intervals=self.fuse_intervals,
            )
        data = []
        dataset = self.root.state.dataset.data
        inv_name_map = {}
        for (cls_name, cls_code) in ruleset.output_class_map.items():
            inv_name_map[cls_code] = cls_name

        is_norm = self.root.state.dataset.is_normalized(self.feature)
        if is_norm:
            # Do some min-max scaling. For that we need to find the max and the
            # min value of the dataset here
            max_val = max(dataset[self.feature])
            min_val = min(dataset[self.feature])
        for (val, cls_name) in zip(
            dataset[self.feature],
            dataset[self.root.state.dataset.target_col]
        ):
            if is_norm:
                scaled_val = (val - min_val) / (max_val - min_val)
            else:
                scaled_val = val
            data.append({
                "class": inv_name_map[cls_name],
                self.feature: scaled_val,
            })

        self.estimated_densities = {}
        for cls_name in self.classes:
            values = list(map(
                lambda x: x[self.feature],
                filter(
                    lambda x: x["class"] == cls_name,
                    data
                )
            ))
            values = np.array(values).reshape(-1, 1)
            kernel = KernelDensity(kernel='gaussian', bandwidth=0.8).fit(
                values
            )
            x_vals = np.linspace(0, 1, 200).reshape(-1, 1)
            density = np.exp(kernel.score_samples(x_vals))
            self.estimated_densities[cls_name] = list(zip(
                x_vals.flatten(),
                density.flatten(),
            ))

        self.view.set_feature_name(self.feature)
        self.view.set_feature_limits(self.feature_limits)
        self.view.set_data(data)

        # And make sure all feature-class dependencies are also handled
        self.update_class()

    @flx.action
    def update_class(self):
        self.view.set_class_name(self.class_name)
        if self.class_name == "":
            combined_intervals = []
            for cls_name, intervals in self.interval_map.items():
                for interval in intervals:
                    my_interval = interval.copy()
                    my_interval["color"] = _CLASS_PALETTE[
                        self.classes.index(cls_name) % len(_CLASS_PALETTE)
                    ]
                    combined_intervals.append(my_interval)
            combined_intervals.sort(key=lambda x: x["bounds"][0])
            combined_intervals.sort(
                key=lambda x: (x["bounds"][1] - x["bounds"][0]),
            )
            self.view.set_rule_bounds(combined_intervals)
        else:
            self.view.set_rule_bounds(self.interval_map[self.class_name])

        self.view.set_estimated_density(
            self.estimated_densities.get(self.class_name, [])
        )
        # if self.class_name == "":
        #     class_color = "black"
        # else:
        #     class_color = _CLASS_PALETTE[
        #         self.classes.index(self.class_name) % len(_CLASS_PALETTE)
        #     ]
        class_color = "black"
        self.view.set_class_color(class_color)

    @flx.action
    def update_view(self):
        self.view.update_plot()


################################################################################
## REMIX Widget Definition
################################################################################


class FeatureExplorerComponent(RemixWindow):
    """
    This widget will define a REMIX window where one is able to explore how
    each individual feature is thresholded by rules in the rule set shared
    across the entire application.

    For us to use this widget, a valid dataset must be provided.
    """

    def init(self):
        self.ruleset = self.root.state.ruleset
        self.all_features = set()
        for rule in self.ruleset.rules:
            for clause in rule.premise:
                for term in clause.terms:
                    self.all_features.add(term.variable)

        self.all_features = list(self.all_features)
        # Make sure we display most used rules first
        self.all_features = sorted(self.all_features)
        self.class_names = sorted(self.ruleset.output_class_map.keys())
        with ui.VBox(
            title="Feature Explorer",
        ):
            # Simple header first
            with ui.HBox(0.15):
                ui.Widget(flex=1)  # filler
                ui.Label(
                    text="Threshold Visualization",
                    css_class="feature-explorer-title",
                    flex=1,
                )
                ui.Widget(flex=1)  # filler

            # Add the box container our threshold visualizer
            first_feature = self.all_features[0]
            feature_limits = self.root.state.get_feature_range(
                first_feature
            )
            self.feature_view = FeatureBoundComponent(
                feature=first_feature,
                class_name="",
                feature_limits=feature_limits,
                flex=0.7,
            )

            # Finally, a simple control panel
            with ui.HBox(
                css_class='threshold-visualizer-control-panel',
                flex=0.05,
            ) as self.control_panel:
                ui.Widget(flex=1)  # filler
                with ui.VBox():
                    ui.Label(
                        text="Feature",
                        css_class="combo-box-label",
                        flex=0,
                    )
                    self.feature_selection = ui.ComboBox(
                        options=self.all_features,
                        selected_index=0,
                        css_class='explorer-selection-box',
                        flex=0,
                    )
                ui.Widget(flex=0.25)  # filler
                with ui.VBox():
                    ui.Label(
                        text="Class",
                        css_class="combo-box-label",
                        flex=0,
                    )
                    self.class_selection = ui.ComboBox(
                        options=["all classes"] + (self.class_names),
                        selected_index=0,
                        css_class='explorer-selection-box',
                        flex=0,
                    )
                ui.Widget(flex=1)  # filler
            ui.Widget(flex=0.15)  # filler

    @flx.reaction('feature_selection.user_selected')
    def select_feature(self, *events):
        """
        Reaction to a change in the feature selected for analysis.
        """

        new_feature = events[-1]['key']
        self.feature_view.set_feature(new_feature)
        self.feature_view.set_feature_limits(
            self.root.state.get_feature_range(
                new_feature
            )
        )
        self.feature_view.update_feature()
        self.feature_view.update_view()

    @flx.reaction('class_selection.user_selected')
    def select_class(self, *events):
        """
        Reaction to the change in the selected class being updated.
        """

        new_cls_name = events[-1]['key']
        if new_cls_name == "all classes":
            new_cls_name = ""
        self.feature_view.set_class_name(new_cls_name)
        self.feature_view.update_class()
        self.feature_view.update_view()

    @flx.action
    def reset(self):
        """
        Resets the entire widget to be updated using the state of the rule set
        shared across the entire application.
        """
        # TODO (mateoespinosa): implement this once it becomes relevant
        pass
