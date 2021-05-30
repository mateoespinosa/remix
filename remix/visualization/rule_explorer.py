"""
Module with widgets for visually exploring rulesets using a hierarchical tree.
"""

from flexx import flx, ui
from pscript.stubs import d3, window

from .hierarchical_tree import ruleset_hierarchy_tree
from .gui_window import RemixWindow
from .rule_statistics import _CLASS_PALETTE

################################################################################
## Flexx Assets
################################################################################

flx.assets.associate_asset(__name__, 'https://d3js.org/d3.v6.min.js')
flx.assets.associate_asset(
    __name__,
    'https://d3js.org/d3-scale-chromatic.v1.min.js',
)

################################################################################
## Global Variables
################################################################################

# Max radius allowed for a single node in a tree's visualization
_MAX_RADIUS = 100


################################################################################
## HTML/SVG Construction Helper Functions
################################################################################


def _diagonal(s, d):
    """
    Helper function to produce an SVG diagonal from point s to point d.
    :param Point s: with coordinates (s.x, s.y)
    :param Point d: with coordinates (d.x, d.y)
    :return svg.Path: path containing the diagonal.
    """
    path = (
        f'M {s.y} {s.x}'
        f'C {(s.y + d.y) / 2} {s.x},'
        f'{(s.y + d.y) / 2} {d.x},'
        f'{d.y} {d.x}'
    )
    return path


################################################################################
## Flexx Helper Widgets
################################################################################


class HierarchicalTreeViz(flx.Widget):
    """
    Widget for visualizing a D3 hierarchical tree with data in each node and
    labels on its leaves.
    """

    # Property: D3 data representation of the hierarchical tree we will
    #           visualize
    data = flx.AnyProp(settable=True)

    # Property: whether or not we want a node in the tree to have a fixed
    #           radius.
    fixed_node_radius = flx.FloatProp(0, settable=True)

    # Property: all the available class names which each leaf in the tree may
    #           have.
    class_names = flx.ListProp([], settable=True)

    # Property: distance to be used between different branches in this tree that
    #           do not share the same parent.
    branch_separator = flx.FloatProp(0.2, settable=True)

    CSS = """
    .flx-HierarchicalTreeViz {
        background: #fff;
    }
    svg {
        display: block;
        margin: 0 auto;
    }

    .link {
       fill: none;
       stroke: #ccc;
       stroke-width: 2px;
    }
    """

    def init(self):
        """
        During initialization, we will load the entire tree after everything
        has been allocated in Flexx and then we will proceed and expand the
        tree.
        """
        self.node.id = self.id
        self.svg = None
        self.tooltip = None
        self.root_tree = None
        self.zoom = None

        def _startup():
            self._init_viz()
            self._load_viz()
        window.setTimeout(_startup, 500)
        window.setTimeout(lambda: self.expand_tree(), 750)

    @flx.action
    def expand_tree(self):
        """
        Action to entirely expand this tree.
        """
        self._expand_tree()

    @flx.action
    def collapse_tree(self):
        """
        Action to entirely collapse this tree.
        """
        self._collapse_tree()

    @flx.action
    def clear(self):
        """
        Clears this entire widget by removing the visualization tree from it.
        """
        if self.svg:
            self.svg.selectAll("g.node").remove()
            self.svg.selectAll("path.link").remove()

    @flx.action
    def highlight_path(self, path):
        """
        Highlights the given path.
        :param List[Tuple[d3.Node, d3.Node]] path: A valid path in the current
            hierarchical tree defined as a list of arcs from source node to
            destination node
        """
        self._color_path(path=path, color="firebrick")

    @flx.action
    def unhighlight_path(self, path):
        """
        Unhighlights the given path to the default color.
        :param List[Tuple[d3.Node, d3.Node]] path: A valid path in the current
            hierarchical tree defined as a list of arcs from source node to
            destination node
        """
        self._color_path(path=path, color="#555")

    @flx.action
    def zoom_fit(self, duration=500):
        """
        Action to force the tree to fit its allocated window size.
        :param int duration:  The duration of the transition in miliseconds.
        """
        self._zoom_fit(duration)

    def _zoom_fit(self, duration=500):
        """
        Zooms in or out to make the tree fit in the space allocated by this
        widget's parent.

        :param int duration: The duration of the zoom transition in
            miliseconds.
        """

        # First find the bounds we were given
        bounds = self.svg.node().getBBox()

        # Compute the full size using our parent
        parent = self.svg.node().parentElement
        full_width = parent.clientWidth or parent.parentNode.clientWidth
        full_height = parent.clientHeight or parent.parentNode.clientHeight
        width = bounds.width
        height = bounds.height
        # Find the mid points
        mid_x = bounds.x + width / 2
        mid_y = bounds.y + height / 2
        if (width == 0) or (height == 0):
            return

        # And do the rescaling and translation
        scale = 0.85 / max(width / full_width, height / full_height)
        translate = [
            full_width / 2 - scale * mid_x,
            full_height / 2 - scale * mid_y
        ]

        transform = d3.zoomIdentity.translate(
            translate[0],
            translate[1],
        ).scale(
            scale
        )

        # This is where the zoom call actually happens for our SVG object
        w, h = self.size
        zoom = d3.zoom().extent(
            [[0, 0], [w, h]]
        ).scaleExtent(
            [0.1, 8]
        ).on(
            "zoom",
            lambda e: self.svg.attr(
                "transform",
                e.transform,
            )
        )

        # And make it smooth by using a translation
        self.svg.transition().duration(
            duration
        ).call(
            zoom.transform,
            transform,
        )

    def _color_path(self, path, color="#555"):
        """
        Colors the given path using the given color.

        :param List[Tuple[d3.Node, d3.Node]] path: A valid path in the current
            hierarchical tree defined as a list of arcs from source node to
            destination node.
        :param str color: A valid HTML color represented as a string to use for
            coloring the path.
        """
        if not self.svg:
            # Then nothing to color in here!
            return

        # Transition links to their new color!
        self.link_update.transition().attr(
            "stroke",
            lambda d: (
                color if (d.data.name, d.parent.data.name) in path
                else "#555"
            ),
        ).attr(
            "d",
            lambda d: _diagonal(d, d.parent),
        )

    @flx.reaction
    def _resize(self):
        """
        Resizes tree accordingly if the parent's size has changed.
        """

        w, h = self.size
        if len(self.node.children) > 0:
            # if the tree is currently being shown and initialized, then let's
            # change its size attributes
            # Starting with the global svg object
            x = d3.select('#' + self.id)
            x.attr("align", "center")
            svg = self.node.children[0]
            svg.setAttribute('width', w)
            svg.setAttribute('height', h)

            # Now the main group needs to have its translation function adjusted
            x.select("svg").select("g").attr(
                "transform",
                f'translate({w//3}, {h//2})'
            )
            graph = x.select("svg").select("g")


            # Finally, let's adjust the zooming so that it is smoother and
            # makes use of the given amount of space.
            def _zoomed(e):
                trans = e.transform
                graph.attr(
                    "transform",
                    f"translate({trans.x + (w//3 * trans.k)}, {trans.y + (trans.k * h//2)}) "
                    f"scale({trans.k})"

                )

            self.zoom = d3.zoom().extent(
                [[0, 0], [w, h]]
            ).scaleExtent(
                [0.1, 8]
            ).on(
                "zoom",
                _zoomed
            )
            x.select("svg").call(self.zoom)

            # Time to redraw our graph
            self._draw_graph(self.root_tree)

    def _draw_graph(
        self,
        current,
        node_size=None,
        duration=750,
    ):
        """
        Draw the entire tree in a hierarchical fashion. We do this in a two
        pass fashion where on the first pass we compute the bounding box
        neeeded for each node to avoid overlapping and on the second pass we
        actually draw them.

        Solution inspired by: https://observablehq.com/@d3/collapsible-tree

        :param D3.Node current: The current root node we will be
            drawing.
        :param int node_size: The size of the current node we are visualizing.
        :param int duration: The duration of each transition in miliseconds.
        """
        ########################################################################
        ## Dimensions setup
        ########################################################################

        max_num_children = self.root_tree.data.num_descendants
        if self.fixed_node_radius:
            _node_radius = lambda d: self.fixed_node_radius
        else:
            # Compute the radius of the descendants
            _node_radius_descendants = lambda num: num/max_num_children
            _node_radius = lambda d: (
                max(
                    5,
                    _MAX_RADIUS * _node_radius_descendants(
                        d.data.num_descendants
                    )
                ) if d.data.depth else _MAX_RADIUS//2
            )
        if node_size == (0, 0):
            # Then we do not draw this just yet
            return

        self._treemap = d3.tree().size(self.size)
        if node_size:
            dx, dy = node_size
            self.root_tree.dx = dx
            self.root_tree.dy = dy
            self._treemap.nodeSize(
                [dx, dy]
            )

            def _separation_function(a, b):
                return (
                    (max(a.bounding_box[1], b.bounding_box[1])/dx) + 0.01
                    if a.parent == b.parent else self.branch_separator
                )
            self._treemap.separation(
                _separation_function
            )

        ########################################################################
        ## Construct Hierarchy Coordinates
        ########################################################################

        tree_data = self._treemap(self.root_tree)

        # Compute the new tree layout.
        nodes = tree_data.descendants()
        links = tree_data.descendants().slice(1)

        ########################################################################
        ## Draw Nodes
        ########################################################################

        def _set_node(d):
            if hasattr(d, "id") and d.id:
                return d.id
            self._id_count += 1
            d.id = self._id_count
            return d.id

        node = self.svg.selectAll("g.node").attr(
            "stroke-linejoin",
            "round"
        ).attr(
            "stroke-width",
            3,
        ).data(
            nodes,
            _set_node
        )

        # Click function for our node
        def _node_click(event, d):
            if d.children:
                # Then hide its children but save the previous children
                d._children = d.children
                d.children = None
            elif hasattr(d, "_children") and d._children:
                d.children = d._children
                d._children = None
            self._draw_graph(
                d,
                node_size,
                duration,
            )

        # Create our nodes with our recursive clicking function
        node_enter = node.enter().append("g").attr(
            "class",
            "node",
        ).attr(
            "id",
            lambda d: f"node-id-{self.id}-{d.id}"
        ).attr(
            "transform",
            lambda d: f"translate({current.y0}, {current.x0})",
        ).style(
            "opacity",
            1,
        ).on(
            "click",
            _node_click,
        ).attr(
            'cursor',
            lambda d: 'pointer' if d.children or d._children else 'default',
        )

        ########################################################################
        ## Draw Children Distribution for Given Node
        ########################################################################

        # Use the same colors as in the other class plots for consistency
        colors = lambda i: _CLASS_PALETTE[i % len(_CLASS_PALETTE)]

        # Compute the position of each group on the pie:
        pie = d3.pie().value(
            lambda d: d.value
        )

        def pie_arc_generator(d):
            # Generates the arc corresponding to the given class for a node with
            # data `d`.
            entries = []
            idx = 0
            i = 0
            for (key, val) in d.data.class_counts.items():
                if key == d.key:
                    idx = i
                entries.append({
                    "key": key,
                    "value": val,
                })
                i += 1
            chunks = pie(entries)
            return d3.arc().innerRadius(0).outerRadius(_node_radius(d))(
                chunks[idx]
            )

        def _pie_data_handler(d):
            # Generates the data necessary to produce our pie chart for each
            # node with data `d`.
            entries = []
            total_sum = 0
            for key, val in d.data.class_counts.items():
                total_sum += val

            for key in sorted(d.data.class_counts.keys()):
                entries.append({
                    "key": key,
                    "data": d.data,
                    "percent": d.data.class_counts[key]/total_sum,
                    "children": d.children or d._children,
                })
            return entries

        _class_to_color = {}
        for i, cls_name in enumerate(self.class_names):
            _class_to_color[cls_name] = colors(i)

        pie_data = node_enter.selectAll("g").data(
            _pie_data_handler
        )
        pie_enter = pie_data.enter().append("path").attr(
            "class",
            "pie_arc",
        ).attr(
            "stroke",
            "black"
        ).style(
            "stroke-width",
            "1px",
        ).style(
            "fill",
            lambda d: _class_to_color[d.key],
        ).attr(
            "d",
            pie_arc_generator,
        ).style(
            "opacity",
            1,
        ).on(
            # When the mouse is over, show the tooltip for its distribution.
            "mouseover",
            lambda event, d: self.tooltip.style(
                "visibility",
                "visible"
            ).html(
                (
                    f"<b>{d.key}</b>: {d.percent*100:.3f}% "
                    f"(count {d.data.class_counts[d.key]})"
                ) if d.children else f"<b>score</b>: {d.data.score}",
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

        # Add text with low opacity for now
        node_enter.append("text").attr(
            "dy",
            "0.35em"
        )

        # Transition nodes to their new position
        node_update = node_enter.merge(node)
        node_update.transition().duration(
            duration
        ).attr(
            "transform",
            lambda d: f"translate({d.y}, {d.x})",
        )

        pie_update = pie_enter.merge(pie_data)
        pie_update.on(
            "mouseover",
            lambda event, d: self.tooltip.style(
                "visibility",
                "visible"
            ).html(
                f"<b>{d.key}</b>: {d.percent*100:.3f}%"
                if d.children else f"<b>score</b>: {d.data.score}",
            )
        ).transition().duration(
            duration
        ).style(
            "fill",
            lambda d: _class_to_color[d.key],
        )

        # And also their text
        node_update.select("text").style(
            "fill-opacity",
            1
        ).attr(
            "text-anchor",
            lambda d: "end" if d.children or d._children else "start"
        ).html(
            lambda d: d.data.name,
        ).attr(
            "font-weight",
            lambda d: "normal" if d.children or d._children else "bolder",
        ).attr(
            "font-size",
            lambda d: 10 if d.children or d._children else 19,
        ).attr(
            "x",
            lambda d: (
                -(_node_radius(d) + 3) if d.children or d._children
                else (_node_radius(d) + 3)
            ),
        ).attr(
            "fill",
            lambda d: (
                "black" if d.children or d._children
                else _class_to_color[d.data.name]
            )
        )

        if node_size is None:
            def _compute_bounding_box(d, i):
                bbox = d3.select(f"#node-id-{self.id}-{d.id}").node().getBBox()
                d["bounding_box"] = (bbox.width, bbox.height)
            node_update.each(_compute_bounding_box)
            dx = d3.max(nodes, lambda d: d.bounding_box[1])
            dy = d3.max(nodes, lambda d: d.bounding_box[0])
            # And re-run this whole thing with our corrected bounding box values
            self._draw_graph(
                current,
                [dx, dy],
                duration,
            )
            return

        # Transition function for the node exit
        # First remove our node
        node_exit = node.exit().transition().duration(
            duration
        ).attr(
            "transform",
            lambda d: f"translate({current.y}, {current.x})",
        ).remove()

        # Then make the path transparent
        node_exit.select("g.path").attr(
            "opacity",
            1e-6,
        )

        # And the text as well
        node_exit.select("text").style(
            "fill-opacity",
            1e-6,
        )

        ########################################################################
        ## Draw Links
        ########################################################################

        link = self.svg.selectAll("path.link").data(
            links,
            lambda d: d.id,
        )

        link_enter = link.enter().insert(
            "path",
            "g"
        ).attr(
            "class",
            "link",
        ).attr(
            "d",
            # Links for now will be kept invisible by remaining in the
            # same spot
            lambda d: _diagonal(
                {"x": current.x0, "y": current.y0},
                {"x": current.x0, "y": current.y0},
            )
        ).attr(
            "fill",
            "none",
        ).attr(
            "stroke",
            lambda d: "#555",
        ).attr(
            "stroke-opacity",
            lambda d: 1,
        ).attr(
            "stroke-width",
            lambda d: 15,
        )

        # Transition links to their new position when the clicking happens
        self.link_update = link_enter.merge(link)
        self.link_update.transition().duration(
            duration
        ).attr(
            "d",
            lambda d: _diagonal(d, d.parent),
        )

        # And put back exit links into hiding
        link.exit().transition().duration(
            duration
        ).attr(
            "d",
            lambda d: _diagonal(
                {"x": current.x, "y": current.y},
                {"x": current.x, "y": current.y},
            )
        ).remove()

        # Finally, save old positions so that we can transition next
        def _save_positions(d):
            d.x0 = d.x
            d.y0 = d.y
        nodes.forEach(_save_positions)

    @flx.reaction('data')
    def reload_viz(self, *events):
        """
        Reaction to changes in the data which force a redraw of this tree.
        """
        if self.svg:
            self.svg.selectAll("g.node").remove()
            self.svg.selectAll("path.link").remove()
            self._load_viz()

    def _init_viz(self):
        """
        Preamble to initialize the visualization of the hierarchical tree.
        """
        x = d3.select('#' + self.id)
        width, height = self.size
        width = max(width, 600)
        height = max(height, 600)
        self.svg = x.append("svg").attr(
            "width",
            width
        ).attr(
            "height",
            height
        ).append(
            "g"
        ).attr(
            "transform",
            f"translate({width//3}, {height//3})"
        )
        x.attr("align", "center")

        # Generate a tooltip for displaying different messages
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

    def _load_viz(self):
        """
        Loads the visualization and sets up the size according to the Flexx
        size given to the parent.
        """

        self._id_count = 0
        _, height = self.size
        self.root_tree = d3.hierarchy(
            self.data,
            lambda d: d.children,
        )
        self.root_tree.x0 = height/2
        self.root_tree.y0 = 0
        self._expand_tree()

    def _collapse_tree(self, root_too=False):
        """
        Collapses the hierarchical tree so that only one level remains visible.

        :param bool root_too: Whether or not we want to collapse the root too
            or not.
        """

        def _collapse(d):
            # Recursively collapse node with data `d`.
            if d.children:
                d._children = d.children
                d._children.forEach(_collapse)
                d.children = None
            elif hasattr(d, "_children") and d._children:
                # Then make sure we collapse all inner children here as well
                # in case things have been partially collapsed
                d._children.forEach(_collapse)

        # Time to call this accordingly
        if root_too:
            _collapse(self.root_tree)
        elif self.root_tree.children:
            self.root_tree.children.forEach(_collapse)

        # And redraw everything
        self._draw_graph(self.root_tree)

    def _expand_tree(self):
        """
        Fully expands the hierarchical tree so all levels are visible.
        """

        def _expand_node(d):
            # Recursive function to expand node with data `d`.
            if hasattr(d, "_children") and (d._children):
                d.children = d._children
                d._children = None
            if d.children:
                d.children.forEach(_expand_node)
        if not self.root_tree.children:
            if (
                hasattr(self.root_tree, "_children") and
                (self.root_tree._children is not None)
            ):
                self.root_tree.children = self.root_tree._children
                self.root_tree._children = None

        # Time to call the function in our tree
        if self.root_tree.children:
            self.root_tree.children.forEach(_expand_node)

        # And redraw everything
        self._draw_graph(self.root_tree)

################################################################################
## Main REMIX Widget
################################################################################

class RuleExplorerComponent(RemixWindow):
    """
    This class describes a widget able of visually representing a rule set in
    a tree format for clarity purposes.
    This visualization contains controls for collapsing and expanding the tree
    which can be useful.
    """

    def init(self):
        with ui.VSplit(
            title="Rule Explorer",
        ):
            # Add the actual tree visualization
            self.tree = HierarchicalTreeViz(
                data=ruleset_hierarchy_tree(
                    ruleset=self.root.state.ruleset,
                    dataset=self.root.state.dataset,
                    merge=self.root.state.merge_branches,
                ),
                class_names=self.root.state.ruleset.output_class_names(),
                flex=0.95,
            )

            # And include a control panel to make visualization easier
            with ui.HBox(0.05):
                ui.Widget(flex=1)
                self.expand_button = flx.Button(
                    text='Expand Tree',
                    flex=0,
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=0.25)
                self.collapse_button = flx.Button(
                    text='Collapse Tree',
                    flex=0,
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=0.25)
                self.fit_button = flx.Button(
                    text="Fit to Screen",
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=1)
            ui.Widget(flex=0.05)

    @flx.reaction('expand_button.pointer_click')
    def _expand_clicked(self, *events):
        """
        Reaction for when expand button has been clicked. Make the whole tree
        visualizer expand.
        """
        self.tree.expand_tree()
        self.tree.zoom_fit()

    @flx.reaction('collapse_button.pointer_click')
    def _collapse_clicked(self, *events):
        """
        Reaction for when the collapse button has been clicked. Make the whole
        tree visualizer collapse to a single hierarchy level.
        """
        self.tree.collapse_tree()
        self.tree.zoom_fit()

    @flx.reaction('fit_button.pointer_click')
    def _fit_clicked(self, *events):
        """
        Reaction to click in fit screen button. Zoom out the tree so that it
        fits the screen.
        """
        self.tree.zoom_fit()

    @flx.action
    def reset(self):
        """
        Resets this entire widget by updating the tree to use the new shared
        rule set in its hierarchical tree format.
        """
        self.tree.set_data(ruleset_hierarchy_tree(
            ruleset=self.root.state.ruleset,
            dataset=self.root.state.dataset,
            merge=self.root.state.merge_branches,
        ))
