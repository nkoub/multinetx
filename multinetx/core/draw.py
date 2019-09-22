#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################
#
#    multiNetX -- a python package for general multilayer graphs
#
#    (C) Copyright 2013-2019, Nikos E Kouvaris
#    multiNetX is part of the deliverables of the LASAGNE project
#    (multi-LAyer SpAtiotemporal Generalized NEtworks),
#    EU/FP7-2012-STREP-318132 (http://complex.ffn.ub.es/~lasagne/)
#
#    multiNetX is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published
#    by the Free Software Foundation, either version 3 of the License,
#    or (at your option) any later version.
#
#    multiNetX is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#    See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program. If not, see http://www.gnu.org/licenses/.
########################################################################

"""
A module for ploting multiNetX
"""

import numpy as np  #  to use matrix
import matplotlib.pyplot as plt  # to use plot
import networkx as nx  # to use graphs
import math  # to use floor
import matplotlib.cm as cmx  # to use cmap (for data color values)
import matplotlib.colors as colors  # to use cmap (for data color values)
import matplotlib.cbook as cb  # to test if an object is a string
from multinetx.core.utilities import dicoLayerNode
from multinetx.core.utilities import cylinder
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d


def get_position(
    G, base_pos=None, layer_vertical_shift=2, layer_horizontal_shift=0.0, proj_angle=45
):
    """Return the position of the nodes.
    Parameters:
    -----------
    base_pos: position of base graph defualt value is None and thus function
                creates a circular layout
    layer_vertical_shift: vertical shift of the nodes coordinates compared
                            to the nodes position of the base graph
    layer_horizontal_shift: horizontal shift of the nodes coordinates
                            compared to the nodes position of the base graph
    proj_angle : angle of the tranfsormation
    Return: a dictionary with the nodes id and their coordinates
    Examples:
    ---------
    import multinetx as mx
    N = 10
    g1 = mx.erdos_renyi_graph(N,0.07,seed=218)
    g2 = mx.erdos_renyi_graph(N,0.07,seed=211)
    mg = mx.MultilayerGraph(list_of_layers=[g1,g2])
    pos = mx.get_position(mg,mx.random_layout(g1),
                          layer_vertical_shift=0.2,
                          layer_horizontal_shift=0.0,
                          proj_angle=4)
    """
    if base_pos is None:
        base_pos = nx.layout.circylar_layout(G.get_layer(0))
    else:
        base_pos = base_pos
    pos = base_pos
    N = G.get_number_of_nodes_in_layers()[0]

    for j in range(N):
        pos[j][0] *= math.cos(proj_angle)
        pos[j][1] *= math.sin(proj_angle)

    for l in range(G.get_number_of_layers()):
        if l % 2 == 0:
            ll = 1.0 * layer_horizontal_shift
        else:
            ll = -1.0 * layer_horizontal_shift
        for j in range(G.get_layer(l).number_of_nodes()):
            pos[l * N + j] = np.array(
                [pos[j][0] + ll, pos[j][1] + l * layer_vertical_shift], dtype=np.float32
            )
    return pos


def get_position3D(G, base_pos=None, x_shift=0, y_shift=0.0, z_shift=1):
    """Return the position of the nodes.
    Parameters:
    -----------
    base_pos: position of base graph defualt value is None and thus function
                creates circular layouts
    x_shift: x shift (alternate left and right)
    y_shift: y shift 
    z_shift: z shift

    Return: a dictionary with the nodes id and their coordinates

    Examples:
        %%%%%%%%%%%%%%%%%%%%%% TO DO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    if base_pos is None:
        base_pos = [
            nx.layout.circular_layout(G.get_layer(i), dim=3)
            for i in range(G.get_number_of_layers())
        ]
    else:
        base_pos = base_pos

    l_pos = []
    for e in base_pos:
        l_pos.append(e.values())
    pos = [item for sublist in l_pos for item in sublist]

    N = G.get_number_of_nodes()

    e = 0
    for l in range(G.get_number_of_layers()):
        if l % 2 == 0:
            ll = 1.0 * x_shift
        else:
            ll = -1.0 * x_shift
        for j in range(G.get_layer(l).number_of_nodes()):
            pos[e][0] = pos[e][0] + ll
            pos[e][1] = pos[e][1] + l * y_shift
            pos[e][2] = l * z_shift
            e = e + 1

    return pos


# OneLayer : Plot a layer
# ax                   Axes3DSubplot object, to plot the figure
# pos                  positions of the nodes of the layer
# graph                graph to plot
# intra_edge_color     intra edge color
# nodes_color          nodes color
# node_size            nodes size
def OneLayer(
    ax, pos, graph, intra_edge_color="black", node_color="white", node_size=25
):
    # Initialize variables
    edgelist = graph.edges()
    x1 = []
    y1 = []
    z1 = []
    # Add all node coordinates to x1, y1 and z1
    N = len(pos)
    for n in range(0, N):
        x1.append(pos[n][0])
        y1.append(pos[n][1])
        z1.append(pos[n][2])

    # Add edges to the figure
    for e in edgelist:
        # Create start of the edge
        xedge1 = pos[e[0]][0]
        yedge1 = pos[e[0]][1]
        zedge1 = pos[e[0]][2]
        # Create end of the edge
        xedge2 = pos[e[1]][0]
        yedge2 = pos[e[1]][1]
        zedge2 = pos[e[1]][2]
        # Create edge
        xedge = [xedge1, xedge2]
        yedge = [yedge1, yedge2]
        zedge = [zedge1, zedge2]
        # Add edge to plot
        ax.plot(xedge, yedge, zedge, color=intra_edge_color)

    ax.scatter(x1, y1, z1, s=node_size, c=node_color, depthshade=False)
    return ax


# Interlayer : plot the interlayer edges
# multinet    multinet to plot interlayer edges
# pos         positions of the nodes
# ax          Axes3DSubplot object, to plot the figure
# inter_edge_color     inter edge color
def InterLayer(multinet, pos, ax, inter_edge_color="grey"):

    edges = multinet.get_inter_layer_edges()
    xedge = []
    yedge = []
    zedge = []

    for e in edges:
        x1 = pos[e[0]][0]
        y1 = pos[e[0]][1]
        z1 = pos[e[0]][2]
        x2 = pos[e[1]][0]
        y2 = pos[e[1]][1]
        z2 = pos[e[1]][2]
        xedge = [x1, x2]
        yedge = [y1, y2]
        zedge = [z1, z2]
        ax.plot(xedge, yedge, zedge, color=inter_edge_color)

    return ax


# FigureByLayer : plot a 3D multinet figure by printing each layer and inter edges
# multinet              multinet to plot
# pos                   positions of the nodes
# ax                    Axes3DSubplot object, to plot the figure
# intra_edge_color      color of intra edges
#                      (can be one color or a list of color
#                       of number of layer size)
# inter_edge_color      color of inter edges
# nodes_color           color of nodes
# node_size             size of nodes
def FigureByLayer(
    multinet,
    pos=None,
    ax=None,
    intra_edge_color="black",
    inter_edge_color="grey",
    node_color="white",
    node_size=25,
):

    # Initialize number of layer in the multinet
    L = multinet.get_number_of_layers()

    # Assertion : intra_edge_color is valid
    if len(intra_edge_color) != L and not colors.is_color_like(intra_edge_color):
        raise ValueError(
            "intra_edge_code must be a color or a size number of layers list"
        )

    # Assertion : layer_node_color is valid
    if len(node_color) != L and not colors.is_color_like(node_color):
        raise ValueError("nodes_color must be a color or a size number of layers list")

    # Initialization ax and pos if not given by user
    if ax is None:
        ax = plt.gca()

    if pos is None:
        pos = get_position3D(multinet)

    # Duplication of colors if only one given
    ## For sake of simplicity
    if colors.is_color_like(intra_edge_color):
        intra_edge_color = [intra_edge_color for i in range(L)]
    if colors.is_color_like(node_color):
        node_color = [node_color for i in range(L)]

    # Initialize number of node in a layer
    N = multinet.get_number_of_nodes()

    # Plot all graph in multinet one by one
    start_pos = 0
    k = 0
    for graph in multinet.get_list_of_layers():

        n = graph.number_of_nodes()
        # get position in list
        pos_num = graph.number_of_nodes()
        end_pos = start_pos + pos_num
        g_pos = pos[start_pos:end_pos][:][:]

        ax = OneLayer(
            ax,
            g_pos,
            graph,
            intra_edge_color=intra_edge_color[k],
            node_color=node_color[k],
            node_size=node_size,
        )
        k = k + 1
        start_pos = end_pos
    # Plot interlayer between graphs
    ax = InterLayer(multinet, pos, ax, inter_edge_color)
    # Show the created figure
    plt.show()
    return


# Check if a value is numerical
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# Figure3D : plot a 3D multinet figure as a single graph
# multinet              multinet to plot
# Optional :
# pos                   1x3 array list, Positions of the nodes
# ax                    Axes3DSubplot object, Axes to plot the figure
# node_linewidth        float, Thickness of the contour line of nodes
# edge_linewidth        float, Thickness of edge line
# edge_style            string, Edge line style
#                       'solid' | 'dashed' | 'dotted' | 'dashdot'
# node_list             int list, List of nodes to draw
# edge_list             pairs list, List of edge to draw
# node_shape            string, Shape of nodes
# node_color            color string or array of floats, Color of nodes
# node_cmap             Matplotlib colormap, Color map used for nodes
# node_size             scalar or array, Size of nodes
# node_alpha            float, Node transparency (1 = opaque, 0 = transparent)
# node_vmin             float, Minimum for node colormap scaling
# node_vmax             float, Maximum for node colormap scaling
# edge_color            color string or array of floats, Color of edges
# edge_cmap             Matplotlib colormap, Color map used for edges
# egde_alpha            float, Edge transparency (1 = opaque, 0 = transparent)
# egde_vmin             float, Minimum for edge colormap scaling
# edge_vmax             float, Maximum for edge colormap scaling
# label                 string, Label for graph legend
# with_labels           bool, Set to true to draw labels on the nodes
# labels                dictionary, Node labels in a dictionary keyed by node
#                       of text label
# font_size             int, Size of labels
# font_color            string, Solor of labels
# font_weight           string, Weight of labels
#                       a numeric value in range 0-1000 | ‘ultralight’ |
#                       ‘light’ | ‘normal’ | ‘regular’ | ‘book’ | ‘medium’ |
#                       ‘roman’ | ‘semibold’ | ‘demibold’ | ‘demi’ |
#                       ‘bold’ | ‘heavy’ | ‘extra bold’ | ‘black’
# font_family           string, Family of labels
#                       FONTNAME | ‘serif’ | ‘sans-serif’ | ‘cursive’ |
#                       ‘fantasy’ | ‘monospace’
def Figure3D(
    multinet,
    pos=None,
    ax=None,
    node_linewidth=None,
    edge_linewidth=1,
    edge_style="solid",
    node_list=None,
    edge_list=None,
    node_shape="o",
    node_color=None,
    node_cmap=None,
    node_size=25,
    node_alpha=1,
    node_vmin=None,
    node_vmax=None,
    edge_color=None,
    edge_cmap=None,
    edge_alpha=1,
    edge_vmin=None,
    edge_vmax=None,
    label=None,
    with_labels=False,
    labels=None,
    font_size=12,
    font_color="k",
    font_weight="normal",
    font_family="sans-serif",
):

    # Initialize non initialize parameters
    if ax is None:
        ax = plt.gca()

    if pos is None:
        pos = get_position3D(multinet)

    if edge_cmap is None:
        edge_cmap = plt.get_cmap("jet")

    if node_cmap is None:
        node_cmap = plt.get_cmap("jet")

    if node_list is None:
        node_list = multinet.nodes()

    if edge_list is None:
        edge_list = multinet.edges()

    if labels is not None:
        with_labels = True

    N = len(node_list)
    E = len(edge_list)

    # node_color treatment
    # If no value of node_color
    if node_color is None:
        node_color = ["white" for node in multinet.nodes()]
        node_scalarMap = None
    # If just one color, duplicate
    elif colors.is_color_like(node_color):
        node_color = [node_color for i in range(N)]
    # If color values of the good size
    elif len(node_color) == N and all(
        colors.is_color_like(color) for color in node_color
    ):
        {}  # it's ok
    # If numerical values of the good size
    elif len(node_color) == N and all(isfloat(color) for color in node_color):
        if node_vmin is None:
            node_vmin = min(node_color)
        if node_vmax is None:
            node_vmax = max(node_color)
        node_cNorm = colors.Normalize(node_vmin, vmax=node_vmax)
        node_scalarMap = cmx.ScalarMappable(norm=node_cNorm, cmap=node_cmap)
        nc = node_color
        node_color = [node_scalarMap.to_rgba(nc[i]) for i in range(N)]
    # If none of the planned cases
    else:
        raise ValueError("node_color not compatible")

    # edge_color treatment
    # If no value of edge_color
    if edge_color is None:
        edge_color = ["black" for edge in multinet.edges()]
        edge_scalarMap = None
    # If just one color, duplicate
    elif colors.is_color_like(edge_color):
        edge_color = [edge_color for i in range(E)]
    # If color values of the good size
    elif len(edge_color) == E and all(
        colors.is_color_like(color) for color in edge_color
    ):
        {}  # it's ok
    # If numerical values of the good size
    elif len(edge_color) == E and all(isfloat(color) for color in edge_color):
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        edge_cNorm = colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_scalarMap = cmx.ScalarMappable(norm=edge_cNorm, cmap=edge_cmap)
        ec = edge_color
        edge_color = [edge_scalarMap.to_rgba(ec[i]) for i in range(E)]
    # If none of the planned cases
    else:
        raise ValueError("edge_color not compatible")

    # Initialize variables
    x1 = []
    y1 = []
    z1 = []
    # Add all node coordinates to x1, y1 and z1
    for n in node_list:
        x1.append(pos[n][0])
        y1.append(pos[n][1])
        z1.append(pos[n][2])

    # Add edge to plot
    i = 0
    for e in edge_list:
        # Create start of the edge
        xedge1 = pos[e[0]][0]
        yedge1 = pos[e[0]][1]
        zedge1 = pos[e[0]][2]
        # Create end of the edge
        xedge2 = pos[e[1]][0]
        yedge2 = pos[e[1]][1]
        zedge2 = pos[e[1]][2]
        # Create edge
        xedge = [xedge1, xedge2]
        yedge = [yedge1, yedge2]
        zedge = [zedge1, zedge2]
        # Add edge to plot
        ax.plot(
            xedge,
            yedge,
            zedge,
            color=edge_color[i],
            alpha=edge_alpha,
            linewidth=edge_linewidth,
            linestyle=edge_style,
        )
        i = i + 1

    # Add nodes to the figure
    ax.scatter(
        x1,
        y1,
        z1,
        s=node_size,
        c=node_color,
        alpha=node_alpha,
        depthshade=False,
        marker=node_shape,
        linewidths=node_linewidth,
        label=label,
    )

    # Add label if necessary
    if with_labels:

        if labels is None:
            labels = dict((n, n) for n in multinet.nodes())

        for n, label in labels.items():
            (x, y, z) = pos[n]
            if not cb.is_string_like(label):
                label = str(label)  # this will cause "1" and 1 to be labeled the same
            t = ax.text(
                x,
                y,
                z,
                label,
                size=font_size,
                color=font_color,
                family=font_family,
                weight=font_weight,
            )

    return


# layerNetwork : draw a network of layer
# multinet          Multinet
# Optional :
# ax                Axes3DSubplot object, Axes to plot the figure
# color             color, Color of the network
# pos_layer         1*5 array list, List of layer position
#                   [x,y,z,dx,dy] list
# radius_coef       float, Coefficient to change the radius of cylinders
# weight              boolean, if True, consider weight
def layerNetwork(
    multinet, ax=None, color="grey", pos_layer=None, radius_coef=0.05, weight=True
):

    # Create ax if necessary
    if ax is None:
        ax = plt.gca()

    # If pos_layer is of the wrong size
    L = multinet.get_number_of_layers()
    if pos_layer is not None and len(pos_layer) != L:
        raise ValueError("pos_layer not compatible")

    # Initialize variables
    dico = dicoLayerNode(multinet)
    network = np.zeros((L, L))

    if pos_layer is None:
        pos_layer = []
        for l in range(L):
            if l % 2 == 0:
                signe = -1
            else:
                signe = 1
            pos_layer.append(np.array([signe * 0.25, 0.25, l, 0.5, 0.5]))

    # Calculate values of links
    for e in multinet.get_inter_layer_edges():
        i = dico.get(e[0]) - 1
        j = dico.get(e[1]) - 1
        if multinet.edges.get(e).has_key("weight") and weight:
            network[i][j] += multinet.edges.get(e)["weight"]
        else:
            network[i][j] += 1
    # Draw square for each layer
    for l in range(L):
        p = Rectangle(
            (pos_layer[l][0], pos_layer[l][1]),
            pos_layer[l][3],
            pos_layer[l][4],
            color=color,
        )
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=pos_layer[l][2])
    # Draw links between layers
    for l1 in range(L):
        for l2 in range(L):
            diam = radius_coef * network[l1][l2] / len(multinet.get_inter_layer_edges())
            Xc, Yc, Zc = cylinder(
                pos_layer[l1][0] + pos_layer[l1][3] / 2,
                pos_layer[l2][0] + pos_layer[l2][3] / 2,
                pos_layer[l1][1] + pos_layer[l1][4] / 2,
                pos_layer[l2][1] + pos_layer[l2][4] / 2,
                diam,
                pos_layer[l1][2],
                pos_layer[l2][2],
            )
            ax.plot_surface(Xc, Yc, Zc, alpha=0.5, linewidth=0, color=color)

    # Show result
    plt.show()

    return


# projectionDisplay : display projection of all layer on a single one
# multinet      Multinet
# pos           1x3 array list, Positions of the nodes
# node_color    color or color list, List of colors
# node_size     integer, Size of nodes
# loop          boolean,
#               if true, draw circle around self loop node
# weight              boolean, if True, consider weight
def projectionDisplay(
    multinet,
    ax=None,
    pos=None,
    node_color=None,
    node_size=25,
    loop=True,
    weight=True,
    z=0,
):

    # Initialize number of nodes
    nb_nodes = multinet.get_layer(0).number_of_nodes()

    # Create ax if necessary
    if ax is None:
        ax = plt.gca()

    # If all layer don't have the same number of nodes
    if not all(
        layer.number_of_nodes() == nb_nodes for layer in multinet.get_list_of_layers()
    ):
        raise ValueError("Warning : Not the same number of nodes in layers")

    # If no position given
    if pos is None:
        pos = nx.layout.circular_layout(multinet.get_layer(0), dim=3)

    if node_color is None:
        node_color = ["white" for node in range(nb_nodes)]

    if len(node_color) != nb_nodes and not colors.is_color_like(node_color):
        raise ValueError("Warning : Wrong size of node_color")

    if len(pos) != nb_nodes:
        raise ValueError("Warning : Wrong size of pos")

    # Initialize variables
    edgelist = multinet.edges()
    x1 = []
    y1 = []
    z1 = []

    # Add all node coordinates to x1, y1 and z1

    for n in range(0, nb_nodes):
        x1.append(pos[n][0])
        y1.append(pos[n][1])
        z1.append(z)

    # Add edges to the figure
    network = np.zeros((nb_nodes, nb_nodes))

    for e in edgelist:
        i = e[0] % nb_nodes
        j = e[1] % nb_nodes
        if multinet.edges.get(e).has_key("weight") and weight:
            network[i][j] += multinet.edges.get(e)["weight"]
        else:
            network[i][j] += 1

    for i in range(nb_nodes):
        for j in range(nb_nodes):
            if network[i][j] != 0:
                # Create start of the edge
                xedge1 = pos[i][0]
                yedge1 = pos[i][1]
                # Create end of the edge
                xedge2 = pos[j][0]
                yedge2 = pos[j][1]
                # Create edge
                xedge = [xedge1, xedge2]
                yedge = [yedge1, yedge2]
                zedge = [z, z]
                # Add edge to plot
                ax.plot(xedge, yedge, zedge, linewidth=network[i][j], color="grey")

                if loop and i == j:
                    circle2 = plt.Circle(
                        (pos[i][0], pos[i][1]), 0.1, color="k", fill=False
                    )
                    ax.add_patch(circle2)
                    art3d.pathpatch_2d_to_3d(circle2, z=z, zdir="z")
    ax.scatter(x1, y1, z1, depthshade=False, c=node_color, s=node_size)
    plt.show()
    return
