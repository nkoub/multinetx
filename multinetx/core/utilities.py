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
A module with various utilities for the multiNetX
"""
import networkx as nx  # to use graphs
import multinetx as mx  # to use multinet
import numpy as np


def get_key_of_value(dictionay, val):
    return [k for k, v in dictionay.iteritems() if v == val]


#
#
#
# def group_nodes_of_same_degree(G, degree=None):
#     '''returns a dict with keys the degrees and values the id of the nodes
#     '''
#     deg = G.degree_property_map('out')
#     deg_dict = {}
#     for i in range(len(deg.fa)):
#         d = deg.fa[i]
#         deg_dict[d] = deg_dict.get(d,[]) + [G.vertex(i)]
#     if degree is None:
#         return deg_dict
#     else:
#         return deg_dict[degree]
# #
#
#
# def group_nodes_of_same_distance(G, root=0, distance=None):
#     '''returns a dict with keys the distances from root and values the id of the nodes
#     '''
#     from graph_tool.topology import shortest_distance
#     dist = shortest_distance(G, G.vertex(root))
#     dist_dict = {}
#     for i in range(len(dist.fa)):
#         d = dist.fa[i]
#         dist_dict[d] = dist_dict.get(d,[]) + [G.vertex(i)]
#     if distance is None:
#         return dist_dict
#     else:
#         return dist_dict[distance]
#
#
#

# Build a ring of neurons and a ring of glias connected together
# N :     number of neurons (and glias)
# R :     number of connected neighbours on each side for each glia
def ring(N, R):

    # Create glias ring
    g1 = nx.cycle_graph(N)

    for glia in range(0, N):
        for r in range(1, R + 1):
            k = glia + r
            if k > N - 1:
                k = k - N
            g1.add_edge(glia, k)
            k = glia - r
            if k < 0:
                k = k + N
            g1.add_edge(glia, k)

    # Create neurons ring
    g2 = nx.cycle_graph(N)

    # Create inter adjacency matrix
    adj_block = mx.lil_matrix(np.zeros((N * 2, N * 2)))

    adj_block[0:N, N : 2 * N] = np.identity(N)
    adj_block += adj_block.T

    mg = mx.MultilayerGraph(list_of_layers=[g1, g2], inter_adjacency_matrix=adj_block)
    return mg


# dicoLayerNode : give a dictionary associating nodes to their layer
# multinet      Multinet
def dicoLayerNode(multinet):
    # Initialization
    dico = {}
    l = 0
    n = 0
    # For each layer
    for layer in multinet.get_list_of_layers():
        l = l + 1
        # For each node in layer
        for i in range(layer.number_of_nodes()):
            # Add node to the dictionary
            dico[n] = l
            n = n + 1
    # Return dictionary
    return dico


# cylinder : return x, y and z coodinate to draw a cylinder
# xmin          abscissa center of the first circle
# xmax          abscissa center of the second circle
# ymin          ordinate center of the first circle
# ymax          ordinate center of the second circle
# radius        diameter of the cylinder
# zmin          applicate center of the first circle
# zmax          applicate center of the second circle
def cylinder(xmin, xmax, ymin, ymax, radius, zmin, zmax):
    # Initialization of classical cylinder
    n = 100
    z = np.linspace(zmin, zmax, n)
    theta = np.linspace(0, 2 * np.pi, n)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + xmin
    y_grid = radius * np.sin(theta_grid) + ymin

    # Incline cylinder depending on x
    pas = 0
    i = 1
    for x in x_grid:
        x += pas
        pas += i * (xmax - xmin) / n

    # Incline cylinder depending on y
    pas = 0
    j = 1
    for y in y_grid:
        y += pas
        pas = j * (ymax - ymin) / n

    # Return x, y and z values
    return x_grid, y_grid, z_grid


def projection(multinet, edge_weight=False, weight=False):

    nb_nodes = multinet.get_layer(0).number_of_nodes()

    if not all(
        layer.number_of_nodes() == nb_nodes for layer in multinet.get_list_of_layers()
    ):
        raise ValueError("Warning : Not the same number of nodes in layers")

    graph = nx.Graph()

    for n in range(nb_nodes):
        graph.add_node(n)

    network = np.zeros((nb_nodes, nb_nodes))

    for e in multinet.edges():
        i = e[0] % nb_nodes
        j = e[1] % nb_nodes
        if multinet.edges.get(e).has_key("weight") and edge_weight:
            network[i][j] += multinet.edges.get(e)["weight"]
        else:
            network[i][j] += 1

    for i in range(nb_nodes):
        for j in range(nb_nodes):
            if network[i][j] != 0 and i != j:
                if weight:
                    graph.add_edge(i, j, weight=network[i][j])
                else:
                    graph.add_edge(i, j)

    return graph


# degreeCentrality : give the degree of each node
# multinet            Multinet
# interconnexion      boolean, if True, consider inter-connexion edges
# weight              boolean, if True, consider weight
def degreeCentrality(multinet, inter_connexion=True, weight=True):

    nb_nodes = multinet.get_layer(0).number_of_nodes()

    # If all layer don't have the same number of nodes
    if not all(
        layer.number_of_nodes() == nb_nodes for layer in multinet.get_list_of_layers()
    ):
        raise ValueError("Warning : Not the same number of nodes in layers")

    # Initialize vector
    degrees = np.zeros((nb_nodes, 1))

    # For each edge
    for e in multinet.edges():
        i = e[0] % nb_nodes
        j = e[1] % nb_nodes
        # Add contribution only once if the node is connected to itself
        if i == j:
            if inter_connexion:
                if multinet.edges.get(e).has_key("weight") and weight:
                    degrees[i] += multinet.edges.get(e)["weight"]
                else:
                    degrees[i] += 1
        # Add contribution to each node end of edge
        else:
            if multinet.edges.get(e).has_key("weight") and weight:
                degrees[i] += multinet.edges.get(e)["weight"]
                degrees[j] += multinet.edges.get(e)["weight"]
            else:
                degrees[i] += 1
                degrees[j] += 1

    return degrees


def degreeCentrality_proj(multinet, edge_weight=False, weight=False):
    return nx.degree_centrality(projection(multinet, edge_weight, weight))


def triangles_proj(multinet, nodes=None):
    return nx.triangles(projection(multinet), nodes)


def transitivity_proj(multinet):
    return nx.transitivity(projection(multinet))


def clustering_proj(multinet, nodes=None):
    if all(multinet.edges.get(e).has_key("weight") for e in multinet.edges()):
        w = "weight"
    else:
        w = None
    return nx.clustering(projection(multinet), nodes, weight=w)


def average_clustering_proj(multinet, nodes=None, weight=None, count_zeros=True):
    return nx.average_clustering(projection(multinet), nodes, weight, count_zeros)


def square_clustering_proj(multinet, nodes=None):
    return nx.square_clustering(projection(multinet), nodes)


def eigenvector_centrality_proj(
    multinet, max_iter=100, tol=1e-06, nstart=None, weight="weight"
):
    return nx.eigenvector_centrality(
        projection(multinet), max_iter, tol, nstart, weight
    )
