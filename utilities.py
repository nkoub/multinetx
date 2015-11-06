#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################
#    
#    multiNetX -- a python package for general multilayer graphs
#
#    (C) Copyright 2013-2015, Nikos E Kouvaris
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
#
#
#
def get_key_of_value(dictionay,val):
    return [k for k,v in dictionay.iteritems() if v == val]
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


    
