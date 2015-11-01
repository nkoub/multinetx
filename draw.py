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
A module for ploting multiNetX
"""

import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_position(G, base_pos=None,layer_vertical_shift=2,layer_horizontal_shift=0.0,proj_angle=45):
    """Return the position of the nodes.
    Parameters:
    -----------
    base_pos : position of base graph defualt value is None and thus function creates a circular layout
    layer_vertical_shift : vertical shift of the nodes coordinates compared to the nodes position of the base graph
    layer_horizontal_shift : horizontal shift of the nodes coordinates compared to the nodes position of the base graph
    proj_angle : angle of the tranfsormation
   
    Return: a dictionary with the nodes id and their coordinates
   
    Examples:
    ---------
    import multinetx as mx
    N = 10
    g1 = mx.erdos_renyi_graph(N,0.07,seed=218)
    g2 = mx.erdos_renyi_graph(N,0.07,seed=211)
    mg = mx.MultilayerGraph(list_of_layers=[g1,g2,g3])
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
    N = G.get_number_of_nodes_in_layer()
    
    for j in range(N):        
        pos[j][0] *= math.cos(proj_angle)
        pos[j][1] *= math.sin(proj_angle)
    
    for l in range(G.get_number_of_layers()):
		if l%2 == 0:
			ll = 1.0*layer_horizontal_shift
		else:
			ll = -1.0*layer_horizontal_shift
		for j in range(N):       
			pos[l*N+j] = np.array([pos[j][0]+ll,
							pos[j][1]+l*layer_vertical_shift],
							dtype=np.float32)
    return pos
