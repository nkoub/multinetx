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
MultilayerGraph: Base class for a multi-layer network
"""
from multinetx.exceptions import multinetxError
import copy
#
try:
    from networkx import Graph, disjoint_union_all
except ImportError:
    raise ImportError("NetworkX is required")
#    
try:
    from numpy import zeros
except ImportError:
    raise ImportError("NumPy is required")
#
try:
    from scipy.sparse import lil_matrix
except ImportError:
    raise ImportError("SciPy is required")
#
#
class MultilayerGraph(Graph):
    ###########################################################################
    ############  Define constructor of the class   ###########################
    ###########################################################################
    def __init__(self, 
                list_of_layers=None, 
                inter_adjacency_matrix=None,
                **attr):
        """Constructor of a MultilayerGraph. 
        It creates a symmetric (undirected) MultilayerGraph object 
        inheriting methods from networkx.Graph
        
        Parameters:
        -----------
        list_of_layers : Python list of networkx.Graph objects
         
        inter_adjacency_matrix : a lil sparse matrix (NxN) with zero 
								 diagonal elements and off-diagonal 
								 block elements defined by the 
                                 inter-connectivity architecture.
        
        Return: a MultilayerGraph object
        
        Examples:
        ---------
        import multinetx as mx
        N = 10
		g1 = mx.erdos_renyi_graph(N,0.07,seed=218)
		g2 = mx.erdos_renyi_graph(N,0.07,seed=211)
		g3 = mx.erdos_renyi_graph(N,0.07,seed=211)
                
        adj_block = mx.lil_matrix(np.zeros((N*3,N*3)))
		adj_block[0:  N,  N:2*N] = np.identity(N)    # L_12
		adj_block[0:  N,2*N:3*N] = np.identity(N)    # L_13
		#adj_block[N:2*N,2*N:3*N] = np.identity(N)    # L_23
		adj_block += adj_block.T

		mg = mx.MultilayerGraph(list_of_layers=[g1,g2,g3], 
								inter_adjacency_matrix=adj_block)

		mg.set_edges_weights(inter_layer_edges_weight=4)
		mg.set_intra_edges_weights(layer=0,weight=1)
		mg.set_intra_edges_weights(layer=1,weight=2)
		mg.set_intra_edges_weights(layer=2,weight=3)
		
		
        """       
        ## Give an empty graph in the list_of_layers
        if list_of_layers is None:
            self.list_of_layers = [Graph()]
        else:
            self.list_of_layers = list_of_layers
        
        ## Number of layers
        self.num_layers = len(self.list_of_layers)
        
        ## Number of nodes in each layer
        self.num_nodes_in_layers = self.list_of_layers[0].number_of_nodes()       
        
        ## Create the MultilayerGraph without inter-layer links.
        try:
            Graph.__init__(self,
                        Graph(disjoint_union_all(self.list_of_layers),
                        **attr))
        except multinetxError:
            raise multinetxError("Multiplex cannot inherit Graph properly")
            
        ## Check if all graphs have the same number of nodes
        for lg in self.list_of_layers:
            try:
                assert(lg.number_of_nodes() == self.num_nodes_in_layers)
            except AssertionError:    
                raise multinetxError("Graph at layer does not have")
                                     
        
        ## Make a zero lil matrix for inter_adjacency_matrix
        if inter_adjacency_matrix is None:
           inter_adjacency_matrix = \
                       lil_matrix(zeros(
                       (self.num_nodes_in_layers*self.num_layers,
                       self.num_nodes_in_layers*self.num_layers)))
        
        ## Check if the matrix inter_adjacency_matrix is lil
        try:
            assert(inter_adjacency_matrix.format == "lil")
        except AssertionError:    
            raise multinetxError("interconnecting_adjacency_matrix "\
                                 "is not scipy.sparse.lil")         
                
        ## Lists for intra-layer and inter-layer edges
        if list_of_layers is None:
		    self.intra_layer_edges = []
        else:
		    self.intra_layer_edges = self.edges()			
        self.inter_layer_edges = []
        
        ## Inter-layer connection
        self.layers_interconnect(inter_adjacency_matrix)
    
        ## MultiNetX name
        self.name = "multilayer"
        for layer in self.list_of_layers:
            self.name += "_" + layer.name   
    #:<~ Constructor  

    ###########################################################################
    ##############  Define methods of the class   #############################
    ##########################################################################
    def add_layer(self, layer, **attr):
        if self.num_nodes_in_layers is 0:
            self.list_of_layers=[layer]
        else:
            self.list_of_layers.append(layer)
            
        self.num_layers = len(self.list_of_layers)
        self.num_nodes_in_layers = self.list_of_layers[0].number_of_nodes()
        
        for i,j in layer.edges():
			self.intra_layer_edges.append((
			i+(len(self.list_of_layers)-1)*layer.number_of_nodes(),
			j+(len(self.list_of_layers)-1)*layer.number_of_nodes()))
			
        try:
            Graph.__init__(self,
                        Graph(disjoint_union_all(self.list_of_layers),
                        **attr))
        except multinetxError:
            raise multinetxError("Multiplex cannot inherit Graph properly")

        ## Check if all graphs have the same number of nodes
        for lg in self.list_of_layers:
            try:
                assert(lg.number_of_nodes() == self.num_nodes_in_layers)
            except AssertionError:
                raise multinetxError("Graph at layer does not have the same number of nodes")  
                
    #:<~  
    def layers_interconnect(self, inter_adjacency_matrix=None):
        """Parameters:
        -----------
        Examples:
        ---------
        """
        ## Make a zero lil matrix for inter_adjacency_matrix
        if inter_adjacency_matrix is None:
           inter_adjacency_matrix = \
                       lil_matrix(zeros(
                       (self.num_nodes_in_layers*self.num_layers,
                       self.num_nodes_in_layers*self.num_layers)))
        
        ## Check if the matrix inter_adjacency_matrix is lil
        try:
            assert(inter_adjacency_matrix.format == "lil")
        except AssertionError:    
            raise multinetxError("interconnecting_adjacency_matrix "\
                                 "is not scipy.sparse.lil") 
        for i,row in enumerate(inter_adjacency_matrix.rows):
            for pos,j in enumerate(row):
                if i>j:
                    self.inter_layer_edges.append((i,j))
        self.add_edges_from(self.inter_layer_edges)
    #:<~     
    def get_number_of_layers(self):
        """Return the number of graphs"""
        return self.num_layers
    #:<~ 
    def get_number_of_nodes_in_layer(self):
        """Return the number of nodes in each graph"""
        return self.num_nodes_in_layers
    #:<~
    def get_intra_layer_edges(self):
        """Return a list with the intra-layer edges"""
        return self.intra_layer_edges
    #:<~ 
    def get_intra_layer_edges_of_layer(self,layer=0):
        """Return a list with the intra-layer edges of layer"""
        edge_list = self.get_layer(layer).edges()
        elist = []
        for i,j in edge_list:
			elist.append((layer*self.get_number_of_nodes_in_layer()+i,
						  layer*self.get_number_of_nodes_in_layer()+j))
        return elist
    #:<~ 
    def get_inter_layer_edges(self):
        """Return a list with the inter-layer edges"""
        return self.inter_layer_edges    
    #:<~ 
    def get_list_of_layers(self):
        """Return a list with the graphs of the layers"""
        return self.list_of_layers
    #:<~
    def get_layer(self,layer_number):
        """Return the networkx graph of the layer layer_number"""
        return self.list_of_layers[layer_number]
    #:<~       
    def set_edges_weights(self,
                        intra_layer_edges_weight=None, 
                        inter_layer_edges_weight=None):
        """Set the weights of the MultilayerGraph edges
        ---
        intra_layer_edges_weight
        inter_layer_edges_weight
        ---
        Set the "intra_layer_edges_weight" and "intra_layer_edges_weight"
        as an edge attribute with the name "weight"
        """
        self.add_edges_from(self.intra_layer_edges,
                            weight=intra_layer_edges_weight)
        self.add_edges_from(self.inter_layer_edges,
                            weight=inter_layer_edges_weight)
    #:<~   
    def set_intra_edges_weights(self,
						layer=0,
                        weight=None):
        """Set the weights of the MultilayerGraph edges
        ---
        intra_layer_edges_weight
        ---
        Set the "intra_layer_edges_weight" and "intra_layer_edges_weight"
        as an edge attribute with the name "weight"
        """
        elist = self.get_intra_layer_edges_of_layer(layer=layer)
        self.add_edges_from(elist,weight=weight)

    #:<~         
    def info(self):
        """Returns some information of the object MultilayerGraph"""
        info = "{}-layer graph, "\
                "intra_layer_edges:{}, "\
                "inter_layer_edges:{}, "\
                "number_of_nodes_in_layer:{} "\
                .format(self.num_layers,
                len(self.intra_layer_edges),
                len(self.inter_layer_edges),
                self.num_nodes_in_layers)
        return info
    #:<~
