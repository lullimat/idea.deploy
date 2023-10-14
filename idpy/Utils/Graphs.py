__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2023 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
__credits__ = ["Matteo Lulli", "Emanuele Zappala", "Antonino Marciano"]
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
__version__ = "0.1"
__maintainer__ = "Matteo Lulli, Emanuele Zappala"
__email__ = "matteo.lulli@gmail.com"
__status__ = "Development"

import matplotlib.pyplot as plt

from idpy.Utils.Geometry import TriDet
from idpy.Utils.Statements import AllTrue

'''
- half-edges order: first vertex is in the bulk, second is virtual
- vvertices, hedges and vcoords are mainly used for plotting purposes
- dictionaries are used for half-edges and virtual vertices, i.e. boundary quantities
    in order to ease the indexing via tuples
- bulk, non-modifiable, quantities are stored in lists.
'''
class Graph:
    def __init__(self):
        self.labels, self.labels_edges, self.hlabels = {}, {}, {}
        self.edges, self.hedges = [], []
        self.vertices, self.coords = [], []
        self.vvertices, self.vcoords = [], {}
        self.planar_flag = None
        
    def FindTriangles(self):
        triangles_list = []
        for vertex_i in self.vertices:
            for vertex_ii in self.adj_lists[vertex_i]:
                not_in_list_ii = [vertex_i]
                if vertex_ii not in not_in_list_ii:
                    for vertex_iii in self.adj_lists[vertex_ii]:
                        not_in_list_iii = [vertex_ii]
                        if vertex_iii not in not_in_list_iii:
                            for vertex_iv in self.adj_lists[vertex_iii]:
                                not_in_list_iv = [vertex_iii]
                                if vertex_iv not in not_in_list_iv and vertex_iv == vertex_i:
                                    vertices = [vertex_i, vertex_ii, vertex_iii]
                                    min_vertex = min(vertices)
                                    min_index = vertices.index(min_vertex)
                                    
                                    permuted_vertices = []
                                    for i in range(3):
                                        permuted_vertices += [vertices[(min_index + i) % 3]]
                                        
                                    permuted_vertices_palyndrome = [permuted_vertices[0]]
                                    permuted_vertices_palyndrome += reversed(permuted_vertices[1:])
                                    
                                    if permuted_vertices not in triangles_list and \
                                        permuted_vertices_palyndrome not in triangles_list:
                                        triangles_list += [permuted_vertices]
        return triangles_list
    
    def FindSquares(self):
        squares_list = []
        for vertex_i in self.vertices:
            for vertex_ii in self.adj_lists[vertex_i]:
                not_in_list_ii = [vertex_i]
                if vertex_ii not in not_in_list_ii:
                    for vertex_iii in self.adj_lists[vertex_ii]:
                        not_in_list_iii = [vertex_ii, vertex_i]
                        if vertex_iii not in not_in_list_iii:
                            for vertex_iv in self.adj_lists[vertex_iii]:
                                not_in_list_iv = [vertex_iii, vertex_ii]
                                if vertex_iv not in not_in_list_iv:
                                    for vertex_v in self.adj_lists[vertex_iv]:
                                        not_in_list_v = [vertex_iv, vertex_iii]
                                        if vertex_v not in not_in_list_v and vertex_v == vertex_i:
                                            vertices = \
                                                [vertex_i, vertex_ii, vertex_iii, vertex_iv]
                                            min_vertex = min(vertices)
                                            min_index = vertices.index(min_vertex)

                                            permuted_vertices = []
                                            for i in range(4):
                                                permuted_vertices += [vertices[(min_index + i) % 4]]

                                            permuted_vertices_palyndrome = [permuted_vertices[0]]
                                            permuted_vertices_palyndrome += reversed(permuted_vertices[1:])

                                            if permuted_vertices not in squares_list and \
                                                permuted_vertices_palyndrome not in squares_list:
                                                squares_list += [permuted_vertices]
        return squares_list
    
    def FindPentagons(self):
        pentagons_list = []
        for vertex_i in self.vertices:
            for vertex_ii in self.adj_lists[vertex_i]:
                not_in_list_ii = [vertex_i]
                if vertex_ii not in not_in_list_ii:
                    for vertex_iii in self.adj_lists[vertex_ii]:
                        not_in_list_iii = [vertex_i, vertex_ii]
                        if vertex_iii not in not_in_list_iii:
                            for vertex_iv in self.adj_lists[vertex_iii]:
                                not_in_list_iv = [vertex_i, vertex_ii, vertex_iii]
                                if vertex_iv not in not_in_list_iv:
                                    for vertex_v in self.adj_lists[vertex_iv]:
                                        not_in_list_v = [vertex_ii, vertex_iii, vertex_iv]
                                        if vertex_v not in not_in_list_v:
                                            for vertex_vi in self.adj_lists[vertex_v]:
                                                not_in_list_vi = [vertex_iii, vertex_iv, vertex_v]
                                                if vertex_vi not in not_in_list_vi and vertex_vi == vertex_i:
                                                    vertices = \
                                                        [vertex_i, vertex_ii, 
                                                         vertex_iii, vertex_iv, vertex_v]
                                                    
                                                    min_vertex = min(vertices)
                                                    min_index = vertices.index(min_vertex)

                                                    permuted_vertices = []
                                                    for i in range(5):
                                                        permuted_vertices += [vertices[(min_index + i) % 5]]

                                                    permuted_vertices_palyndrome = [permuted_vertices[0]]
                                                    permuted_vertices_palyndrome += reversed(permuted_vertices[1:])

                                                    if permuted_vertices not in pentagons_list and \
                                                        permuted_vertices_palyndrome not in pentagons_list:
                                                        pentagons_list += [permuted_vertices]
        return pentagons_list

    '''
    This strategy below seems to be the more general one.
    For sure it can be used in the three cases above
    Possibly it can be put in a lambda that works for any polygon
    '''
    def FindHexagons(self):
        hexagons_list = []
        for vertex_i in self.vertices:
            for vertex_ii in self.adj_lists[vertex_i]:
                for vertex_iii in self.adj_lists[vertex_ii]:
                    not_back_neighbors_iii = \
                        AllTrue([vertex not in [vertex_i] 
                                 for vertex in self.adj_lists[vertex_iii]])
                    not_back_close_iii = \
                        AllTrue([vertex != vertex_iii for vertex in [vertex_i]])
                    if not_back_neighbors_iii and not_back_close_iii:
                        
                        for vertex_iv in self.adj_lists[vertex_iii]:
                            not_back_neighbors_iv = \
                                AllTrue([vertex not in [vertex_i, vertex_ii] 
                                         for vertex in self.adj_lists[vertex_iv]])
                            not_back_close_iv = \
                                AllTrue([vertex != vertex_iv for vertex in [vertex_i, vertex_ii]])
                            if not_back_neighbors_iv and not_back_close_iv:
                                
                                for vertex_v in self.adj_lists[vertex_iv]:
                                    not_back_neighbors_v = \
                                        AllTrue([vertex not in [vertex_i, vertex_ii, vertex_iii] 
                                                 for vertex in self.adj_lists[vertex_v]])
                                    not_back_close_v = \
                                        AllTrue([vertex != vertex_v 
                                                 for vertex in [vertex_i, vertex_ii, vertex_iii]])
                                    if not_back_neighbors_v and not_back_close_v:
                                        
                                        for vertex_vi in self.adj_lists[vertex_v]:
                                            '''
                                            Closure conditions
                                            '''
                                            not_back_neighbors_vi = \
                                                AllTrue([vertex not in 
                                                         [vertex_ii, vertex_iii, vertex_iv] 
                                                         for vertex in self.adj_lists[vertex_vi]])
                                            not_back_close_vi = \
                                                AllTrue([vertex != vertex_vi 
                                                         for vertex in 
                                                         [vertex_i, vertex_ii, vertex_iii, vertex_iv]])
                                            if not_back_neighbors_vi and not_back_close_vi:
                                                
                                                for vertex_vii in self.adj_lists[vertex_vi]:
                                                    if vertex_vii == vertex_i:
                                                        vertices = \
                                                            [vertex_i, vertex_ii, 
                                                             vertex_iii, vertex_iv, 
                                                             vertex_v, vertex_vi]
                                                        
                                                        min_vertex = min(vertices)
                                                        min_index = vertices.index(min_vertex)

                                                        permuted_vertices = []
                                                        for i in range(6):
                                                            permuted_vertices += [vertices[(min_index + i) % 6]]

                                                        permuted_vertices_palyndrome = [permuted_vertices[0]]
                                                        permuted_vertices_palyndrome += reversed(permuted_vertices[1:])

                                                        if permuted_vertices not in hexagons_list and \
                                                            permuted_vertices_palyndrome not in hexagons_list:
                                                            hexagons_list += [permuted_vertices]
        return hexagons_list

    def FindPolygons(self, n = 3):
        if n == 3:
            return self.FindTriangles()
        if n == 4:
            return self.FindSquares()
        if n == 5:
            return self.FindPentagons()
        if n == 6:
            return self.FindHexagons()
        
    def FindPolygonsEdges(self, n = 3):
        polygons = self.FindPolygons(n)
        edges = []
        for polygon in polygons:
            swap_edges, plen = [], len(polygon)
            for i in range(plen):
                edge_0 = (polygon[i], polygon[(i + 1) % plen])
                edge_1 = (edge_0[1], edge_0[0])
                swap_edges += [edge_0 if edge_0 in self.edges else edge_1]
            edges += [swap_edges]
        return edges
    
    def FindPolygonsLabels(self, n = 3):
        polygons_edges = self.FindPolygonsEdges(n)
        labels = []
        for edges in polygons_edges:
            labels_swap = []
            for edge in edges:
                labels_swap += [self.labels[edge]]
            labels += [labels_swap]
        return labels
    
    def AreEdgesContiguous(self, edges):
        edges_swap = edges.copy()    
        no_merge_flag = False

        while len(edges_swap) > 1:        
            N = len(edges_swap)
            break_for_i = False
            for i in range(N - 1):
                for j in range(i + 1, N):
                    if edges_swap[i][0] in edges_swap[j]:
                        edge_i, edge_j = edges_swap[i], edges_swap[j]
                        break_for_i = True
                        new_edge = \
                            (edge_i[1], edge_j[0] if edge_j[0] != edge_i[0] else edge_j[1])
                        ## print(edge_i, edge_j, new_edge, "A")

                        edges_swap += [new_edge]
                        edges_swap.remove(edge_i)
                        edges_swap.remove(edge_j)
                        ## print(edges_swap)

                        break
                    elif edges_swap[i][1] in edges_swap[j]:
                        edge_i, edge_j = edges_swap[i], edges_swap[j]
                        break_for_i = True
                        new_edge = \
                            (edge_i[0], edge_j[0] if edge_j[0] != edge_i[1] else edge_j[1])
                        ## print(edge_i, edge_j, new_edge, "B")

                        edges_swap += [new_edge]
                        edges_swap.remove(edge_i)
                        edges_swap.remove(edge_j)
                        ## print(edges_swap)

                        break

                if break_for_i:
                    break

            if not break_for_i:
                no_merge_flag = True
                break

        return not no_merge_flag
    
    
    '''
    We need to have a notion of boundary: it makes life easier
    '''
    def SetBoundaryVertices(self):
        self.boundary_vertices = [False] * len(self.vertices)
        for edge in self.edges:
            if self.IsEdgeBoundary(edge):
                for vertex in edge:
                    self.boundary_vertices[vertex] = True      
    
    def IsEdgeBoundary(self, edge):
        if edge not in self.edges:
            raise Exception("Edge is not in present")
            
        boundary_vertices = []
        for vertex in edge:
            for neighbor in self.adj_lists[vertex]:
                if neighbor not in edge:
                    boundary_vertices += [neighbor]
        
        determinants = []
        for vertex in boundary_vertices:
            v_triangle = edge + (vertex,)
            triangle = [self.coords[v] for v in v_triangle]
            '''
            compute determinant
            '''
            det = TriDet(triangle)
            if det != None:
                determinants += [1 if det > 0 else 0]
            
        if sum(determinants) == 0 or \
            sum(determinants) == len(determinants):
            return True
        else:
            return False
            
    '''
    After preparing the spin network we need to call this method
    '''
    def SetLabelsEdges(self):
        del self.labels_edges
        self.labels_edges = {}
        for edge in self.labels:
            self.labels_edges[self.labels[edge]] = edge
    '''
    This method needs to be called after recoupling operations:
    the labels are the `invariant` while the vertices indices,
    i.e. the edges are not.
    - Need to call it before RealignVertices
    '''
    def SetEdgesLabels(self):
        del self.labels
        self.labels = {}
        for label in self.labels_edges:
            self.labels[self.labels_edges[label]] = label
        
    def AmITetrahedron(self):
        _am_i_tetrahedron = True
        '''
        Checking the number of vertices
        '''
        if len(self.vertices) != 4:
            _am_i_tetrahedron = False
                            
        '''
        Checking pointers excluding self-references
        '''
        _pointers_count = [0] * len(self.vertices)
        for vertex in self.vertices:
            for neighbor in self.adj_lists[vertex]:
                if neighbor != vertex:
                    _pointers_count[neighbor] += 1
        
        _check_count = [count == 3 for count in _pointers_count]
        if False in _check_count:
            _am_i_tetrahedron = False
            
        return _am_i_tetrahedron
            
    def PruneHalfEdges(self):
        del self.hedges
        del self.vvertices
        self.hedges, self.vvertices = [], []        
        
    def GetAdjacencyList(self):
        if hasattr(self, 'adj_list'):
            del self.adj_lists
        _adj_swap = [[] for i in range(len(self.vertices))]
        for edge in self.edges:
            _adj_swap[edge[0]] += [edge[1]]
            _adj_swap[edge[1]] += [edge[0]]
            
        for vertex in self.vertices:
            _adj_swap[vertex].sort()
                    
        self.adj_lists = _adj_swap
        return _adj_swap
        
    def GetDegrees(self):
        if not hasattr(self, 'adj_lists'):
            self.GetAdjacencyList()
            
        _degrees = []
        for vertex in self.vertices:
            _degrees += [len(self.adj_lists[vertex])]
            
        self.degrees = _degrees
        return _degrees
    
    '''
    - Still this version is prone to errors if the intermediate
    objects are not reset to a contiguous counting: is this true?
    '''
    def ShiftVertices(self, offset):
        '''
        - vvertices, coords, vcoords are unchanged
        '''
        _swap_vertices = []
        _swap_edges, _swap_hedges = [], []
        
        for vertex in self.vertices:
            _swap_vertices += [vertex + offset]
        self.vertices = _swap_vertices
        
        for edge in self.edges:
            _swap_edges += [(edge[0] + offset, edge[1] + offset)]
        self.edges = _swap_edges
        
        for hedge in self.hedges:
            _swap_hedges += [(hedge[0] + offset, hedge[1])]
        self.hedges = _swap_hedges
        '''
        need to change the labels as well
        '''
        _tmp_labels = {}
        for edge in self.labels:
            _tmp_labels[(edge[0] + offset, edge[1] + offset)] = \
                self.labels[edge]
        del self.labels
        self.labels = _tmp_labels
        
    '''
    RealignVertices: get a contiguous vertices numbering
    '''
    def RealignVertices(self):
        _n_vertices = len(self.vertices)
        _vertex_max = max(self.vertices)
        _vertices_indices = [-1] * (_vertex_max + 1)
        _coords_swap = [None for i in range(_vertex_max + 1)]
        
        for vertex in self.vertices:
            _vertices_indices[vertex] = self.vertices.index(vertex)

        for vertex_i, vertex in enumerate(self.vertices):
            _coords_swap[vertex_i] = self.coords[vertex]
        del self.coords
        self.coords = _coords_swap[:_n_vertices].copy()
        
        if hasattr(self, 'boundary_vertices'):
            _boundary_vertices_swap = [False] * (_vertex_max + 1)
            for vertex_i, vertex in enumerate(self.vertices):
                _boundary_vertices_swap[vertex_i] = self.boundary_vertices[vertex]
            del self.boundary_vertices
            self.boundary_vertices = _boundary_vertices_swap[:_n_vertices].copy()

        '''
        update labels
        '''
        _tmp_labels = {}
        for edge_i, edge in enumerate(self.edges):
            _new_vertex_0 = _vertices_indices[self.edges[edge_i][0]]
            _new_vertex_1 = _vertices_indices[self.edges[edge_i][1]]
            _new_edge = (_new_vertex_0, _new_vertex_1)
            _tmp_labels[_new_edge] = self.labels[edge]
                        
        del self.labels
        self.labels = _tmp_labels
        
        '''
        substitute new edges
        '''
        for edge_i, edge in enumerate(self.edges):
            _new_vertex_0 = _vertices_indices[self.edges[edge_i][0]]
            _new_vertex_1 = _vertices_indices[self.edges[edge_i][1]]
            self.edges[edge_i] = (_new_vertex_0, _new_vertex_1)
                    
        self.vertices = list(range(_n_vertices))

        '''
        reassign the adjacency list
        '''
        self.GetAdjacencyList()
                
    def PlotPlanarGraph(self, figsize=None, f_list=[], args_lists=[{}], 
                        edges_lw=0.5, edges_lt='-.', edges_col='black', point_size=1):
        if self.planar_flag:
            plt.figure(figsize=figsize)
            plt.gca().set_aspect('equal')
            plt.axis('off')
            
            for edge in self.edges:
                _coords_x = [self.coords[vertex][0] for vertex in edge]
                _coords_y = [self.coords[vertex][1] for vertex in edge]
                plt.plot(_coords_x, _coords_y, edges_lt, color=edges_col, 
                         linewidth=edges_lw)

            for hedge in self.hedges:
                _coords_x = [self.coords[hedge[0]][0], self.vcoords[hedge[1]][0]]
                _coords_y = [self.coords[hedge[0]][1], self.vcoords[hedge[1]][1]]
                plt.plot(_coords_x, _coords_y, color='black')
                
            for vertex in self.vertices:
                plt.plot([self.coords[vertex][0]], 
                         [self.coords[vertex][1]], 
                         'o', color='black', markersize=point_size)

            for vvertex in self.vvertices:
                plt.plot([self.vcoords[vvertex][0]], 
                         [self.vcoords[vvertex][1]], 
                         'x', color='black')
                
            for f_i, f in enumerate(f_list):
                f(self, plt, **args_lists[f_i])
                
            plt.show()
            plt.close()
        else:
            raise Exception("The graph is not planar!")
            
    '''
    labels is a dict
    '''
    def InitLabels(self, labels):
        for key in labels:
            self.labels[key] = labels[key]

    '''
    hlabels is a dict
    '''
    def InitHLabels(self, hlabels):
        for key in hlabels:
            self.hlabels[key] = hlabels[key]

    def InitEdges(self, edges):
        for edge in edges:
            self.edges += [edge]
            
    def InitHEdges(self, hedges):
        for hedge in hedges:
            self.hedges += [hedge]            
            
    def InitVertices(self, vertices):
        for vertex in vertices:
            self.vertices += [vertex]

    def InitVvertices(self, vvertices):
        for vvertex in vvertices:
            self.vvertices += [vvertex]
            
    def InitCoords(self, coords):
        for coord in coords:
            self.coords += [coord]            

    def InitVvertices(self, vvertices):
        for vvertex in vvertices:
            self.vvertices += [vvertex]
            
    def InitVCoords(self, vcoords):
        for key in vcoords:
            self.vcoords[key] = vcoords[key]
