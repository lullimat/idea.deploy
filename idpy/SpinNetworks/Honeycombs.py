__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2023 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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

import numpy as np
import sympy as sp

from collections import defaultdict
from functools import reduce
import math

from idpy.Utils.SimpleTiming import SimpleTiming
from idpy.Utils.Statements import AllTrue

from idpy.Utils.Combinatorics import GetSinglePermutations, Multinomial
from idpy.Utils.Combinatorics import FindSumCombinationsTuples, GetUniquePermutations

from idpy.Utils.ManageData import ManageData
from idpy.Utils.DictHandle import RunThroughDict, Edit_ListToNPArray
from idpy.Utils.Geometry import GetLen2Pos, GetDiffPos, PosFromIndex

from idpy.Utils.Graphs import Graph
from idpy.SpinNetworks.Evaluations import FromLabelToStrSym, Evaluation

'''
SKN Section
'''
class SKN(Graph):
    delta_x_graph, delta_y_graph = np.sqrt(3)/2, 1 + 1 / 2
    vertical_length, dx_length, dy_length = 1, np.sqrt(3) / 4, 1 / 4
    
    base_labels = ['e']
    base_hlabels = ['a', 'b', 'c', 'd']
    base_vertices = [0, 1]
    base_vvertices = [0, 1, 2, 3]
    
    int_coords_text_offset = \
        [(+ 0.15, + 0.15), 
         (+ 0.15, - 0.15)]    

    int_vcoords_text_offset = \
        [(+ 0.1, + 0.3), 
         (- 0.3, + 0.3), 
         (+ 0.1, - 0.3), 
         (- 0.3, - 0.3)]    
        
    def __init__(self, k, n):
        if k % 2 == 0 and n == 0:
            raise Exception("'n' cannot be zero if 'k' is even!")
        self.k, self.n = k, n
        self.coords_int, self.vcoords_int = [], {}
        Graph.__init__(self)
        self.InitVars()
        self.planar_flag = True        

    '''
    This operation should actually be done in the class Graph
    '''
    def InitVars(self, offset=0):        
        self.SetVertices(offset=offset)
        self.SetVvertices()        
        self.SetEdges()
        self.SetHEdges()
        self.SetLabels()
        self.SetHLabels()
        self.SetCoords()
        
    '''
    When adding two SKN we need to actually add two Honeycomb that are initialized from the
    two SKN: like this all the summation rules will be encoded in Honeycomb
    '''    
    def __add__(self, other):
        return Honeycomb(self) + Honeycomb(other)
    
    def __iadd__(self, other):
        return self.__add__(other)
        
    def SetVertices(self, offset=0):
        for vertex in self.base_vertices:
            self.vertices += [vertex + offset]

    def SetVvertices(self):
        for vvertex in self.base_vvertices:
            self.vvertices += [((self.k, self.n), vvertex)]
            
    '''
    - Labels and edges are assumed to be in the same order: this is enforced by the order of self.base_labels
    '''
    def SetLabels(self):
        for i, label in enumerate(self.base_labels):
            self.labels[self.edges[i]] = ((self.k, self.n), label)
            
    def SetHLabels(self):
        for i, label in enumerate(self.base_hlabels):
            self.hlabels[self.vvertices[i]] = ((self.k, self.n), label)
            
    def SetEdges(self):
        self.edges = \
            [(self.vertices[0], self.vertices[1])]

    def SetHEdges(self):
        self.hedges = \
            [(self.vertices[0], self.vvertices[0]), 
             (self.vertices[0], self.vvertices[1]),
             (self.vertices[1], self.vvertices[2]), 
             (self.vertices[1], self.vvertices[3])]
        
    def SetCoords(self):
        ##for vertex in self.base_vertices:
        parity_offset_x = \
            0 if self.k % 2 == 1 else \
            (-2 * self.dx_length if self.n > 0 else 2 * self.dx_length)
        
        _x2, _y2 = \
            2 * self.n * self.delta_x_graph + parity_offset_x, \
            self.k * self.delta_y_graph
        
        self.coords = \
            [(_x2, _y2), 
             (_x2, _y2 + self.vertical_length)]
        
        self.vcoords = \
            {self.vvertices[0]: (_x2 - self.dx_length, _y2 - self.dy_length), 
             self.vvertices[1]: (_x2 + self.dx_length, _y2 - self.dy_length), 
             self.vvertices[2]: (_x2 + self.dx_length, _y2 + self.vertical_length + self.dy_length), 
             self.vvertices[3]: (_x2 - self.dx_length, _y2 + self.vertical_length + self.dy_length)}


"""
    half edges labels
               [3]     [2]
                 \\    /
                  \\  /
                   \\/
                   /\\
                  /  \\
                 /    \\
               [0]     [1]
"""
'''
LUT: XOR best way
'''
_xors_LUT = [0, 1, 1, 0]
def SKNHEdgesMatch(hedge):
    _which_kn, _which_hedge = hedge[0], hedge[1]

    _opposite_hedge = (_which_hedge + 2) % 4
    _delta_hedge = _which_hedge - _opposite_hedge
    _k_shift = -1 if _delta_hedge < 0 else 1
    _opposite_k = _which_kn[0] + _k_shift    
    
    _k_parity = _which_kn[0] % 2
    _n = _which_kn[1]
    if _k_parity == 1:
        _n_shift = \
            (1 if _n > 0 else 0) \
            if _xors_LUT[_which_hedge] == 1 else \
            (0 if _n > 0 else -1)
    else:
        _n_shift = \
            (0 if _n > 0 else +1) \
            if _xors_LUT[_which_hedge] == 1 else \
            (-1 if _n > 0 else 0)
                
    _opposite_n = _which_kn[1] + _n_shift
        
    _target_hedge = ((_opposite_k, _opposite_n), _opposite_hedge)
    return _target_hedge

def RulesHLabelsMerge(label_0, label_1):
    _k_0, _n_0 = label_0[0][0], label_0[0][1]
    _k_1, _n_1 = label_1[0][0], label_1[0][1]
    
    if _k_0 < _k_1:
        return label_0
    else:
        return label_1
    
def RulesLabelsMerge(label_0, label_1):
    _k_0, _n_0 = label_0[0][0], label_0[0][1]
    _k_1, _n_1 = label_1[0][0], label_1[0][1]
        
    _l0, _l1 = label_0[1], label_1[1]
    
    if _n_0 >= 0 and _n_1 >= 0:
        _rank_list = ['e', 'c', 'd']
        _rank_0, _rank_1 = _rank_list.index(_l0), _rank_list.index(_l1)
        
    if _n_0 <= 0 and _n_1 <= 0:
        _rank_list = ['e', 'd', 'c']
        _rank_0, _rank_1 = _rank_list.index(_l0), _rank_list.index(_l1)            

    '''the tip'''
    if _n_0 * _n_1 < 0:
        _rank_list = ['c', 'd', 'e']
        _rank_0, _rank_1 = _rank_list.index(_l0), _rank_list.index(_l1)
        
    _new_k = _k_0 if _rank_0 > _rank_1 else _k_1
    _new_n = _n_0 if _rank_0 > _rank_1 else _n_1
    _new_label = _l0 if _rank_0 > _rank_1 else _l1          
        
    return ((_new_k, _new_n), _new_label)

'''
Honeycomb section
'''
def TotalNumberOfSequencesLambda(extrema_list):
    return reduce(lambda x, y: x * y, [extrema[0] - extrema[1] + 1 for extrema in extrema_list])

def TotalNumberOfSequences(extrema_list):
    return math.prod([extrema[0] - extrema[1] + 1 for extrema in extrema_list])

repeated_range = lambda m, M, reps: [[n] * reps for n in range(m, M)]

def StridesSequences(extrema_list):
    ranges_sizes = [extrema[0] - extrema[1] + 1 for extrema in extrema_list]
    n_ranges = len(ranges_sizes)
    strides = \
        [1] + [math.prod(ranges_sizes[:i]) 
               for i in range(1, n_ranges)]
    rstrides = \
        [math.prod(ranges_sizes[i:]) 
         for i in range(1, n_ranges)] + [1]

    return strides, rstrides

def FromExtremaToAllSequences(extrema_list):
    strides, rstrides = StridesSequences(extrema_list)
        
    sequences = []
    for i, elem in enumerate(extrema_list):
        sequences += [
            sum(repeated_range(elem[1], elem[0] + 1, rstrides[i]) * strides[i])
        ]
    return list(zip(*sequences))

def FromExtremaToSequence(extrema_list, i=0):
    N, n_extrema = TotalNumberOfSequences(extrema_list), len(extrema_list)
    if i >= N:
        return None
        
    lengths = [extrema[0] - extrema[1] + 1 for extrema in extrema_list]
    strides = [math.prod(lengths[-j-1:]) for j in range(n_extrema - 1)]
    
    coords = PosFromIndex(i, strides)
    sequence = [extrema_list[j][1] + coords[-j-1] for j in range(n_extrema)]

    return sequence

def UnrollBranches(tree, level, max_level, debug_flag=False):
    ## tree can be None only for the 3 x 3: there is only one sum
    if tree is None or len(tree) == 0:
        return None

    if debug_flag:
        print("tree:", tree)

    '''
    - We need to distiguish at this level wether or not 'tree' contains
    a list of 'tuples' or a list of integers
    - This is consistent since we know the two types cannot coexist
    - Furthermore, the distinction sets aside either a single node case (3x3)
        or the last level of the tree
    '''

    is_tuple_tree = AllTrue([type(branch) == tuple for branch in tree])
    is_int_tree = AllTrue([type(branch) == int for branch in tree])

    unrolled_branches = []
    
    if is_tuple_tree:
        if debug_flag:
            print("Tuples!")
        unrolled_branches_swap = []
        for branch in tree:
            if debug_flag:
                print("branch:", branch)
            unrolled_branches_swap += [[branch[0]]]
            
            lower_levels = \
                UnrollBranches(branch[1], level + 1, max_level, debug_flag=debug_flag)
            if lower_levels is not None:
                if debug_flag:
                    print("lower_levels:", lower_levels)

                for ll_list in lower_levels:
                     unrolled_branches += [unrolled_branches_swap[-1] + ll_list]
    
                if debug_flag:
                    print("unrolled_branches_tmp:", unrolled_branches)
            
    if is_int_tree:
        if debug_flag:
            print("Ints!")
        for branch in tree:                
            unrolled_branches += [[branch]]
    
    if level > 0:
        return unrolled_branches
    else:
        pruned_branches = []
        for branch in unrolled_branches:
            if len(branch) == max_level + 1:
                pruned_branches += [branch]
        return pruned_branches

def FromLabelToStrLatex(label):
    return '$' + FromLabelToStrSym(label) + '$'

class Honeycomb(Graph, Evaluation):
    def __init__(self, skn=None):
        Graph.__init__(self)
        Evaluation.__init__(self)
        self.planar_flag = True
        self.N = None
        self.N_init = None
        self.polygons_found = False
        self.lexicographic_polygons = None
        self.evaluated_tetra = False
        self.cycles_edges, self.cycles_lex_poly = None, None
        
        if skn is not None:
            self.InitLabels(skn.labels)
            self.InitHLabels(skn.hlabels)            
            self.InitEdges(skn.edges)
            self.InitHEdges(skn.hedges)            
            self.InitVertices(skn.vertices)
            self.InitVvertices(skn.vvertices)            
            self.InitCoords(skn.coords)
            self.InitVCoords(skn.vcoords)
            
    def SetN(self, N):
        self.N = N

    def SetNInit(self, N):
        self.N_init = N
            
    def InitColors(self):
        self.colors = {}
        for edge in self.labels:
            self.colors[self.labels[edge]] = [0]
        
    def FindNeighboringPolygons(self, index):
        x, y = index % self.N, index // self.N
        neighbors_list = []

        px = index + 1
        xpx, ypx = px % self.N, px // self.N
        if xpx - x > 0:
            ## print("px", px)
            neighbors_list += [px]

        mx = index - 1
        xmx, ymx = (mx + self.N) % self.N, mx // self.N
        if xmx - x < 0:
            ## print("mx", mx)
            neighbors_list += [mx]

        py = index + self.N
        xpy, ypy = py % self.N, py // self.N
        if ypy < self.N:
            ## print("py", py)        
            neighbors_list += [py]

        my = index - self.N
        xmy, ymy = (my + self.N) % self.N, my // self.N
        if ymy >= 0:
            ## print("my", my, (xmy, ymy))        
            neighbors_list += [my]

        pymx = index + self.N - 1
        xpymx, ypymx = pymx % self.N, pymx // self.N
        if xpymx - x < 0 and ypymx < self.N:
            ## print("pymx", pymx)        
            neighbors_list += [pymx]

        mypx = index - self.N + 1
        xmypx, ymypx = (mypx + self.N) % self.N, mypx // self.N
        if xmypx - x > 0 and ymypx >= 0:
            ## print("mypx", mypx)        
            neighbors_list += [mypx]

        return neighbors_list
    
    def FindLabelsInCycle(self):
        return list(dict(filter(lambda x: x[1][0] > 0, self.colors.items())).keys())

    def FindAllHCPolygons(self):
        if not self.polygons_found and not self.AmITetrahedron():
            self.triangles = self.FindPolygonsLabels(n=3)
            self.squares = self.FindPolygonsLabels(n=4)
            if self.N > 2:
                self.hexagons = self.FindPolygonsLabels(n=6)
                self.pentagons = self.FindPolygonsLabels(n=5)
            elif self.N == 2:
                self.squares.pop(0)
            self.polygons_found = True

        if not self.polygons_found and self.AmITetrahedron():
            self.triangles = self.FindPolygonsLabels(n=3)
            self.triangles.pop(1)
            self.polygons_found = True 
            
    def SetLexicographicPolygons(self):
        self.FindAllHCPolygons()
        
        if self.lexicographic_polygons is None:
            if self.AmITetrahedron():
                self.lexicographic_polygons = self.triangles
            else:
                self.lexicographic_polygons = []
                '''
                Bottom line
                '''
                self.lexicographic_polygons += [self.triangles[0]]
                self.lexicographic_polygons += \
                    [self.pentagons[i] for i in range(1, 2 * self.N - 3, 2)]
                self.lexicographic_polygons += [self.squares[1]]

                '''
                Bulk
                '''
                a = list(range(2, self.N - 1))
                b = a.copy()
                b.reverse()
                hex_x_strides = np.array(a + b)

                hex_x_strides_y = np.zeros([self.N - 2, self.N - 2], dtype=np.int32)

                for y_i in range(self.N - 2):
                    for x_i, x in enumerate(hex_x_strides[0 + y_i: self.N - 3 + y_i]):
                        hex_x_strides_y[y_i, x_i + 1] = x

                hex_y_strides = np.arange(self.N - 2)

                for y_i in range(len(hex_y_strides)):
                    self.lexicographic_polygons += [self.pentagons[2 * (y_i)]] # First of the first half of the even ones

                    h_indices = []
                    for x_i in range(len(hex_x_strides_y[y_i, :])):
                        h_indices += [np.sum(hex_x_strides_y[y_i, :x_i + 1]) +  np.sum(hex_y_strides[:y_i + 1])]
                    ##print(h_indices)

                    self.lexicographic_polygons += \
                        [self.hexagons[i] for i in h_indices]
                    self.lexicographic_polygons += \
                        [self.pentagons[2 * self.N - 3 + 2 * (y_i)]] # First of the second half of the odd ones

                '''
                Last line
                '''
                self.lexicographic_polygons += \
                    [self.squares[0]]
                self.lexicographic_polygons += \
                    [self.pentagons[i] 
                     for i in range(2 * self.N - 4, 4 * self.N - 8, 2)]
                self.lexicographic_polygons += \
                    [self.triangles[1]]
        
    def SetMaximalCycle(self):
        self.SetLexicographicPolygons()
        self.SetLoopColorsMod2(self.lexicographic_polygons)
        
        self.polygons_in_cycle = \
            list(range(N)) + \
            [N - 1 + y * N for y in range(1, N)] + \
            [N - 1 - x + (N - 1) * N for x in range(1, N)] + \
            [0 + (N - 1 - y) * N for y in range(1, N - 1)]
        
    def SetLabelsInCycle(self):
        self.labels_in_cycle = self.FindLabelsInCycle()
        
        
    def FindCycleLabelInPolygon(self, polygon_index):
        belongs, does_not_belong = [], []
        for label in self.lexicographic_polygons[polygon_index]:
            if label in self.labels_in_cycle:
                belongs += [label]
            else:
                does_not_belong += [label]

        return belongs, does_not_belong
    
    def __iadd__(self, other):
        return self.__add__(other)
        
    def __add__(self, other):
        _other_offset = len(self.vertices)
        other.ShiftVertices(offset=_other_offset)
        '''
        self.labels, self.hlabels = {}, {}
        self.edges, self.hedges = [], []
        self.vertices, self.coords = [], []
        self.vvertices, self.vcoords = [], {}        
        '''
        _new_hnc = Honeycomb()
        
        '''
        '''
        _new_labels = {**self.labels, **other.labels}
        _new_hnc.InitLabels(_new_labels)
        
        _new_hlabels = {**self.hlabels, **other.hlabels}
        _new_hnc.InitHLabels(_new_hlabels)
        
        _new_edges = self.edges + other.edges
        _new_hnc.InitEdges(_new_edges)
        
        _new_hedges = self.hedges + other.hedges
        _new_hnc.InitHEdges(_new_hedges)
        
        _new_vertices = self.vertices + other.vertices
        _new_hnc.InitVertices(_new_vertices)
        
        _new_vvertices = self.vvertices + other.vvertices
        _new_hnc.InitVvertices(_new_vvertices)
        
        _new_coords = self.coords + other.coords
        _new_hnc.InitCoords(_new_coords)
        
        _new_vcoords = {**self.vcoords, **other.vcoords}
        _new_hnc.InitVCoords(_new_vcoords)
                
        return _new_hnc
        
    def ClassifyHalfEdges(self):
        self.free_hedges = []
        '''
        list of (i, j) tuples of indices of self.hedges which is a list
        '''
        self.coupled_hedges = []
        self.coupled_hedges_indices = []

        _swap_hedges = [hedge[1] for hedge in self.hedges]
        for i, hedge_i in enumerate(_swap_hedges):
            _target_hedge = SKNHEdgesMatch(hedge_i)
                        
            if _target_hedge in _swap_hedges:                
                j = _swap_hedges.index(_target_hedge)
                    
                if (i, j) not in self.coupled_hedges and \
                    (j, i) not in self.coupled_hedges:
                    self.coupled_hedges += [(i, j)]
        
        _unrolled_coupled = \
            [elem for couple in self.coupled_hedges for elem in couple]
        self.free_hedges = list(range(len(self.hedges)))
        
        _unrolled_coupled.sort(reverse=True)
        self.coupled_hedges_indices = _unrolled_coupled

        for elem in _unrolled_coupled:
            self.free_hedges.pop(elem)
                
    def MergeHalfEdges(self):
        _added_edges, _added_labels = [], []
        for hedge in self.coupled_hedges:
            _vertex_0 = self.hedges[hedge[0]][0]
            _vertex_1 = self.hedges[hedge[1]][0]
            
            _hedge_0 = self.hedges[hedge[0]][1]
            _hedge_1 = self.hedges[hedge[1]][1]
            _hlabel_0 = self.hlabels[_hedge_0]
            _hlabel_1 = self.hlabels[_hedge_1]            
            
            _new_label = RulesHLabelsMerge(_hlabel_0, _hlabel_1)
            _added_labels += [_new_label]
            _added_edges += [(_vertex_0, _vertex_1)]

        '''
        Adding new edges
        '''
        self.edges += _added_edges
        
        for i, edge in enumerate(_added_edges):
            self.labels[edge] = _added_labels[i]
        '''
        Deleting merged half-edges: indices are in reverse order
        '''
        for index in self.coupled_hedges_indices:
            self.hedges.pop(index)
            self.vvertices.pop(index)
            
    '''
    For pruning the degree one vertices we only need
    to pop: vertices, edges
    '''
    def PruneDegreeOne(self):
        self.GetDegrees()
        _to_be_popped = []
        for i, degree in enumerate(self.degrees):
            if degree == 1:
                _to_be_popped += [i]
            
        '''
        Sorted by construction
        '''
        _to_be_popped.reverse()
        for i in _to_be_popped:
            self.vertices.pop(i)
            
        _to_be_popped_edges = []
        for index in _to_be_popped:
            for i, edge in enumerate(self.edges):
                if edge[0] == index or edge[1] == index:
                    _to_be_popped_edges += [i]
                    
        _to_be_popped_edges.sort(reverse=True)
        for i in _to_be_popped_edges:
            self.edges.pop(i)
            ## self.labels.pop(i)
        '''
        cuda-style: clean after ourselves
        '''
        self.RealignVertices()
        
    def PruneDegreeTwo(self):
        self.GetAdjacencyList()
        _degrees = np.array(self.GetDegrees())
        _degrees_two_count = len(list(np.where(_degrees == 2)[0]))

        for k in range(_degrees_two_count):
            self.PruneSingleDegreeTwo()
            
        self.GetAdjacencyList()
            
    def PruneSingleDegreeTwo(self):
        '''
        Count first how many vertices of degree two: use the limit in the for loop
        '''
        self.GetAdjacencyList()
        _degrees = np.array(self.GetDegrees())
        _degrees_two_count = 0
        for degree in _degrees:
            _degrees_two_count += 1 if degree == 2 else 0
        _where_two = list(np.where(_degrees == 2)[0])
        _where_two.sort(reverse=True)
        
        _target_vertex = _where_two[0]
        self.GetAdjacencyList()
        _degrees_swap = self.GetDegrees()

        _new_couple = self.adj_lists[_target_vertex]
        _new_edge = (_new_couple[0], _new_couple[1])
        self.edges += [_new_edge]
        
        '''
        Pimple Popping
        '''
        self.vertices.pop(_target_vertex)
        _edges_indices_pop, _edges_target = [], []
        for edge_i, edge in enumerate(self.edges):
            if edge[0] == _target_vertex or \
                edge[1] == _target_vertex:
                _edges_indices_pop += [edge_i]
                _edges_target += [edge]
                
        '''
        Reassign the labels and del
        '''
        _old_label_0, _old_label_1 = \
            self.labels[_edges_target[0]], self.labels[_edges_target[1]]
        
        _new_label = RulesLabelsMerge(_old_label_0, _old_label_1)
        self.labels[_new_edge] = _new_label
        
        del self.labels[_edges_target[0]], self.labels[_edges_target[1]]
                
        '''
        Popping edges
        '''
        _edges_indices_pop.sort(reverse=True)
        for edge_i in _edges_indices_pop:
            self.edges.pop(edge_i)     

        self.RealignVertices()

    def LemmaZappalaTip(self):
        _triangles = self.FindTriangles()
        '''
        Get Triangles Labels
        '''
        _t_labels = []
        for triangle in _triangles:
            _swap_labels = []
            for i in range(3):
                vertex_i = triangle[i]
                vertex_j = triangle[(i + 1) % 3]

                edge_0, edge_1 = \
                    (vertex_i, vertex_j), (vertex_j, vertex_i)

                label = None
                if edge_0 in self.labels:
                    label = self.labels[edge_0]
                if edge_1 in self.labels:
                    label = self.labels[edge_1]

                k, n, lett = label[0][0], label[0][1], label[1]
                _swap_labels += [(k, n, lett)]
            
            _t_labels += [_swap_labels]
        '''
        Get the highest triangle
        '''
        _k_lists = [[label[0] for label in tri] for tri in _t_labels]
        _k_max_lists = [max(k_l) for k_l in _k_lists]
        _k_max = max(_k_max_lists)
        _k_max_index = _k_max_lists.index(_k_max)
        '''
        Find edge 'F', i.e. the one with '0' for the 'n' coordinate
        1. collect all the "outside" labels
        2. find the edge with 'n = 0'
        '''
        _tip_triangle = _triangles[_k_max_index]
        _ext_vertices = []
        for vertex in _tip_triangle:
            for vertex_i in self.adj_lists[vertex]:
                if vertex_i not in _tip_triangle:
                    _ext_vertices += [vertex_i]
        
        ## Find external edges, and put them in the right order
        _ext_edges = list(map(lambda x, y: (x, y), _tip_triangle, _ext_vertices))
        for edge_i, edge in enumerate(_ext_edges):
            edge_1 = (edge[1], edge[0])
            if edge_1 in self.edges:
                _ext_edges[edge_i] = edge_1
        
        ## Find F_EDGE, B_EDGE, C_EDGE
        ext_edges_dict = {}
        for i_edge, edge in enumerate(_ext_edges):
            label = self.labels[edge]
            n = label[0][1]
            if n == 0:
                _F_EDGE = edge
                ext_edges_dict['F'] = i_edge  
            if n > 0:
                _C_EDGE = edge
                ext_edges_dict['C'] = i_edge  
            if n < 0:
                _B_EDGE = edge
                ext_edges_dict['B'] = i_edge  
                
        ## Find A_EDGE, D_EDGE, E_EDGE
        _A_EDGE = (_tip_triangle[ext_edges_dict['F']], 
                   _tip_triangle[ext_edges_dict['B']])
        _D_EDGE = (_tip_triangle[ext_edges_dict['F']], 
                   _tip_triangle[ext_edges_dict['C']])
        _E_EDGE = (_tip_triangle[ext_edges_dict['C']], 
                   _tip_triangle[ext_edges_dict['B']])
        
        ## Correct edge order
        if _A_EDGE not in self.edges:
            _A_EDGE = (_A_EDGE[1], _A_EDGE[0])
            
        if _D_EDGE not in self.edges:
            _D_EDGE = (_D_EDGE[1], _D_EDGE[0])
        
        if _E_EDGE not in self.edges:
            _E_EDGE = (_E_EDGE[1], _E_EDGE[0])
            
        _ordered_edges = [_F_EDGE, _B_EDGE, _C_EDGE, 
                          _A_EDGE, _D_EDGE, _E_EDGE]
        
        _ordered_labels = [self.labels[_F_EDGE], self.labels[_B_EDGE], 
                           self.labels[_C_EDGE], self.labels[_A_EDGE], 
                           self.labels[_D_EDGE], self.labels[_E_EDGE]]
                          
        self.LemmaZappala(_tip_triangle, _ordered_labels)

    def LemmaZappalaUpperTriangles(self):
        _triangles = self.FindTriangles()
        '''
        Get Triangles Labels
        '''
        _t_labels, _t_edges = [], []
        for triangle in _triangles:
            _swap_labels, _swap_edges = [], []
            for i in range(3):
                vertex_i = triangle[i]
                vertex_j = triangle[(i + 1) % 3]

                edge_0, edge_1 = \
                    (vertex_i, vertex_j), (vertex_j, vertex_i)

                if edge_0 in self.edges:
                    _swap_labels += [self.labels[edge_0]]
                    _swap_edges += [edge_0]
                else:
                    _swap_labels += [self.labels[edge_1]]
                    _swap_edges += [edge_1]
            
            _t_labels += [_swap_labels]
            _t_edges += [_swap_edges]
        '''
        Pop the lowest triangle
        '''
        _k_lists = [[label[0][0] for label in tri] for tri in _t_labels]
        _k_min_lists = [min(k_l) for k_l in _k_lists]
        _k_min = min(_k_min_lists)
        _k_min_index = _k_min_lists.index(_k_min)
        
        _triangles.pop(_k_min_index)
        _t_labels.pop(_k_min_index)
        _t_edges.pop(_k_min_index)
                
        for labels in _t_labels:
            edges = [self.labels_edges[label] for label in labels]
            triangle = list(set([v for edge in edges for v in edge]))
            
            '''
            Find boundary edge
            '''
            for edge in edges:
                if self.boundary_vertices[edge[0]] and \
                    self.boundary_vertices[edge[1]]:
                    E_edge = edge
            '''
            opposite vertex
            '''
            vertex_F = None
            for vertex in triangle:
                if vertex not in E_edge:
                    vertex_F = vertex
                    break
            '''
            external vertex to vertex_F
            '''
            evertex_F = None
            for vertex in self.adj_lists[vertex_F]:
                if vertex not in E_edge:
                    evertex_F = vertex
                    break
                    
            F_edge = \
                (vertex_F, evertex_F) if (vertex_F, evertex_F) in self.edges else \
                (evertex_F, vertex_F)
            '''
            Find A_edge: depends on parity
            1. find R and L vertices: depends on y coordinate, on x if horizontal
            '''
            E0_pos, E1_pos = self.coords[E_edge[0]], self.coords[E_edge[1]]
            horizontal_flag = E0_pos[1] == E1_pos[1]
            if horizontal_flag:
                vertex_C = E_edge[0] if E0_pos[0] < E1_pos[0] else E_edge[1]
                vertex_B = E_edge[1] if E0_pos[0] < E1_pos[0] else E_edge[0]
            else:
                vertex_C = E_edge[0] if E0_pos[1] > E1_pos[1] else E_edge[1]
                vertex_B = E_edge[1] if E0_pos[1] > E1_pos[1] else E_edge[0]
            
            A_edge = \
                (vertex_B, vertex_F) if (vertex_B, vertex_F) in self.edges else (vertex_F, vertex_B)
            D_edge = \
                (vertex_C, vertex_F) if (vertex_C, vertex_F) in self.edges else (vertex_F, vertex_C)
            
            evertex_B = None
            for vertex in self.adj_lists[vertex_B]:
                if vertex not in [vertex_C, vertex_F]:
                    evertex_B = vertex
                    break
                    
            B_edge = \
                (vertex_B, evertex_B) if (vertex_B, evertex_B) in self.edges else (evertex_B, vertex_B)
            
            evertex_C = None
            for vertex in self.adj_lists[vertex_C]:
                if vertex not in [vertex_B, vertex_F]:
                    evertex_C = vertex
                    break

            C_edge = \
                (vertex_C, evertex_C) if (vertex_C, evertex_C) in self.edges else (evertex_C, vertex_C)
                    
            ordered_edges = [F_edge, B_edge, C_edge, A_edge, D_edge, E_edge]
            ordered_labels = [self.labels[edge] for edge in ordered_edges]
            
            self.LemmaZappala(triangle, ordered_labels)
       
    '''
    We need to make the recoupling functions dependant only on the labels:
    if we make explicit use of the edges, then every time we change the graph
    we need to keep track somehow, whereas, if we only make successions of 
    recoupling moves that do not affect each other for the labels that are removed
    which is reasonable
    '''
    def LemmaZappala(self, triangle, FBCADE_labels):
        '''
        1. Add new vertex with at n_vertices + 1
        2. Delete old edges and labels
        3. Delete old vertices
        3. Add new edges
        4. realign
        '''
        FBCADE_edges = [self.labels_edges[label] for label in FBCADE_labels]
        
        '''
        is the new vertex boundary?
        '''
        bnd_counter = 0
        for vertex in triangle:
            if self.boundary_vertices[vertex]:
                bnd_counter += 1
        
        n_vertices = len(self.vertices)
        self.vertices += [n_vertices]
        self.boundary_vertices += [bnd_counter > 1]
        
        _x, _y = 0, 0

        for vertex in triangle:
            _x += self.coords[vertex][0]
            _y += self.coords[vertex][1]
        _x /= 3
        _y /= 3
        
        self.coords += [(_x, _y)]
        
        '''
        keep the labels
        '''
        _F_label = self.labels[FBCADE_edges[0]]
        _B_label = self.labels[FBCADE_edges[1]]
        _C_label = self.labels[FBCADE_edges[2]]        

        _A_label = self.labels[FBCADE_edges[3]]
        _D_label = self.labels[FBCADE_edges[4]]
        _E_label = self.labels[FBCADE_edges[5]]        

        '''
        delete all labels and edges
        '''
        for edge in FBCADE_edges:
            del self.labels[edge]
        
        _edges_indices = []
        for edge in FBCADE_edges:
            _edges_indices += [self.edges.index(edge)]
        
        _edges_indices.sort(reverse=True)
        for i in _edges_indices:
            self.edges.pop(i)
        
        '''
        delete old vertices
        '''
        sorted_triangle = triangle.copy()
        sorted_triangle.sort(reverse=True)
        for i in sorted_triangle:
            self.vertices.pop(i)
        
        '''
        external vertices
        '''
        _evertex_F = \
            FBCADE_edges[0][0] \
            if FBCADE_edges[0][0] not in triangle else \
            FBCADE_edges[0][1]
            
        _evertex_B = \
            FBCADE_edges[1][0] \
            if FBCADE_edges[1][0] not in triangle else \
            FBCADE_edges[1][1]
            
        _evertex_C = \
            FBCADE_edges[2][0] \
            if FBCADE_edges[2][0] not in triangle else \
            FBCADE_edges[2][1]
        
        '''
        new edges
        '''
        self.edges += [(n_vertices, _evertex_F)]
        self.edges += [(n_vertices, _evertex_B)]        
        self.edges += [(n_vertices, _evertex_C)]
        
        self.labels_edges[_F_label] = (n_vertices, _evertex_F)
        self.labels_edges[_B_label] = (n_vertices, _evertex_B)
        self.labels_edges[_C_label] = (n_vertices, _evertex_C)        
        
        '''
        adjusting the graph structure
        '''
        self.SetEdgesLabels()
        self.RealignVertices()
        self.SetLabelsEdges()
        
        '''
        evaluation
        '''
        _6j_tuple = (_A_label, _B_label, 
                     _C_label, _D_label, 
                     _F_label, _E_label)
        self.Push6j(_6j_tuple)
        
        _th_tuple = (_A_label, _D_label, _F_label)
        self.PushTh(_th_tuple)
        
        _dn_single = (_F_label,)
        self.PushDnM1(_dn_single)
     
    def EvaluateTetrahedron(self):
        if not self.AmITetrahedron():
            raise Exception("I am not a tetrahedron!")        
        elif not self.evaluated_tetra:
            '''
            1. start from first label
            2. find the two triangles
            3. compute the tuple
            '''
            _F_edge = list(self.labels.keys())[0]
            _F_edge_list = list(_F_edge)
            _F_edge_list.sort(reverse=True)
            
            _tip_vertices = list(range(4))
            for vertex in _F_edge_list:
                _tip_vertices.pop(vertex)
                
            '''
            find associated edges
            '''
            _tip_edges = []
            for vertex in _F_edge_list:
                for edge in self.edges:
                    if vertex in edge and edge != _F_edge:
                        _tip_edges += [edge]
            
            _loop_tip_edges = [_tip_edges[0]]
            _swap_tip_edges = _tip_edges.copy()

            '''
            need to make sure about the ordering
            '''            
            while len(_loop_tip_edges) < 4:
                for j, edge_j in enumerate(_swap_tip_edges[1:]):
                    if _loop_tip_edges[-1][0] in edge_j or _loop_tip_edges[-1][1] in edge_j:
                        _loop_tip_edges += [edge_j]
                        _swap_tip_edges.pop(j)
                        continue
            '''
            find 'E' edge: the only one left
            '''
            _E_edge = None
            for edge in self.edges:
                if edge not in _loop_tip_edges + [_F_edge]:
                    _E_edge = edge
                    
                
            '''
            define the edges labels
            '''
            _F_label = self.labels[_F_edge]
            _E_label = self.labels[_E_edge]
            
            _A_label = self.labels[_loop_tip_edges[0]]
            _B_label = self.labels[_loop_tip_edges[1]]            
            _C_label = self.labels[_loop_tip_edges[2]]
            _D_label = self.labels[_loop_tip_edges[3]]
            
            '''
            evaluation
            '''            
            _tet_tuple = (_A_label, _B_label, 
                          _C_label, _D_label, 
                          _E_label, _F_label)
            self.PushTet(_tet_tuple)
            self.evaluated_tetra = True
            
    def FindBoundaryBulkLabels(self):
        main_edges = []
        for vertex in self.vertices:
            if self.boundary_vertices[vertex]:
                for neighbor in self.adj_lists[vertex]:
                    if not self.boundary_vertices[neighbor]:
                        main_edges += \
                            [(vertex, neighbor) 
                             if (vertex, neighbor) in self.edges else 
                             (neighbor, vertex)]
                        
        main_labels = [self.labels[edge] for edge in main_edges]
        return main_labels
    
    def FindRecouplingLabels(self):
        main_labels = self.FindBoundaryBulkLabels()
        '''
        cutting as a function of the height
        '''
        delenda_labels = []
        for i, label in enumerate(main_labels):
            if label[0][0] < self.N - 2:
                delenda_labels += [i]
                
        delenda_labels.sort(reverse=True)
        for i in delenda_labels:
            main_labels.pop(i)
            
        '''
        Packaging neighbors
        '''
        packaged_labels = []
        for J_label in main_labels:
            J_edge = self.labels_edges[J_label]
            J_is_boundary = [self.boundary_vertices[v] for v in J_edge]
            J_boundary_vertex = J_edge[J_is_boundary.index(True)]
            J_bulk_vertex = J_edge[J_is_boundary.index(False)]
            
            '''
            Finding bulk edges: A & D
            '''
            bulk_neighbors, bulk_neighbors_edges = [], []
            for neighbor in self.adj_lists[J_bulk_vertex]:
                if neighbor != J_boundary_vertex:
                    bulk_neighbors += [neighbor]
                    swap_edge = (J_bulk_vertex, neighbor)
                    bulk_neighbors_edges += \
                        [swap_edge 
                         if swap_edge in self.edges else 
                         (neighbor, J_bulk_vertex)]
                    
            bulk_neighbors_labels = \
                [self.labels[edge] 
                 for edge in bulk_neighbors_edges]
            
            bulk_neighbor_k = \
                [label[0][0] for label in bulk_neighbors_labels]
            bulk_neighbor_n = \
                [label[0][1] for label in bulk_neighbors_labels]            
            same_k = bulk_neighbor_k[0] == bulk_neighbor_k[1]
            same_n = bulk_neighbor_n[0] == bulk_neighbor_n[1]
            
            if not same_k:
                first = bulk_neighbor_k[0] < bulk_neighbor_k[1]
            elif not same_n:
                first = bulk_neighbor_n[0] < bulk_neighbor_n[1]                
            else:
                first = 0 if bulk_neighbors_labels[0][1] != 'e' else 1
            
            D_label = \
                bulk_neighbors_labels[0] \
                if first else \
                bulk_neighbors_labels[1]

            C_label = \
                bulk_neighbors_labels[1] \
                if first else \
                bulk_neighbors_labels[0]
            
            '''
            Finding boiundary edges: B & C
            '''
            boundary_neighbors, boundary_neighbors_edges = [], []
            ## print('J_boundary_vertex', J_boundary_vertex)
            for neighbor in self.adj_lists[J_boundary_vertex]:
                if neighbor != J_bulk_vertex:
                    boundary_neighbors += [neighbor]
                    swap_edge = (J_boundary_vertex, neighbor)
                    boundary_neighbors_edges += \
                        [swap_edge 
                         if swap_edge in self.edges else 
                         (neighbor, J_boundary_vertex)]

            boundary_neighbors_labels = \
                [self.labels[edge] 
                 for edge in boundary_neighbors_edges]

            boundary_neighbor_k = \
                [label[0][0] for label in boundary_neighbors_labels]
            bulk_neighbor_n = \
                [label[0][1] for label in bulk_neighbors_labels]            
            same_k = bulk_neighbor_k[0] == bulk_neighbor_k[1]
            same_n = bulk_neighbor_n[0] == bulk_neighbor_n[1]
            
            if not same_k:
                first = bulk_neighbor_k[0] < bulk_neighbor_k[1]
            elif not same_n:
                first = bulk_neighbor_n[0] < bulk_neighbor_n[1]                
            else:
                first = 0 if bulk_neighbors_labels[0][1] != 'e' else 1
            
            ## first_smaller_k = boundary_neighbor_k[0] < boundary_neighbor_k[1]

            A_label = \
                boundary_neighbors_labels[0] \
                if first else \
                boundary_neighbors_labels[1]

            B_label = \
                boundary_neighbors_labels[1] \
                if first else \
                boundary_neighbors_labels[0]
            
            ## print(boundary_neighbors_edges, boundary_neighbors_labels)
            labels_list = [J_label, A_label, B_label, C_label, D_label]
            packaged_labels += [labels_list]
            
        return packaged_labels        
            
    def Recoupling(self, JABCD_labels):
        JABCD_edges = \
            [self.labels_edges[label] for label in JABCD_labels]
        '''
        1. Add two new vertices
        2. Delete old two vertices and old edges
        3. create the new edges
        '''

        '''
        find new coordinates orthogonal to the previous
        J-edge
        '''
        _J_edge = JABCD_edges[0]
        bnd_counter = 0
        for vertex in _J_edge:
            if self.boundary_vertices[vertex]:
                bnd_counter += 1

        n_vertices = len(self.vertices)
        self.vertices += [n_vertices, n_vertices + 1]
        self.boundary_vertices += [bnd_counter > 0, bnd_counter > 0]
        
        _J0_x, _J0_y = \
            self.coords[_J_edge[0]][0], \
            self.coords[_J_edge[0]][1]
        _J1_x, _J1_y = \
            self.coords[_J_edge[1]][0], \
            self.coords[_J_edge[1]][1]

        _Jell = np.sqrt((_J1_y - _J0_y) ** 2 + (_J1_x - _J0_x) ** 2)
        _cm_x, _cm_y = \
            0.5 * (_J0_x + _J1_x), \
            0.5 * (_J0_y + _J1_y)

        '''
        Computing new coordinates: need to manage the case of a vertical edge
        '''
        _J_horizontal_flag = np.abs(_J0_x - _J1_x) > 1e-6
        if _J_horizontal_flag:
            _Jm = (_J1_y - _J0_y) / (_J1_x - _J0_x)
            _sign_Jm = 1 if _Jm > 0 else -1

            _Im = - 1 / _Jm
            ##print(_sign_Jm, _Im, _Jm, type(_Im))
            _Icos = - _sign_Jm / np.sqrt(1 + (_Im ** 2))
            _Isin = 1 / np.sqrt(1 + (_Jm ** 2))

            _I0_x, _I0_y = \
                _cm_x + _Jell * _Icos / 2, _cm_y + _Jell * _Isin / 2 
            _I1_x, _I1_y = \
                _cm_x - _Jell * _Icos / 2, _cm_y - _Jell * _Isin / 2
        else:
            _I0_y = _I1_y = _cm_y
            _I0_x = _cm_x - _Jell / 2
            _I1_x = _cm_x + _Jell / 2            

        _new_coords_0, _new_coords_1 = \
            (_I0_x, _I0_y), (_I1_x, _I1_y)
            
        '''
        adding vertices and new coords and label
        '''
        self.coords += [_new_coords_0]
        self.coords += [_new_coords_1]

        self.edges += [(n_vertices, n_vertices + 1)]

        _I_label = (self.labels[_J_edge][0], 'p')
        self.labels_edges[_I_label] = (n_vertices, n_vertices + 1)
        
        '''
        keep the labels
        '''
        _A_label = self.labels[JABCD_edges[1]]
        _B_label = self.labels[JABCD_edges[2]]
        _C_label = self.labels[JABCD_edges[3]]
        _D_label = self.labels[JABCD_edges[4]]
        
        '''
        Evaluation: insert the sum and the 6j
        '''
        _J_label = JABCD_labels[0]
                
        self.PushSum((_I_label,))
        self.PushSumExtrema(((_A_label, _D_label), (_B_label, _C_label)))
        self.Push6j((_A_label, _B_label, _C_label, _D_label, 
                     _I_label, _J_label))

        '''
        declare the new edges and assign to them the previous labels
        '''
        _evertex_A = \
            JABCD_edges[1][0] if JABCD_edges[1][0] not in _J_edge else JABCD_edges[1][1]
        _evertex_B = \
            JABCD_edges[2][0] if JABCD_edges[2][0] not in _J_edge else JABCD_edges[2][1]
        _evertex_C = \
            JABCD_edges[3][0] if JABCD_edges[3][0] not in _J_edge else JABCD_edges[3][1]
        _evertex_D = \
            JABCD_edges[4][0] if JABCD_edges[4][0] not in _J_edge else JABCD_edges[4][1]

        _evertex_A_coords = self.coords[_evertex_A]
        _evertex_B_coords = self.coords[_evertex_B]
        _evertex_C_coords = self.coords[_evertex_C]
        _evertex_D_coords = self.coords[_evertex_D]            

        '''
        new edges: connect with the closest
        '''
        _evertex_A_pos = self.coords[_evertex_A]
        _evertex_B_pos = self.coords[_evertex_B]        
        _evertex_C_pos = self.coords[_evertex_C]
        _evertex_D_pos = self.coords[_evertex_D]
        
        _new_vertices = [n_vertices, n_vertices + 1]
        
        _l2_0A = GetLen2Pos(GetDiffPos(_evertex_A_pos, _new_coords_0))
        _l2_1A = GetLen2Pos(GetDiffPos(_evertex_A_pos, _new_coords_1))
        _l2_list = [_l2_0A, _l2_1A]
        _l2_min = min(_l2_list)
        _which_min = _l2_list.index(_l2_min)
        _new_edge_A = (_new_vertices[_which_min], _evertex_A)

        
        _l2_0B = GetLen2Pos(GetDiffPos(_evertex_B_pos, _new_coords_0))
        _l2_1B = GetLen2Pos(GetDiffPos(_evertex_B_pos, _new_coords_1))
        _l2_list = [_l2_0B, _l2_1B]
        _l2_min = min(_l2_list)
        _which_min = _l2_list.index(_l2_min)
        _new_edge_B = (_new_vertices[_which_min], _evertex_B)
        
        _l2_0C = GetLen2Pos(GetDiffPos(_evertex_C_pos, _new_coords_0))
        _l2_1C = GetLen2Pos(GetDiffPos(_evertex_C_pos, _new_coords_1))
        _l2_list = [_l2_0C, _l2_1C]
        _l2_min = min(_l2_list)
        _which_min = _l2_list.index(_l2_min)
        _new_edge_C = (_new_vertices[_which_min], _evertex_C)
        
        _l2_0D = GetLen2Pos(GetDiffPos(_evertex_D_pos, _new_coords_0))
        _l2_1D = GetLen2Pos(GetDiffPos(_evertex_D_pos, _new_coords_1))
        _l2_list = [_l2_0D, _l2_1D]
        _l2_min = min(_l2_list)
        _which_min = _l2_list.index(_l2_min)
        _new_edge_D = (_new_vertices[_which_min], _evertex_D)

        '''
        do we have a bubble? Pop!
        '''
        _new_edges = {'A': _new_edge_A, 'B': _new_edge_B, 
                      'C': _new_edge_C, 'D': _new_edge_D}
        _new_labels = {'A': _A_label, 'B': _B_label, 
                       'C': _C_label, 'D': _D_label}        
        
        _new_edges_names = list(_new_edges.keys())
        _new_edges_list = list(_new_edges.values())
        _new_labels_names = list(_new_labels.keys())
        _new_labels_list = list(_new_labels.values())        
        
        _bubbles = []
        for i in range(len(_new_edges_list)):
            _edge_ref = _new_edges_list[i]
            for j in range(i + 1, len(_new_edges_list)):
                _edge_compare = _new_edges_list[j]
                _edge_compare_flip = (_edge_compare[1], _edge_compare[0])
                if _edge_ref == _edge_compare or _edge_ref == _edge_compare_flip:
                    _bubbles += [(i, j)]
                    
        if len(_bubbles) > 0:
            _edges_delenda, _vertices_delenda = [], []
            
            for bubble in _bubbles:
                '''
                need to find next label
                '''
                vertices_bubble = _new_edges_list[bubble[0]]
                bubble_label_0 = _new_labels_list[bubble[0]]
                bubble_label_1 = _new_labels_list[bubble[1]]

                target_neighbor = None
                for neighbor in self.adj_lists[vertices_bubble[1]]:
                    if neighbor not in JABCD_edges[0]:
                        target_neighbor = neighbor
                        break
                        
                target_edge = (target_neighbor, vertices_bubble[1])
                target_edge_flip = (vertices_bubble[1], target_neighbor)
                target_edge = \
                    target_edge if target_edge in self.edges else target_edge_flip
                target_label = self.labels[target_edge]
                '''
                Evaluation
                '''
                bubble_labels = [_new_labels_list[i] for i in bubble]
                self.PushKDelta((_I_label, target_label))
                self.PushTh((target_label, bubble_labels[0], bubble_labels[1]))
                self.PushDnM1((target_label,))
                
                '''
                graph management
                1. connect target_neighbor and vertices_bubble[0]
                2. re associate label
                3. del label
                4. del edge(vertices_bubble[1], target_neighbor)
                5. del vertices_bubble[1]
                '''
                new_edge = (vertices_bubble[0], target_neighbor)
                self.edges += [new_edge]
                self.labels_edges[target_label] = new_edge
                                
                _edges_delenda += [target_edge]
                _vertices_delenda += [vertices_bubble[1]]
            '''
            Need to reassociate the remaining non-bubble edges
            '''
            bubble_indices = []
            for bubble in _bubbles:
                for edge in bubble:
                    bubble_indices += [edge]
                
            remaining_indices = []
            for index in range(4):
                if index not in bubble_indices:
                    remaining_indices += [index]
                    
            for index in remaining_indices:
                self.edges += [_new_edges_list[index]]
                self.labels_edges[_new_labels_list[index]] = _new_edges_list[index]
            '''
            local cleanup
            '''                
            for edge in _edges_delenda:
                del self.labels[edge]
                
            _edges_indices = []
            for edge in _edges_delenda:
                _edges_indices += [self.edges.index(edge)]
            _edges_indices.sort(reverse=True)
            for i in _edges_indices:
                self.edges.pop(i)
                
            _vertices_delenda.sort(reverse=True)
            for vertex in _vertices_delenda:
                self.vertices.pop(vertex)
            
        else:        
            self.edges += [_new_edge_A]
            self.edges += [_new_edge_B]
            self.edges += [_new_edge_C]
            self.edges += [_new_edge_D]

            self.labels_edges[_A_label] = _new_edge_A        
            self.labels_edges[_B_label] = _new_edge_B       
            self.labels_edges[_C_label] = _new_edge_C        
            self.labels_edges[_D_label] = _new_edge_D
        
        '''
        Clean up: labels and edges
        '''
        for edge in JABCD_edges:
            del self.labels[edge]

        _edges_indices = []
        for edge in JABCD_edges:
            _edges_indices += [self.edges.index(edge)]

        _edges_indices.sort(reverse=True)
        for i in _edges_indices:
            self.edges.pop(i)

        '''
        delete the two old vertices
        '''
        _J_edge_list = list(JABCD_edges[0])
        _J_edge_list.sort(reverse=True)
        for i in _J_edge_list:
            self.vertices.pop(i)
                
        self.SetEdgesLabels()
        self.RealignVertices()
        self.SetLabelsEdges()
        
    def DeployKroneckerDelta(self, delta_labels, debug_flag=False):
        '''
        1. find edges
        2. find common vertex
        3. pick the surviving label
        3a. run across all the lists and apply the substitution
        4. create the new edge
        5. assign the new label
        6. delete the old edges
        7. delete the old labels
        8. delete the common vertex
        '''
        # 1.
        delta_edges = [self.labels_edges[label] for label in delta_labels]
        # 2.
        common_vertex = None
        for vertex in delta_edges[0]:
            if vertex in delta_edges[1]:
                common_vertex = vertex
                break
        # 3. Need to understand if there is a better choice for the final
        # adjacency list
        surviving_label = delta_labels[0]
        delenda_label = delta_labels[1]

        is_surviving_summed, i_surviving_sum = False, None
        for i, elem in enumerate(self.elists['sums']):
            if surviving_label in elem:
                is_surviving_summed = True
                i_surviving_sum = i
                break
        
        # 3a.
        pop_sum_indices = []
        for kind in self.elists:
            for i_ntuple, ntuple in enumerate(self.elists[kind]):
                if delenda_label in ntuple:
                    if kind != 'sums':
                        if debug_flag:
                            print("kind", kind)                        
                        list_ntuple = list(ntuple)
                        for i, elem in enumerate(list_ntuple):
                            if elem == delenda_label:
                                list_ntuple[i] = surviving_label
                        self.elists[kind][i_ntuple] = tuple(list_ntuple)
                    else:
                        pop_sum_indices += [i_ntuple]
                        '''
                        Need to update the summation extrema
                        '''
                        if is_surviving_summed:
                            extrema_delenda = self.elists['sums_extrema'][i_ntuple]
                            self.elists['sums_extrema'][i_surviving_sum] = \
                                extrema_delenda + self.elists['sums_extrema'][i_surviving_sum]
                        
        pop_sum_indices.sort(reverse=True)        
        for i in pop_sum_indices:
            self.list_sums.pop(i)
            self.list_sums_extrema.pop(i)
        
        # 4.
        uncommon_vertices = []
        for edge in delta_edges:
            for vertex in edge:
                if vertex != common_vertex:
                    uncommon_vertices += [vertex]
                    
        new_edge = tuple(uncommon_vertices)
        if new_edge in self.edges:
            new_edge = (new_edge[1], new_edge[0])

        self.edges += [new_edge]
        # 5.
        self.labels_edges[surviving_label] = new_edge
        # 6.
        edges_indices = []
        for edge in delta_edges:
            edges_indices += [self.edges.index(edge)]
        edges_indices.sort(reverse=True)
        
        for i in edges_indices:
            self.edges.pop(i)
        # 7.
        del self.labels_edges[delenda_label]
        # 8.
        self.vertices.pop(common_vertex)
        
        self.SetEdgesLabels()
        self.RealignVertices()
        self.SetLabelsEdges()
        
    def SetLoopColors(self, loops):
        for loop in loops:
            for label in loop:
                self.colors[label][0] += 1

    def SetLoopColorsMod2(self, loops):
        for loop in loops:
            for label in loop:
                self.colors[label][0] = (self.colors[label][0] + 1) % 2
                
    def ResetColors(self):
        for label in self.colors:
            self.colors[label][0] = 0
            
    def InitSumColors(self, sums_labels):
        for label in sums_labels:
            self.colors[label[0]] = [0]

    def FindExtremaLevel(self, level, q=None):
        colors_extrema = []

        for label_i in self.labels_sums_indices_levels[level]:
            colors_extrema += [self.GetColorSumExtrema(label_i, q=q)]

        return colors_extrema

    def FindExtremaLevelOld(self, level):
        colors_extrema = []

        for i in range(len(self.sum_labels_adj_list)):
            if self.sum_labels_levels[i] == level:
                colors_extrema += [self.GetColorSumExtrema(self.elists['sums'][i][0])]

        return colors_extrema

    def FindSumConfigurations(self, present_level, q=None):
        N_levels = np.amax(self.sum_labels_levels)
        if present_level < N_levels + 1:
            present_extrema = self.FindExtremaLevel(level=present_level, q=q)
            '''
            Check if extrema are ordered
            '''
            is_there_empty_sum = False
            for extrema in present_extrema:
                if extrema[0] < extrema[1]:
                    is_there_empty_sum = True
                    break

            if is_there_empty_sum:
                return None
            else:
                present_N = TotalNumberOfSequences(present_extrema)
                n_extrema = len(present_extrema)
                lengths_extrema = [extrema[0] - extrema[1] + 1 for extrema in present_extrema]
                strides_extrema = [math.prod(lengths_extrema[-j-1:]) for j in range(n_extrema - 1)]
                
                present_level_list = []
                if present_level < N_levels:            
                    for i in range(present_N):
                        ith_conf = \
                            FromExtremaToSequence(present_extrema, present_N, n_extrema,
                                                  strides_extrema, i=i)

                        '''
                        Assign the selected colors
                        '''
                        for j in range(n_extrema):
                            self.colors_sums_levels[present_level][j][0] = ith_conf[j]
                            
                        next_level_list = \
                            self.FindSumConfigurations(present_level=present_level + 1, q=q)


                        if next_level_list is not None:
                            present_level_list += [(i, next_level_list)]

                else:                    
                    present_level_list = list(range(present_N))                        

                return present_level_list

    def GetEvaluation(self, q=None, debug_flag=False):
        if self.N is None and self.AmITetrahedron():
            self.EvaluateTetrahedron()
            self.GetFactors()
            self.evaluation = self.GetPrefactorEvaluation(q=q)
            return self.evaluation
        elif self.N_init == 2:
            self.GetFactors()
            self.evaluation = self.GetPrefactorEvaluation(q=q)
            return self.evaluation
        elif self.N_init > 2:
            self.GetFactors()
            '''
            Find adjacency list for summed indices
            '''
            self.sum_labels_adj_list = self.FindSumsAdjacencyList()
            
            '''
            Find levels
            '''
            self.sum_labels_levels = []
            for i in range(len(self.sum_labels_adj_list)):
                if len(self.sum_labels_adj_list[i]) == 0:
                    self.sum_labels_levels += [0]
                else:
                    levels_tmp = []
                    for j in self.sum_labels_adj_list[i]:
                        levels_tmp += [self.sum_labels_levels[j]]
                    self.sum_labels_levels += [max(levels_tmp) + 1]
            '''
            Define a dict of lists of pointers selecting the colors by level
            '''
            self.colors_sums_levels, self.labels_sums_levels = {}, {}
            self.labels_sums_indices_levels = {}

            N_levels = np.amax(self.sum_labels_levels)
            for level in range(N_levels + 1):
                self.colors_sums_levels[level] = []
                self.labels_sums_levels[level] = []
                self.labels_sums_indices_levels[level] = []

            for sum_i, label in enumerate(self.elists['sums']):
                self.colors_sums_levels[self.sum_labels_levels[sum_i]] += \
                    [self.colors[label[0]]]
                self.labels_sums_levels[self.sum_labels_levels[sum_i]] += \
                    [label[0]]
                self.labels_sums_indices_levels[self.sum_labels_levels[sum_i]] += \
                    [self.sums_indices[label[0]]]

            ## return
                    
            '''
            Find the tree
            '''
            if debug_flag:
                print("Looking for compatible sums configurations...")
                st = SimpleTiming()
                st.Start()

            self.color_tree = self.FindSumConfigurations(present_level=0, q=q)

            if debug_flag:
                st.End()
                st.PrintElapsedTime()

            '''
            Prune branches
            '''
            if debug_flag:
                print("Unrolling the branches...")
                st.Start()

            self.unrolled_color_branches = \
                UnrollBranches(self.color_tree, 0, 
                               self.sum_labels_levels[-1],
                               debug_flag=debug_flag)

            if debug_flag:
                st.End()
                st.PrintElapsedTime()
                
                print("The unrolled branches are:")
                print(self.unrolled_color_branches)
            
            '''
            Now, we are ready to cycle over all the branches in order to sum the
            evaluations:
            1. GetFactors
            2. Initialize the evaluation to 0 and sum the 'not-prefactor' part
            3. Multiply by the evaluation of the prefactor
            '''        
            self.evaluation = sp.core.numbers.Zero()
            
            if self.unrolled_color_branches is not None:
                for branch in self.unrolled_color_branches:
                    if debug_flag:
                        print("branch:", branch)
                    for level, x in enumerate(branch):
                        if debug_flag:
                            print("level, x:", level, x)

                        ranges = self.FindExtremaLevel(level, q=q)
                        x_colors = FromExtremaToSequence(ranges, i=x)

                        if debug_flag:
                            print("ranges:", ranges, "x_colors:", x_colors)

                        '''
                        Assign the selected colors
                        '''
                        for j in range(len(ranges)):
                            self.colors_sums_levels[level][j][0] = x_colors[j]
                                
                    '''
                    After assigning all the colors we can compute the evaluation
                    and sum
                    '''
                    self.evaluation += self.GetNotPrefactorEvaluation(q=q)
                    
                if type(self.evaluation) != sp.core.numbers.Zero:
                    self.evaluation *= self.GetPrefactorEvaluation(q=q)
            
            return self.evaluation

    def FindAllCyclesLexPoly(self, debug_flag=False):
        M = len(self.lexicographic_polygons)

        self.cycles_lex_poly = {}
        self.cycles_lex_poly[1] = [[m] for m in range(M)]

        st = SimpleTiming()

        st.Start()
        for i in range(1, M):
            polygons_colorings_trial = \
                GetSinglePermutations(1, i + 1, M - (i + 1))
            if debug_flag:
                print(polygons_colorings_trial)
            
            connected_colorings = []
            for coloring in polygons_colorings_trial:
                non_zero_polygons_list = np.where(coloring == 1)[0]

                self.SetLoopColorsMod2(
                    [self.lexicographic_polygons[i] for i in non_zero_polygons_list]
                )
                
                self.SetLabelsInCycle()
                set_edges = \
                    [self.labels_edges[label] for label in self.labels_in_cycle]
                
                if self.AreEdgesContiguous(set_edges):
                    connected_colorings += [non_zero_polygons_list.tolist()]
                
                self.SetLoopColorsMod2(
                    [self.lexicographic_polygons[i] for i in non_zero_polygons_list]
                )

            self.cycles_lex_poly[i + 1] = connected_colorings
            print(i + 1, len(polygons_colorings_trial), len(connected_colorings))
        st.End()
        elapsed_time = st.GetElapsedTime()

        ##RunThroughDict(dictionary=self.cycles, edit_function=Edit_ListToNPArray)

        return elapsed_time

    def GetAllCyclesLexPoly(self, dump_file=None):
        cycles_dump, md = False, None
        M = len(self.lexicographic_polygons)

        if dump_file is not None:
            cycles_key = 'cycles/' + str(M)
            md = ManageData(dump_file=dump_file)
            if md.ReadJson():
                cycles_dump = md.IsThereKey(cycles_key)
        
        if cycles_dump == False and self.cycles_lex_poly is None:
            self.cycles_elapsed_time = self.FindAllCyclesLexPoly()

        if cycles_dump == False and md is not None and self.cycles_lex_poly is not None:
            md.PushData(data=True, key=cycles_key)
            md.PushData(data=self.cycles_lex_poly, key=cycles_key + '/dict')
            md.PushData(data=self.cycles_elapsed_time, key=cycles_key + '/elapsed_time')

            md.DumpJson(indent=None)

        elif cycles_dump != False and self.cycles_lex_poly is None:

            self.cycles_elapsed_time = \
                md.PullData(key=cycles_key + '/elapsed_time')
            self.cycles_lex_poly = md.PullData(key=cycles_key + '/dict')

    def GetAllCycles(self, dump_file=None):
        self.GetAllCyclesLexPoly(dump_file=dump_file)
        self.cycles_edges = []

        for N_lex_polys in self.cycles_lex_poly:
            for cycle in self.cycles_lex_poly[N_lex_polys]:
                ## COLOR
                self.SetLoopColorsMod2(
                    [self.lexicographic_polygons[i] for i in cycle]
                )
                ## Find Edges in Cycle
                self.SetLabelsInCycle()
                self.cycles_edges += [self.labels_in_cycle]

                ## UNCOLOR
                self.SetLoopColorsMod2(
                    [self.lexicographic_polygons[i] for i in cycle]
                )
        self.n_cycles = len(self.cycles_edges)

        return self.cycles_edges

    def InitAllCyclesLength(self):
        self.cycles_edges_length = []
        for cycle in self.cycles_edges:
            self.cycles_edges_length += [len(cycle)]

        ## Preparing the dictionary
        self.cycles_edges_ldict = {}
        for l in range(min(self.cycles_edges_length), max(self.cycles_edges_length) + 1):
            self.cycles_edges_ldict[l] = []

        for cycle in self.cycles_edges:
            self.cycles_edges_ldict[len(cycle)] += [cycle]

        return self.cycles_edges_ldict

    def GetCountCyclesCombinations(self, min_color, max_color, print_flag=False):
        count = 0
        for color_sum in range(min_color, max_color + 1):
            if print_flag:
                print("color_sum:", color_sum)
                
            for conf in FindSumCombinationsTuples(color_sum, cutoff=self.n_cycles):
                multinomial_tuple = tuple()
                sum_elements = 0
                for c_elem in conf:
                    # print(c_elem)
                    sum_elements += c_elem[1]
                    multinomial_tuple += (c_elem[1],)

                multinomial_tuple = (self.n_cycles - sum_elements,) + multinomial_tuple

                if print_flag:
                    print("multinomial_tuple:", self.n_cycles, multinomial_tuple)
                    
                count += int(Multinomial(self.n_cycles, multinomial_tuple))
        return count


    def GetCyclesCombinations(self, min_color, max_color):
        cycles_configurations, colors_combinations = [], []
        for color_sum in range(min_color, max_color + 1):
            for conf in FindSumCombinationsTuples(color_sum, cutoff=self.n_cycles):
                cycles_configurations += [GetUniquePermutations(conf, self.n_cycles)]
                colors_combinations += [conf]
        return cycles_configurations, colors_combinations

        
def GetNBYN(N: int, prune_depth=6):
    _N_parity = N % 2
    '''
    Starting from the base
    '''
    _sum = SKN(-1, 0)
    '''
    Growing larger...
    '''
    _n_seqs = []
    for k in range(0, N):
        _n_seq_hi = np.arange(1, k // 2 + 2)
        _n_seq_lo = -np.flip(_n_seq_hi)
        
        _n_seq_swap = list(_n_seq_lo) + ([0] if k % 2 == 1 else []) + list(_n_seq_hi)
        _n_seqs += [_n_seq_swap]
    '''
    ...then reverse
    '''
    _hi_len = len(_n_seqs)
    if _hi_len > 1:
        for k in range(_hi_len - 1):
            _n_seqs += [_n_seqs[_hi_len - 2 - k]]
    '''
    summing up...
    '''
    for k in range(len(_n_seqs)):
        for n in _n_seqs[k]:
            _sum += SKN(k, n)
    '''
    ending with the tip
    '''
    _sum += SKN(2 * N - 1, 0)
    _sum.SetN(N)
    _sum.SetNInit(N)

    '''
    final pruning
    '''
    for i in range(prune_depth):
        if i == 0:
            _sum.ClassifyHalfEdges()
        if i == 1:
            _sum.MergeHalfEdges()
        if i == 2:
            _sum.PruneHalfEdges()
        if i == 3:
            _sum.PruneDegreeOne()
        if i == 4:
            _sum.PruneDegreeTwo()
        if i == 5:
            _sum.InitColors()
            _sum.SetLabelsEdges()
            _sum.SetBoundaryVertices()
            _sum.SetLexicographicPolygons()
            
    return _sum

def GetTetrahedron():
    tetrahedron = \
        SKN(0, 1) + SKN(0, -1) + SKN(-1, 0) + \
        SKN(1, 0) + SKN(1, -1) + SKN(1, 1) + \
        SKN(2, -1) + SKN(2, 1)

    tetrahedron.ClassifyHalfEdges()
    tetrahedron.MergeHalfEdges()
    tetrahedron.PruneHalfEdges()
    tetrahedron.PruneDegreeOne()
    tetrahedron.PruneDegreeTwo()

    tetrahedron.InitColors()
    tetrahedron.SetLabelsEdges()
    tetrahedron.SetBoundaryVertices()
    tetrahedron.SetLexicographicPolygons()

    return tetrahedron


def ProcessCrown(hexagon, prune_depth=5, select_step=None):
    if select_step is not None:
        prune_depth=0
        
    if hexagon.N > 2:
        if prune_depth > 0 or (select_step is not None and select_step == 0):
            '''
            1. Take care of the tip triangle
            '''     
            hexagon.LemmaZappalaUpperTriangles()            

        if prune_depth > 1 or (select_step is not None and select_step == 1):
            '''
            2. Find recoupling labels
            '''                
            _recoupling_labels = hexagon.FindRecouplingLabels()
            '''
            3. Perform the recoupling
            '''
            for label in _recoupling_labels:
                hexagon.Recoupling(label)

        if prune_depth > 2 or (select_step is not None and select_step == 2):
            '''
            4. Removing the deltas
            '''
            _tmp_list_kdelta = hexagon.list_kdelta.copy()
            for delta in _tmp_list_kdelta:
                hexagon.DeployKroneckerDelta(delta)
                hexagon.CleanDeltas()

    if hexagon.N - 1 > 2 or hexagon.N == 2:
        if prune_depth > 3 or (select_step is not None and select_step == 3):
            '''
            5. Reapply Zappala's lemma
            '''
            hexagon.LemmaZappalaUpperTriangles()

    '''
    6. N--
    '''
    if prune_depth > 4 or (select_step is not None and select_step == 4):
        hexagon.SetN(hexagon.N - 1)

        if hexagon.N == 1 and hexagon.AmITetrahedron():
            hexagon.EvaluateTetrahedron()
            
def FullReduction(hexagon):
    if not hexagon.AmITetrahedron():
        N_start = hexagon.N
        for i in range(N_start - 1):
            ProcessCrown(hexagon=hexagon, prune_depth=5, select_step=None)
        hexagon.SetSumIndices()
        hexagon.InitSumColors(hexagon.elists['sums'])
    else:
        hexagon.EvaluateTetrahedron()

############# Plots modifiers functions

def PrintEdgesLabels(myself, plt, fontsize=15):
    for edge in myself.edges:
        _v0, _v1 = edge
        _edge_label = FromLabelToStrLatex(myself.labels[edge])
        
        _delta_x, _delta_y = 0, 0
        if myself.labels[edge][1] == 'c':
            _delta_x, _delta_y = -0.2, 0.2
            
        x_v0, y_v0 = myself.coords[_v0][0], myself.coords[_v0][1]
        x_v1, y_v1 = myself.coords[_v1][0], myself.coords[_v1][1]

        x_pos_label, y_pos_label = \
            0.5 * (x_v0 + x_v1) + (_delta_x), \
            0.5 * (y_v0 + y_v1) + (_delta_y)
                
        plt.text(x_pos_label, y_pos_label, _edge_label, fontsize=fontsize)
        
def PrintVertex(myself, plt, fontsize=15):
    for vertex in myself.vertices:
        x, y = myself.coords[vertex][0], myself.coords[vertex][1]
        vertex_label = '$' + str(vertex) + '$'
        plt.text(x, y, vertex_label, fontsize=fontsize)
        
def PrintBoundaryVertex(myself, plt, markersize=10, color='red'):
    for vertex in myself.vertices:
        x, y = myself.coords[vertex][0], myself.coords[vertex][1]
        vertex_label = '$' + str(vertex) + '$'
        if myself.boundary_vertices[vertex]:
            plt.plot([x], [y], 'o', markersize=markersize, color=color)
        
def HighlightEdges(myself, plt, edges_list=[], colors=[], lineswidth=[]):
    if len(edges_list) == 0:
        raise Exception("Need a list of edges!")
    if len(colors) == 0:
        colors = ['red'] * len(edges_list)
    if len(lineswidth) == 0:
        lineswidth = [1] * len(edges_list)        
        
    for edge, color, lw in zip(edges_list, colors, lineswidth):
        _coords_x = [myself.coords[vertex][0] for vertex in edge]
        _coords_y = [myself.coords[vertex][1] for vertex in edge]
        plt.plot(_coords_x, _coords_y, color=color, linewidth=lw)
        
def PlotColors(myself, plt, color='red', colorp='darkgreen', width_mult=1):
    for label in myself.labels_edges:
        edge = myself.labels_edges[label]
        _coords_x = [myself.coords[vertex][0] for vertex in edge]
        _coords_y = [myself.coords[vertex][1] for vertex in edge]
        
        edge_color = None
        
        if label[1] != 'p':
            thickness = myself.colors[label][0]
            edge_color=color
        else:
            thickness = 1
            edge_color=colorp
        
        plt.plot(_coords_x, _coords_y, color=edge_color, 
                 linewidth=thickness*width_mult)
