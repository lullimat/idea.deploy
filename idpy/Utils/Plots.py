__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2022 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
__credits__ = ["Matteo Lulli"]
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
__maintainer__ = "Matteo Lulli"
__email__ = "matteo.lulli@gmail.com"
__status__ = "Development"

import matplotlib.pyplot as plt

def CreateFiguresPanels(_nx, _ny, _x_size = 6.5, _y_size = 4.8):
    return plt.figure(figsize = (_nx * _x_size, _ny * _y_size))

def SetMatplotlibLatexParamas(rc, rcParams, mathfont='sourcesanspro'):
    if type(rc) != list:
        raise Exception("Parameter 'rc' must be a list, typically passed as [rc]")

    if type(rcParams) != list:
        raise Exception("Parameter 'rcParams' must be a list, typically passed as [rcParams]")

    rc[0]('font',**{'family':'STIXGeneral'})
    ##rc[0]('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc[0]('mathtext', **{'fontset': 'stix'})
    rcParams[0]['text.usetex'] = True
    rc[0]('text', usetex=True)
    #rcParams[0]['text.latex.preview'] = True
    if mathfont == 'sourcesanspro':
        rcParams[0]['text.latex.preamble']=[r"\usepackage{amsmath, sourcesanspro}"]
    elif mathfont == 'ams':
        print("Setting 'ams' font")
        rcParams[0]['text.latex.preamble']=[r"\usepackage{eucal}"]

def SetDefaultFonts(rc, 
                    font_size = 20, legend_font_size = 17, 
                    marker_size_small = 9, marker_size_large = 12, 
                    marker_size_large_c = 20, 
                    thick_line_width = 4, normal_line_width = 1, thin_line_width = 0.5):
    if type(rc) != list:
        raise Exception("Parameter 'rc' must be a list, typically passed as [rc]")
    
    _fs = font_size
    _l_fs = legend_font_size
    _ms_small = marker_size_small
    _ms_large = marker_size_large
    _ms_large_c = marker_size_large_c
    
    rc[0]('legend', fontsize = _l_fs)
    
    return {'font_size': _fs, 'fs': _fs, 'legend_font_size': _l_fs, 'l_fs': _l_fs, 
            'marker_size_small': _ms_small, 'ms_small': _ms_small, 
            'marker_size_large': _ms_large, 'ms_large': _ms_large, 
            'marker_size_large_c': _ms_large_c, 'ms_large_c': _ms_large_c, 
            'thick_line_width': thick_line_width, 'lw_T': thick_line_width, 
            'normal_line_with': normal_line_width, 'lw_n': normal_line_width, 
            'thin_line_width': thin_line_width, 'lw_t': thin_line_width}

def SetAxTicksFont(ax, fs):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fs)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)
        
def SetAxPanelLabel(ax, label, fs, x_pos = 0.015, y_pos = 0.91, pos=None, *args, **kwargs):
    if pos is not None and pos == 'ul':
        x_pos, y_pos = x_pos, y_pos
    if pos is not None and pos == 'll':
        x_pos, y_pos = x_pos, x_pos + 0.03
    if pos is not None and pos == 'ur':
        x_pos, y_pos = y_pos, y_pos
    if pos is not None and pos == 'lr':
        x_pos, y_pos = y_pos, x_pos
        
    ax.text(x_pos, y_pos, label, transform=ax.transAxes, fontsize=fs, *args, **kwargs)

def SetAxPanelLabelCoords(ax, label, fs, x_pos = 0.015, y_pos = 0.91, pos=None, *args, **kwargs):
    if pos is not None and pos == 'ul':
        x_pos, y_pos = x_pos, y_pos
    if pos is not None and pos == 'll':
        x_pos, y_pos = x_pos, x_pos + 0.03
    if pos is not None and pos == 'ur':
        x_pos, y_pos = y_pos, y_pos
    if pos is not None and pos == 'lr':
        x_pos, y_pos = y_pos, x_pos
        
    ax.text(x_pos, y_pos, label, fontsize=fs, *args, **kwargs)