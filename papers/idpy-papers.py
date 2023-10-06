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

'''
Provides a database for cloning the published papers
'''

from collections import defaultdict
import subprocess, os

class IdpyPapers:
    def __init__(self):
        self.arxiv_papers = defaultdict(
            lambda: defaultdict(dict)
        )

        self.SetPapers()

    def ShowPapers(self):
        print("The repositories related to these papers can be cloned")
        for key in self.arxiv_papers:
            print("['" + key +"']",
                  self.arxiv_papers[key]["Title"],
                  "-- doi: " + self.arxiv_papers[key]["doi"]
                  if self.arxiv_papers[key]["doi"] != "" else "")
            print()

    def SetPapers(self):
        self.arxiv_papers['arXiv-2009.12522'] = \
            {"Title": "Structure and Isotropy of Lattice Pressure Tensors for Multi-range Potentials",
             "Authors": ["Matteo Lulli", "Luca Biferale",
                         "Giacomo Falcucci", "Mauro Sbragaglia",
                         "Xiaowen Shan"],
             "doi": "10.1103/PhysRevE.103.063309",
             "doi-dir": "doi-10.1103-PhysRevE.103.063309",
             "git": "https://github.com/lullimat/arXiv-2009.12522.git"}
        
        self.arxiv_papers['arXiv-2105.08772'] = \
            {"Title": "A Mesoscale Perspective on the Tolman Length",
             "Authors": ["Matteo Lulli", "Luca Biferale",
                         "Giacomo Falcucci", "Mauro Sbragaglia",
                         "Xiaowen Shan"],
             "doi": "10.1103/PhysRevE.105.015301",
             "doi-dir": "doi-10.1103-PhysRevE.105.015301",
             "git": "https://github.com/lullimat/arXiv-2105.08772.git"}

        self.arxiv_papers['arXiv-2112.02574'] = \
            {"Title": "Mesoscale Modelling of the Tolman Length in Multi-component Systems",
             "Authors": ["Matteo Lulli", "Luca Biferale",
                         "Giacomo Falcucci", "Mauro Sbragaglia",
                         "Xiaowen Shan"],
             "doi": "", "doi-dir": "",
             "git": "https://github.com/lullimat/arXiv-2112.02574.git"}
        
        self.arxiv_papers['arXiv-2212.07848'] = \
            {"Title": "Metastable and Unstable Dynamics in multi-phase lattice Boltzmann",
             "Authors": ["Matteo Lulli", "Luca Biferale",
                         "Giacomo Falcucci", "Mauro Sbragaglia", "Dong Yang",
                         "Xiaowen Shan"],
             "doi": "", "doi-dir": "",
             "git": "https://github.com/lullimat/arXiv-2212.07848.git"}

        self.arxiv_papers['arXiv-2310.03632'] = \
            {"Title": "The exact evaluation of hexagonal spin-networks and topological quantum neural networks",
             "Authors": ["Matteo Lulli", "Antonino Marciano", "Emanuele Zappala"],
             "doi": "", "doi-dir": "",
             "git": "https://github.com/lullimat/arXiv-2310.03632"}
        
    def GitClone(self, key):
        subprocess.call(["git", "clone", self.arxiv_papers[key]['git']])

    def SymLink(self, key):
        if self.arxiv_papers[key]['doi'] != "":
            print("Creating symlink to:", self.arxiv_papers[key]['doi-dir'])
            os.symlink(key, self.arxiv_papers[key]['doi-dir'])

if __name__ == "__main__":
    print("Welcome!")
    print("Here you can retreive the papers published in the idea.deploy framework")
    
    print("")
    idpy_papers = IdpyPapers()
    idpy_papers.ShowPapers()
    key = input("Copy and paste the repository string: ")
    key = str(key)
    if key in idpy_papers.arxiv_papers:
        print("Cloning the repository...")
        idpy_papers.GitClone(key)
        print()
        idpy_papers.SymLink(key)
        
    else:
        raise Exception(key, "not present in the data base!")
        
