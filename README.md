# Welcome to the **idea.deploy** project!

**Changes are being pushed quite frequently. Please, keep on checking for updates.**

Thanks for passing by! If you are visiting this repository it is likely because of the first two preprints making use of it (see below).

The repository is at an early stage in this moment and will keep evolving (hopefully for the best!) as time passes. The main spirit is to make public what has been used to obtain the numerical results published both on the public archive arxiv.org and on peer-reviewed journals. The software and the other contents will be steadily updated, so that the main criteria is not to provide the "definitive" code version but the code version that actually made the published results possible. As the code-base grows and evolves backwards compatibility will be assured.

For reproducing the papers results please look below in the **Installation** Section and in **Typical usage for reproducing the results reported in a paper** Subsection.

## Available Peer-Reviewed Papers
### **Mesoscale Modelling of the Tolman Length in Multi-component Systems**
- Publisher link: to be announced: accepted for publication on **Proceedings of the Conference "The Legacy of Carlo Cercignani: from Kinetic Theory to Turbulence Modeling"**
- arXiv link: [https://arxiv.org/abs/2112.02574](https://arxiv.org/abs/2112.02574)
- github link: [https://github.com/lullimat/arXiv-2112.02574.git](https://github.com/lullimat/arXiv-2112.02574.git)

The Jupyter notebook has only been tested in the 'devel' branch of 'idea.deploy'. Make sure you type in the terminal from the root directory of 'idea.deploy' the command ```git checkout devel``` before using it.
### **Mesoscale Perspective on the Tolman Length**
- Publisher link: [https://journals.aps.org/pre/abstract/10.1103/PhysRevE.105.015301](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.105.015301)
- arXiv link: [https://arxiv.org/abs/2105.08772](https://arxiv.org/abs/2105.08772)
- github link: [https://github.com/lullimat/arXiv-2105.08772.git](https://github.com/lullimat/arXiv-2105.08772.git)

### **Structure and Isotropy of Lattice Pressure Tensors for Multi-range Potentials**
- Publisher link: [https://journals.aps.org/pre/abstract/10.1103/PhysRevE.103.063309](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.103.063309)
- arXiv link: [https://arxiv.org/abs/2009.12522](https://arxiv.org/abs/2009.12522)
- github link: [https://github.com/lullimat/arXiv-2009.12522](https://github.com/lullimat/arXiv-2009.12522)

## Available Preprints
Will populate this section soon...

## Installation
### Install idea.deploy python virtual environment: idpy-env
For the time being the project has been tested only on some Linux platforms (opensuse and ubuntu) and on MacOS Catalina/Big Sur/Monterey for both Intel and M1 machines.

At the moment the main code base for the project is written in python and it heavily relies, among other modules, on PyCUDA [(https://documen.tician.de/pycuda/)](https://documen.tician.de/pycuda/), PyOpenCL [(https://documen.tician.de/pyopencl/)](https://documen.tician.de/pyopencl/) and numpy [(https://numpy.org)](https://numpy.org). In order to keep the user computer as clean as possible a python virtual environment is created through the script **idpy-init.sh** and the necessary dependencies downloaded and installed. PyCUDA and PyOpenCL are both compiled from source when possible. The python environment and the other optional system modifications (like new aliases in the .bashrc) can be cleaned by using the script **idpy-clean.sh**. Since erasing files must be taken with care, each "rm" command is executed only after the user reviews the files/directories that will be removed (please check the terms of the license). The purpose is to fully clean the changes performed by the use of **idpy-init.sh**

### Typical usage for reproducing the results reported in a paper
- Install the virtual environment: **bash idpy-init.sh**, reply to the questions and follow the prompted instructions
- If you opted for installing the aliases either execute "source ~/.bashrc" or open a new terminal session and type "idpy-go" to reach the idea.deploy directory
- Type "idpy-load" to load the python virtual environment
- Go to the './papers' directory and execute "python idpy-papers.py" to list the available papers and select the one you wish to clone from its own git repository
- "cd" into the cloned directory and launch the Jupyter server "idpy-jupyter"
- copy and paste the url from the terminal in your browser to access and execute the notebook

### Details
- The alias "idpy-jupyter" launches a Jupyter notebook server (preferably) on the port 4379 with the "--no-browser" option
- The alias "idpy-jupyter-forward remotehost" forwards to your computer the port 4379 (supposedly) used on the "remotehost" computer by the jupyter session opened within the idea.deploy python virtual environment

## Purpose of the project
The purpose of the idea.deploy project can be described in three points
- To provide a unified open-source environment for scientific code for physics simulations
- To assure the reproducibility of published numerical results (one of the fundamental features of the scientific method)
- Acknowledgement of contributions

## Git model
This project will consist of this one repository with two branches, 'master' and
'devel'. All the papers and work in progress will be developed in separated git repositories so that important updates from the software side will be pushed into the master branch much faster while the original scientific work will be available only when ready. The folder ./collabs will serve as a docking point for ongoing research and will be kept empty while the folder ./papers will contain the most updated version of the preprints in ./papers/arxiv:* and the related symlink ./papers/doi:* -> ./papers/arxiv:* when the work is published on a peer-reviewed journal.

This scheme should allow for the parallel improvement of the code (all research private repositories in ./collabs would be served by the most updated version) while keeping a clean separation of the ongoing projects that will be publicly available once, at least, posted on the arxiv.

## Meta-Language
**idea.deploy** is growing as a response to my personal (and probably common) need of wasting as little code as possible while keeping a code base that is as general as possible. One of the ideas is that code development and computational/physics output should happen side by side, in order to implement a faster feedback loop leading to a wider scientific output. This is achieved by trying to spend less and less time in the development of the core coding part (GPU programming, multi-threading, mpi, disk I/O etc.) while focusing more on the new and original parts of the code, i.e. mainly the kernels for simulations and data analysis. Given the heterogeneity of the available architectures and programming languages for parallel implementations (OpenCL, CUDA, and soon Metal) pushed me to use Python for a high level implementation heavily based on PyCUDA [(https://documen.tician.de/pycuda/)](https://documen.tician.de/pycuda/)(GPUs only) and PyOpenCL [(https://documen.tician.de/pyopencl/)](https://documen.tician.de/pyopencl/) (GPUs and CPUs), but with a twist: merging the two approaches into one that automatically resolves the specific features of the language that is eventually used. Hence, all the code development can be carried out either making specific use of each language features, or by using the Idpy "meta-language" allowing to write kernels only once, which are then automatically compiled according to one of the two standards, or assigning and manipulating the device memory through a common interface that, once the language has been set at the beginning, does not make any distinction between the two languages. This elements can all be examined at work in the Lattice Boltzmann Method (LBM) module in idpy/LBM for having an idea of what it means to develop in the idea.deploy framework.

A very important consequence of this approach is that simulations belonging to physics areas with a weak mutual interaction (say Lattice Boltzmann and Ising models) will coexist in the same framework allowing for the opening of new shared directions.

Clearly, this is a growing computational project based on my work as a theoretical/computational physicist, hence, with much room for improvement from the computational point of view. I will look for ways to publish this project on its own right in order to make citation easier. However, as described in the next section, the hope is to provide a framework fostering collaborations between different people.

## Underlying Philosophy: a hopeful deal
**idea.deploy** allows whoever is interested to pick up the work that is published through this platform exactly from the (edge) state at which it has been published, either on the arxiv, or after peer review in a journal. The main idea is that of serving the scientific community with reliable results which can be improved minimizing the cost of reproducing the results already obtained. Doing this implies much supplementary work on the side of those who publish their work through this platform (so far only myself) which is meant to be only for the sake of sparing people's time and give a possible way to implement reproducibility (a pillar of the scientific method) in computational physics. This is done as a "generous" act.

Clearly, I am willing to collaborate with those researchers who would like to port their code/work in this framework and use the idea.deploy project as a gateway for their own research. At the same time, researchers might be perfectly able to adapt the code themselves in which case the only thing to do would be to interact for merging the code in the master branch and make it available. Moreover, the code could just be used, as it is, for other research outlets in which case citing the proper reference paper is highly recommended and appreciated.

## To do (short-term)
At the moment the scripts can create a python virtual environment only for Linux and MacOS: no testing of the **idpy-init.sh** and **idpy-clean.sh** has been performed so far on Windows Linux Subsystem. Most probably the best approach will be to rewrite the above scripts in python so that they will be Os independent. However, quite a few implementation details depend on the operative system, especially as far as compilers are concerned, e.g. escaping spaces when passing options flags and so on.

Matteo Lulli, June 1st 2022
[www.ideadeploy.io](www.ideadeploy.io)
[www.lullimat.org](https://www.lullimat.org)
matteo.lulli@gmail.com