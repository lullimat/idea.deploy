"""
Provides a simple function to check whether a module is installed
in the system. Works with Python3
"""

import importlib

__author__ = "Matteo Lulli"
__copyright__ = "Copyright 2020, idea.deploy"
__credits__ = ["Matteo Lulli"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Matteo Lulli"
__email__ = "matteo.lulli@gmail.com"
__status__ = "Development"

def IsModuleThere(module_name = None):
    '''
    function IsModuleThere:
    simple wrapping function for checking module existence in the
    system. Can be used for running system's test and conditional
    modules imports
    '''
    swap_spec = importlib.util.find_spec(module_name)
    return swap_spec is not None

def AreModulesThere(modules_list = None):
    '''
    function AreModulesThere:
    execute IsModuleThere in a loop
    '''
    if not isinstance(modules_list, list):
        raise Exception("modules_list must be a list")
    
    answers = {}
    for module in modules_list:
        answers[module] = IsModuleThere(module)
    return answers
