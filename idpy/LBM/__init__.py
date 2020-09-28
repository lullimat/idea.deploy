import inspect, os, sys
from sys import platform

_module_abs_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
_idea_dot_deploy_path = os.path.dirname(os.path.abspath(_module_abs_path + "../../"))
'''
append to sys path in order to avoid relative imports
'''
sys.path.append(_idea_dot_deploy_path)
