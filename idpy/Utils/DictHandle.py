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

from collections import defaultdict
from collections.abc import Iterable
import numpy as np

def RunThroughDict(dictionary, edit_function = None, verbose = False):
	for _key in dictionary:
		if type(dictionary[_key]) == dict or type(dictionary[_key]) == defaultdict:
			if verbose:
				print("Rec:", _key)
			RunThroughDict(dictionary[_key], edit_function)
		else:
			if isinstance(dictionary[_key], Iterable):
				for _elem_i, _elem in enumerate(dictionary[_key]):
					if type(dictionary[_key][_elem_i]) == dict or type(dictionary[_key][_elem_i]) == defaultdict:
						if verbose:
							print("Rec:", _key, _elem_i)
						RunThroughDict(dictionary[_key][_elem_i], edit_function)

					elif hasattr(dictionary[_key], '__setitem__'):
						for _elem_i, _elem in enumerate(dictionary[_key]):
							dictionary[_key][_elem_i], _out = edit_function(_elem, verbose)
							if verbose:
								print("Key:", _elem)
								if _out is not None and _out:
									print(_out)

					elif not hasattr(dictionary[_key], '__setitem__') and type(dictionary[_key]) != str:
						for _elem_i, _elem in enumerate(dictionary[_key]):
							_null, _out = edit_function(_elem, verbose)
							if verbose:
								print("Key:", _elem)
								if _out is not None and _out:
									print(_out)


			dictionary[_key], _out = edit_function(dictionary[_key], verbose)
			if verbose:
				print("Key:", _key)
			if _out is not None and _out:
				if verbose:
					print(_out)

def Edit_NPArrayToList(elem = None, verbose = False):
	if type(elem) == np.ndarray:
		return elem.tolist(), True
	else:
		return elem, False

def Edit_ListToNPArray(elem = None, verbose = False):
	if type(elem) == list:
		return np.array(elem), True
	else:
		return elem, None

def Edit_Int64ToInt(elem = None, verbose = False):
	if type(elem) == np.int64:
		return int(elem), True
	else:
		return elem, None

def Edit_Float64ToFloat(elem = None, verbose = False):
	if type(elem) == np.float64:
		return float(elem), True
	else:
		return elem, None

def Edit_Float32ToFloat(elem = None, verbose = False):
	if type(elem) == np.float32:
		return float(elem), True
	else:
		return elem, None

def Check_WhichNDArray(elem = None, verbose = False):
	return elem, type(elem) == np.ndarray

def Check_WhichInt64(elem = None, verbose = False):
	return elem, type(elem) == np.int64

def Check_WhichType(elem = None, verbose = False):
	if verbose:
		print(type(elem))
	return elem, None