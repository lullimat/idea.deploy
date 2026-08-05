__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2020-2026 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
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
A shared exit convention for the print-style test scripts.

Those scripts (test_shared, test_residency, test_residency_policy, test_linkage,
test_constants, test_hostmodule, test_overlap) are written to be read by a human
running them on a machine with particular hardware: they print a line per check
and skip backends that are absent. That is the right shape for what they do, but
until now every one of them returned None from main() and therefore exited 0 no
matter what -- a forced failure still reported success. As gates they were
theatre, which is precisely the trap already documented for idpy/LBM/test.py
running zero tests and exiting 0.

The convention:

    exit 1   at least one check failed, or a backend raised
    exit 0   every check that ran passed

and, separately from the exit code, a script that verified *nothing* says so in
as many words. A CTypes-only machine skipping every GPU backend is a legitimate
outcome, not a failure -- but "0 checks ran" must never look like "all checks
passed", which is the same confusion in a different costume.
'''

import sys


def report_exit(ok, checks_run=True, what='checks'):
    '''
    Terminate a print-style test script with a meaningful status.

    'ok' is the aggregate verdict, 'checks_run' whether anything was actually
    verified. Prints the no-op case loudly rather than letting an empty run pass
    for a clean one, then exits 1 on failure and 0 otherwise.
    '''
    if not checks_run:
        print(f"\n  NOTE: no {what} ran on this machine -- nothing was "
              f"verified here. This is not a pass.")
    sys.exit(0 if ok else 1)
