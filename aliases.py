###############################################################################
''''''
###############################################################################


import os
import sys


thispath = os.path.abspath(os.path.dirname(__file__))
workpath = os.path.dirname(thispath)
everestpath = os.path.join(workpath, 'everest')
if not everestpath in sys.path:
    sys.path.insert(0, everestpath)


###############################################################################

###############################################################################
