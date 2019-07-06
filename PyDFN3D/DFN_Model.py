
import numpy as np

from PyDFN3D.DFN_Geometry.FracsGeo import FractureNetworks

class DFN_Model:
    """Main PyDFN3D class to handle all generic operations"""

    def __init__(self):
        """Creates a BEM objecti with some specific paramters
        
        Arguments
        ---------
        Dim            -- Model Dimension
        NumFracs       -- Number of Fractures

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        """
        self.fname=''
        self.Geometry=FractureNetworks()
    
    def loadDFN(self,fname,ftype='PyGeoMesh'):
        """Load DFN from file

        Support format:
        PyGeoMesh (release soon) *.dat

        Future plan:
        *.fab
        
        Arguments
        ---------
        Dim            -- Model Dimension
        NumFracs       -- Number of Fractures

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        """
        self.Geometry.loadFracs(fname,ftype)
