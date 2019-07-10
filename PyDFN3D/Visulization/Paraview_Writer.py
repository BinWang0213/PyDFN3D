#########################################################################
#       (C) 2017 Department of Petroleum Engineering,                   #
#       Univeristy of Louisiana at Lafayette, Lafayette, US.            #
#                                                                       #
# This code is released under the terms of the BSD license, and thus    #
# free for commercial and research use. Feel free to use the code into  #
# your own project with a PROPER REFERENCE.                             #
#                                                                       #
# PYBEM2D Code                                                          #
# Author: Bin Wang                                                      #
# Email: binwang.0213@gmail.com                                         #
# Reference: Wang, B., Feng, Y., Berrone, S., et al. (2017) Iterative   #
# Coupling of Boundary Element Method with Domain Decomposition.        #
# doi:                                                                  #
#########################################################################

import numpy as np
import os
from meshio import vtu_io
from meshio import Mesh

#DFN VTK Writer
from .vtkDFNWriter import vtkDFNWriter
from .IO_PyGeoMesh_msh import read_PyGeoMeshFile_msh

class Paraview_Writer:
    def __init__(self, fname=None,folder="Results"):
    
        #Set up the output folder
        self.fname=fname
        self.folder=folder

        self.fname_geo = None
        self.fname_mesh = None
        self.GetFilePath()

        os.path.splitext(fname)[0] #remove the extension
        os.makedirs(folder,exist_ok=True)#Create a output folder

        #Mesh Data Container
        self.DFN_MeshData = None
        
        #External DFN Geo
        self.DFN_Writer = None

        #Extendable cell data (nested dict)
        self.meshio_celldata = None  #element-wise data
        self.meshio_ptsdata  = None #point-wise data

    def GetFilePath(self):
        self.fname_mesh = os.path.join(self.folder, self.fname + "_solution.vtu")
        self.fname_geo = os.path.join(self.folder, self.fname + "_DFN.vtp")

    def SetInputMesh(self,mesh_input_fname):
        self.DFN_MeshData=read_PyGeoMeshFile_msh(mesh_input_fname)

    def SetInputDFN(self,FracNets):
        self.DFN_Writer = vtkDFNWriter(self.fname_geo, FracNets)
    
    def AppendCellData(self,varName,varData):
        if(self.meshio_celldata is None):
            self.meshio_celldata={'triangle': {}}
        
        self.meshio_celldata['triangle'][varName]=varData

    def AppendPointData(self,varName,varData):
        if (self.meshio_ptsdata is None):
            self.meshio_ptsdata = {}

        self.meshio_ptsdata[varName]=varData

    def Write(self):
        #Update filepath
        self.GetFilePath()

        #DFN geo file
        if(self.DFN_Writer is not None):
            self.DFN_Writer.Write()
            print("[Output] Saved Geometry VTK file %s) !" % (self.fname_geo))
            #return self.fname_geo

        #Mesh file
        if(self.DFN_MeshData is not None):
            meshio_cells = {'triangle': self.DFN_MeshData.Elements}
            meshio_points = self.DFN_MeshData.Points
            mesh=Mesh(meshio_points,meshio_cells,point_data=self.meshio_ptsdata,cell_data=self.meshio_celldata)
            vtu_io.write(self.fname_mesh, mesh)
            print("[Output] Saved VTK mesh and solution file %s) !" % (self.fname_mesh))
            #return self.fname_mesh

