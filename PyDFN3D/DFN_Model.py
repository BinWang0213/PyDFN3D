
import numpy as np

from .DFN_Geometry.FracsGeo import FractureNetworks
from .Flow_Solver.FlowSolver import Flow_Solver
from .Visulization.Paraview_Writer import Paraview_Writer

class DFN_Model:
    """Main PyDFN3D class to handle all generic operations"""

    def __init__(self):
        """Creates a BEM object with some specific paramters
        
        Arguments
        ---------
        Dim            -- Model Dimension
        NumFracs       -- Number of Fractures

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        """
        self.fname=''
        self.FracNets=FractureNetworks()
        self.FlowSolver=None
    
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
        self.FracNets.loadFracs(fname,ftype)
    
    def setFlowSolver(self,solver='BEM',h_frac=1.0,h_trace=1.0,h_well=1.0):
        """Set up the flow solver

        1. create solver object
        2. convert DFN into mesh
        3. do meshing
        
        Arguments
        ---------
        solver  -- name of solver
        h_frac  -- mesh size of fracture plane, e.g. 1.0
        h_trace -- mesh size of trace, e.g. h_frac/5
        h_well  -- mesh size of well, e.g. h_frac/10

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        """
        assert self.FracNets.NumFracs>0,"Please load the discrete fracture network!"

        if(solver=='BEM'):
            self.FlowSolver=Flow_Solver(solver_type=solver,DDM=True)
            self.FlowSolver.link2DFN(self.FracNets,h_frac,h_trace,h_well)

    def setBoundaryConditions(self,fracID,DirichletBC=[],NeumannBC=[]):
        """Set up the flow solver

        1. create solver object
        2. convert DFN into mesh
        3. do meshing
        
        Arguments
        ---------
        DirichletBC  -- Dirichlet boundary condition for a specifc line, (edgeID/traceID/sourceID, value)
        NeumannBC    -- Neumann boundary condition for a specific line, (edgeID/traceID/sourceID, value)

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        """
        self.FlowSolver.setBCs(fracID,DirichletBC,NeumannBC)
    
    def setFracProperties(self,perm=None,aperature=None,perm_array=[],aperature_array=[]):
        """Set up the permeability and aperature for each fracture
        
        BEM algorihtm only support uniform or anistorpic permeability

        Arguments
        ---------
        perm,aperature     -- constant fracture perm/aperature for all fractures
        perm_array         -- fracture perm for all fractures
        aperature_array    -- fracture aperature for all fractures

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        """
        if(perm is not None): perm_array=[perm]*self.FracNets.NumFracs
        if(aperature is not None): aperature_array=[aperature]*self.FracNets.NumFracs
        
        self.FlowSolver.setFracProps(perm_array,aperature_array)
    
    def solveFlow(self,max_iters=100,tolerance=1e-5):
        """Solve the flow based on given parameters

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        """
        self.FlowSolver.solve(self.FracNets,max_iters,tolerance)
    
    def writeDFN(self,fname='Case1.vtp'):
        """write DFN geometry vtk file for visulization

        Usage
        ---------
        Case.writeDFN(fname='Case1.vtk')

        Author: Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        """
        Writer=Paraview_Writer(fname)
        if(self.FracNets.NumFracs>0):
            Writer.SetInputDFN(self.FracNets)
            Writer.Write()

    def writeSolution(self,mesh_input='PyGeoMesh.msh',fname='Case1.vtu'):
        """write solution vtk file for visulization based on given mesh

        Usage
        ---------
        Case.writeDFN(fname='Case1.vtk')

        Author: Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        """
        Writer=Paraview_Writer(fname)
        if(self.FlowSolver.SubProblems is not None or
           self.FlowSolver.Problem is not None):
            Writer.SetInputMesh(mesh_input)
            DFN_Mesh=Writer.DFN_MeshData

            Pressures=self.FlowSolver.getFracsSols(self.FracNets,DFN_Mesh)
            #! Velocity didn't implemented yet

            Writer.AppendPointData('Pressure',Pressures)

            Writer.Write()


