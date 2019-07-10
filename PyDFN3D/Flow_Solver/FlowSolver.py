import numpy as np

from .PyBEM2D import PyBEM2D as bem_solver 


class Flow_Solver:
    """Main flow solver class"""
    def __init__(self,solver_type='BEM-DDM',DDM=True):
        """Creates a flow solver object
        
        Currently only support BEM with domain decomposition method

        Arguments
        ---------
        solver_type    -- solver type, e.g. BEM, FEM ...
        DDM            -- domain decomposition method

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        """
        self.NumFracs=0
        self.solver_type=solver_type

        if(DDM):#Domain decomposition algorithm
            self.DDM=True
            self.SubProblems=None
        else:#Single domain
            self.Problem=None
        
    
    def link2DFN(self,FracNets,h_frac=1.0,h_trace=1.0,h_well=1.0):
        '''Convert DFN into solution domain for solvers 

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        '''
        NumDOFs=0

        if(self.solver_type=='BEM'): #BEM DDM solver
            self.SubProblems=[None]*FracNets.NumFracs
            for i in range(FracNets.NumFracs):
                print('------------Setting up Fracture %d------------'%(i+1))
                SubProblem=bem_solver.BEM2D()

                Boundary_vert, Trace_vert, Well_node = FracNets.Get2DFracGeo(i)
                SubProblem.set_Mesh(Boundary_vert,Trace_vert,Well_node,h_frac,h_trace,Type="Quad",geo_check=False)
                self.SubProblems[i]=SubProblem
                NumDOFs += SubProblem.Mesh.Ndof
        
        self.NumFracs=FracNets.NumFracs
        print("Total DOF=",NumDOFs)
    
    def setBCs(self,fracID,DirichletBC=[],NeumannBC=[]):
        '''Set up boundary conditions for generated bem mesh using element marker
           Dirichlet or Neumann boundary can be applied for each edge/trace/source
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        '''
        if(self.solver_type=='BEM'): #BEM DDM solver
            self.SubProblems[fracID].set_BoundaryCondition(DirichletBC=DirichletBC,NeumannBC=NeumannBC,update=1)
    
    def setFracProps(self,perm_array=[],aperature_array=[]):
        """Set up the permeability and aperature for each fracture
        
        BEM algorihtm only support uniform or anistorpic permeability

        Arguments
        ---------
        perm_array  -- Dirichlet boundary condition for a specifc line, (edgeID/traceID/sourceID, value)
        aperature_array    -- Neumann boundary condition for a specific line, (edgeID/traceID/sourceID, value)

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        """
        assert len(perm_array)==self.NumFracs,'Input perm array size is not equal to NumFracs!'
        assert len(aperature_array)==self.NumFracs,'Input aperature array size is not equal to NumFracs!'
        
        if(self.solver_type=='BEM'): #BEM DDM solver
            for fi in range(self.NumFracs):
                self.SubProblems[fi].SetProps(h=aperature_array[fi],k=perm_array[fi],miu=0.001)
    
    def solve(self,FracNets,max_iters=100,tolerance=1e-5):
        '''solve the flow based on solver given
        
        DDM solver requires connection table to link all subdomains together

        Arguments
        ---------
        solver_type    -- solver type, e.g. BEM, FEM ...

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        '''
        if(self.DDM):
            Intersection_table=FracNets.getFracsIntersectTable()
            DDM_solver=bem_solver.DDM_Solver(BEMobj=self.SubProblems,Intersection=Intersection_table,plot_mesh=0)
            DDM_solver.Solve_Iter(Method="P-DD",max_iters=max_iters,TOL=tolerance,alpha=0.5,opt=1) #P-DD algorithm

    def getFracPtsSols(self,FracNets,fracID=0,FracPts3D=[]):
        '''Get the pressure and velocity solution on a specific fracture

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        '''
        FracPts2D=FracNets.get2DFracPts(fracID,FracPts3D)
        #Get the solution on these points
        P,Vx,Vy=[],[],[]
        for Pts in FracPts2D:
            puv=self.SubProblems[fracID].get_Solution(list(Pts))
            P.append(puv[0])
            Vx.append(puv[1])
            Vy.append(puv[2])

        return np.asarray(P),np.asarray(Vx),np.asarray(Vy)
    
    def getFracsSols(self,FracNets,DFN_Mesh):
        '''Get the solution for all fratures
        #! Velocity didn't implemented yet which requires rotation as well

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        '''
        Pressures = np.zeros(DFN_Mesh.NumOfPts)
        for fi in range(DFN_Mesh.NumFracs):
            Pts3D=DFN_Mesh.Points[DFN_Mesh.FracPts[fi]]
            P,U,V=self.getFracPtsSols(FracNets,fracID=fi,FracPts3D=Pts3D) #U,V needs to rotate back into 3D
            for pi in range(len(P)):
                PtsID=DFN_Mesh.FracPts[fi][pi]
                Pressures[PtsID]=P[pi]
        
        return Pressures

    def showMesh(self,fracID=0):
        '''Plot/output the discretized fracture domain if possibile 

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        '''

        if(self.solver_type=='BEM'): #Each fracture domain can be plotted without Paraview
            self.SubProblems[fracID].plot_Mesh()

    def showSolution(self,fracID=0,p_range=None,v_range=None):
        '''Plot/output the solution if possibile 

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2019
        '''

        if(self.solver_type=='BEM'): #Each fracture domain can be plotted without Paraview
            self.SubProblems[fracID].PostProcess.plot_Solution(p_range=p_range,v_range=v_range)


