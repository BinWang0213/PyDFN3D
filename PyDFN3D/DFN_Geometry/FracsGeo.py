#########################################################################
#       (C) 2017 Department of Petroleum Engineering,                   #
#       Univeristy of Louisiana at Lafayette, Lafayette, US.            #
#                                                                       #
# This code is released under the terms of the BSD license, and thus    #
# free for commercial and research use. Feel free to use the code into  #
# your own project with a PROPER REFERENCE.                             #
#                                                                       #
# PYDFN3D Code                                                          #
# Author: Bin Wang                                                      #
# Email: binwang.0213@gmail.com                                         #
# Reference: Wang, B., Feng, Y., Berrone, S., et al. (2017) Iterative   #
# Coupling of Boundary Element Method with Domain Decomposition.        #
# doi:                                                                  #
#########################################################################

import numpy as np
import time
import os

from PyDFN3D.Utils.Geometry import RotatePlanePts,LineLineDist_3D

from .IO_PyGeoMesh import read_PyGeoMeshFile,write_PyGeoMeshFile

class FractureNetworks:
    """Contains Fracture Networks Geometry and Intersection info"""

    def __init__(self):
        """Creates a BEM objecti with some specific paramters
        
        Arguments
        ---------
        Dim            -- Model Dimension
        NumFracs       -- Number of Fractures
        NumInts        -- Number of Intersections
        NumClusters    -- Number of Clusters
        NumPtsFracs    -- Total number of points for fractures
        
        Points         -- [Point Id]All of point coords for each fracture and intersection line
        Fractures      -- [Frac Id]Pts ID Table for each fracture, each fracture is polygon may have different number of ids
        ClusterIDs     -- [Cluster Id]Cluster ID Table for each fracture
        NumPtsFrac     -- [Frac Id]Array of number of nodes for each fracture (Rectangle=4, Triangle=3, Polygon=n)
        IntersectsLines-- [Intersect Id]Pts ID Table for each intersection, each intersectin line have two ids
        IntersectFracs -- [Intersect Id]Fracture ID Table for each intersection
        FracIntersects -- [Frac Id] Intersections id for each fracture
        ClusterFracs   -- [Cluster ID] Fractures id for each cluster

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: June. 2018
        """
        self.fname=None

        self.Dim=3
        self.NumFracs = 0
        self.NumInts = 0
        self.NumClusters = 0
        self.NumWells=0

        self.NumFracsPts=0

        self.Points = []
        self.Fractures = []
        self.NumPtsFrac = []
        
        self.IntersectsLines = []
        self.IntersectFracs = []
        self.FracIntersects = []
        self.WellFracIntersects = []
        
        self.ClusterFracs = []
    
    def loadFracs(self,fname,ftype='PyGeoMesh'):
        """ Read fracture networks from 3rdparty tool
        
        Currently support:
        1. open source DFN modeling tool: ADFNE 1.5 @ http://alghalandis.net
           script 3rdParty/FracGenerationTool/ADFNE2PyGeoMesh.m is written to output fracture network from ADFNE

        Usage
        ---------
        Fracture.loadFracs('./Datas/DFN30.data','PyGeoMesh')

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: Sep. 2018
        """
        self.fname=fname

        #Read fracs from file
        if(ftype=='PyGeoMesh'):#This is native format which includes all information needed
            read_PyGeoMeshFile(self, fname,delimiter=' ',commenter='--',blockender='/')

    def writeFracs(self,fname,ftype='PyGeoMesh'):
        """ Write fracture networks from 3rdparty tool

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: May. 2019
        """
        basename=os.path.splitext(os.path.basename(fname))[0]
        if not os.path.exists("Results"): os.makedirs('Results')
        path=os.path.join('Results',basename + '.dat')

        #Write fracs to file
        if(ftype=='PyGeoMesh'):
            write_PyGeoMeshFile(self, path,delimiter=' ',commenter='--',blockender='/')

    #*---------------------------------------- 
    #*          Fracture Data access
    #*----------------------------------------
    def Get3DFracGeo(self,FracID):
        '''Get the Fracture Geometry data
           
        Arguments
        ---------
        FracID -- Global Fracture ID

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: June. 2018
        '''
        #Collect fracture polygon vertex info
        Boundary_vert_3D = []
        for i in range(self.NumPtsFrac[FracID]):
            PtsID = self.Fractures[FracID][i]
            Boundary_vert_3D.append(self.Points[PtsID])

        #Collect Trace vetex info
        Intersect_vert_3D = []
        NumFracTraces = len(self.FracIntersects[FracID])
        for i in range(NumFracTraces):
            IntID = self.FracIntersects[FracID][i]
            for j in range(2):
                PtsID = self.IntersectsLines[IntID][j]
                Intersect_vert_3D.append(self.Points[PtsID])
        
        return Boundary_vert_3D, Intersect_vert_3D

    def Get2DFractureGeo(self, FracID):
        ''' Get a projected 2D fracture info from 3D fracture
            http://alghalandis.net/products/adfne/adfne15
        
        Arguments
        ---------
        Boundary_vert -- Fracture boundary vertices
        Trace_vert    -- Fracture intersection vertices
        3D            -- Original 3D Coords
        2D            -- Rotated 2D Coords

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: June. 2018
        '''

        RotateMat = []

        #Collect fracture polygon vertex & Trace vetex info
        Boundary_vert_3D, Intersect_vert_3D=self.Get3DFracGeo(FracID)

        #Rotate into 2D
        # Plane_Coords,Query_Points,TargetPlane
        Boundary_vert_2D = RotatePlanePts(
            Boundary_vert_3D, Boundary_vert_3D, '2D')        

        Intersect_vert_2D = RotatePlanePts(
            Boundary_vert_3D, Intersect_vert_3D, '2D')

        #Debug
        #check1 = RotatePlanePts(Boundary_vert_3D, Boundary_vert_2D, '3D')
        #check2 = RotatePlanePts(Boundary_vert_3D, Intersect_vert_2D, '3D')
        #print(np.sum(np.array(check1)-np.array(Boundary_vert_3D)))
        #print(np.sum(np.array(check2) - np.array(Intersect_vert_3D)))
        
        #Re-arrange vertex info into 2D format
        for i in range(self.NumPtsFrac[FracID]):
            Boundary_vert_2D[i] = Boundary_vert_2D[i][:-1]
        # Boundary Element Anticlock-wise, ADFNE clockwise, 
        # ADFNE is anticlock wise now, Bin, 09/25/2018
        # Boundary_vert_2D = list(reversed(Boundary_vert_2D))

        Trace_vert_2D = []
        NumFracTraces = len(self.FracIntersects[FracID])
        for i in range(NumFracTraces):
            temp = []
            for j in range(2):
                temp.append(Intersect_vert_2D[i * 2 + j][:-1])
            Trace_vert_2D.append(temp)
        
        Trace_vert_2D = np.array(Trace_vert_2D)
        if(np.isnan(np.sum(Trace_vert_2D))):
            print('bad boy',FracID)
        
        
        return Boundary_vert_2D, Trace_vert_2D

    def GetIntersectFracID(self, IntID, FracID=-1):
        '''Get the FracID based on intersection id and exclude the offered IntID (optional)
           
        Arguments
        ---------
        FracID -- Global Fracture ID
        IntID  -- Globa Intersection ID

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: June. 2018
        '''
        if(FracID == -1):
            return self.IntersectFracs[IntID]
        else:
            if(self.IntersectFracs[IntID][0] == FracID):
                return self.IntersectFracs[IntID][1]
            if(self.IntersectFracs[IntID][1] == FracID):
                return self.IntersectFracs[IntID][0]
            else:
                print("Error FracID is not belongs this Intersection!")

    def GetFracIntersectLocalEdgeID(self, FracID, IntID):
        '''Get the Intersection Local Edge ID on a Fracture

        Default Edge ID ordering:
        Boundary Edge (1,2,3,4...) Intersection Edge (5,6,7...)

        Arguments
        ---------
        FracID -- Global Fracture ID
        IntID  -- Globa Intersection ID
        EdgeID -- Local Edge ID on a Frac

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: June. 2018
        '''
        NumBoundaryEdge = len(self.Fractures[FracID])

        IntIDs = self.FracIntersects[FracID]
        try:
            return NumBoundaryEdge + IntIDs.index(IntID)
        except:
            print("Error! Intersection is not belongs this Fracture")
            return -1

            #for i in range(len(IntIDs)):
        #    if(IntID==IntIDs[i]):
        #        return NumBoundaryEdge+i
        #    else:
        #        print("Error! Intersection is not belongs this Fracture")

    