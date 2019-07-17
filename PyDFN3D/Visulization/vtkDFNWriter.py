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

import os
import numpy as np
try:
    import vtk
    import vtk.util.numpy_support as vtk_np
except ImportError:
    import warnings
    warnings.warn("No vtk module loaded.")

class vtkDFNWriter:
    def __init__(self, fname=None,FracNets=None):
        """Write DFN and its trace into two vtk polydata file

        Arguments
        ---------
        Dim            -- Model Dimension
        NumFracs       -- Number of Fractures
        NumInts        -- Number of Intersections
        NumClusters    -- Number of Clusters
        NumPtsFracs    -- Total number of points for fractures

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: May. 2019
        """
        #Set up the output folder
        self.fname=fname
        self.DFN_GeoData = FracNets

        #DFN VTK Object
        self.DFNData = vtk.vtkPolyData()
        self.DFNIntData = vtk.vtkPolyData()
        self.points = vtk.vtkPoints() #Shared Point for DFN and DFN intersection

        #Init the basic Data
        self.Build_DFN()
        self.Build_DFNIntersection()

    def Build_DFN(self):
        #Points
        self.points.SetNumberOfPoints(len(self.DFN_GeoData.Points))
        for i in range(len(self.DFN_GeoData.Points)):
            self.points.SetPoint(i, self.DFN_GeoData.Points[i][0],self.DFN_GeoData.Points[i][1],self.DFN_GeoData.Points[i][2])
        
        #Fractures
        Fractures = vtk.vtkCellArray()
        for i in range(self.DFN_GeoData.NumFracs):
            polygon = vtk.vtkPolygon()
            NumIDs = len(self.DFN_GeoData.Fractures[i])
            polygon.GetPointIds().SetNumberOfIds(NumIDs)
            for j in range(NumIDs):
                polygon.GetPointIds().SetId(
                    j, self.DFN_GeoData.Fractures[i][j])
            Fractures.InsertNextCell(polygon)

        self.DFNData.SetPoints(self.points)
        self.DFNData.SetPolys(Fractures)

        #Append the Fracture ID into Output
        FracID=vtk.vtkIntArray()
        FracID.SetNumberOfValues(self.DFN_GeoData.NumFracs)
        #intArray->SetNumberOfComponents(1);
        FracID.SetName("FractureID")
        for i in range(self.DFN_GeoData.NumFracs):
            FracID.SetValue(i,i)
        
        self.DFNData.GetCellData().SetScalars(FracID)
        self.DFNData.Modified()

        #Append frac props if applicable
        if(len(self.DFN_GeoData.perm_array)>0):
            Perm=vtk_np.numpy_to_vtk(num_array=self.DFN_GeoData.perm_array, deep=True, array_type=vtk.VTK_FLOAT)
            Perm.SetName('Permeability (m^2)')
            self.DFNData.GetCellData().AddArray(Perm)
        
        if(len(self.DFN_GeoData.aperature_array)>0):
            Aperature=vtk_np.numpy_to_vtk(num_array=self.DFN_GeoData.aperature_array, deep=True, array_type=vtk.VTK_FLOAT)
            Aperature.SetName('Aperature (m)')
            self.DFNData.GetCellData().AddArray(Aperature)

    def Build_DFNIntersection(self):
        #Intersection Lines
        Intersections = vtk.vtkCellArray()
        for i in range(self.DFN_GeoData.NumInts):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, self.DFN_GeoData.IntersectsLines[i][0])
            line.GetPointIds().SetId(1, self.DFN_GeoData.IntersectsLines[i][1])
            Intersections.InsertNextCell(line)

        #Append the Fracture ID into Output
        IntersectionID = vtk.vtkIntArray()
        IntersectionID.SetNumberOfValues(self.DFN_GeoData.NumInts)
        #intArray->SetNumberOfComponents(1);
        IntersectionID.SetName("IntersectionID")
        for i in range(self.DFN_GeoData.NumInts):
            IntersectionID.SetValue(i, i)

        self.DFNIntData.SetPoints(self.points)
        self.DFNIntData.SetLines(Intersections)
        self.DFNIntData.GetCellData().SetScalars(IntersectionID)
        self.DFNIntData.Modified()



    def Write(self):
        basename=os.path.splitext(os.path.basename(self.fname))[0]
        if not os.path.exists("Results"): os.makedirs('Results')
        path=os.path.join('Results',basename + '_vtk')

        #Write Fracture Network File
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(path + '.vtp')
        writer.SetInputData(self.DFNData)
        writer.Write()

        #Write Intersection File
        writer.SetFileName(path + '_intersection.vtp')
        writer.SetInputData(self.DFNIntData)
        writer.Write()

        return path + '_intersection.vtp'
    
