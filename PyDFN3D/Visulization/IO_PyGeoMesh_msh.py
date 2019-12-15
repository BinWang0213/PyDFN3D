import numpy as np

try:
    import meshio
    #from meshio import msh_io
    #from meshio import vtu_io
except ImportError:
    import warnings
    warnings.warn("No mesh-io module loaded. Please download from Github @ nschloe/meshio")

class DFN_Mesh:
    """Simple mesh object to storage DFN mesh data

    Programmer: Bin Wang(binwang.0213@gmail.com)
    Date: July. 2019
    """
    def __init__(self):
        #Mesh data
        self.NumOfPts=0
        self.NumOfElements=0

        self.Points=[]
        self.Elements=[]

        self.NumFracs=0
        self.FracPts=[]
        self.FracElements=[]

def read_PyGeoMeshFile_msh(fname):
    """Read PyGeoMesh msh file into DFN mesh class data
    Powerful meshio reader/writer https://github.com/nschloe/meshio
    
    PyGeoMesh will have following marked DFN object
    1. FRAC0_EDGE1
    2. FRAC0_TRACE1
    3. WELL0_SEGMENT
    4. FRAC_0

    Programmer: Bin Wang(binwang.0213@gmail.com)
    Date: July. 2019
    """

    #Read Gmsh *.msh data using meshio
    mesh = meshio.gmsh.read(fname)
    pts=mesh.points
    cells=mesh.cells
    cell_info=mesh.cell_data
    phys_names=mesh.field_data
    #pts, cells, _, cell_info, phys_names
    # Conver meshio_physname into [MarkerID,MarkerName] pair
    phys_names = {v[0]: k.rstrip() for k, v in phys_names.items()}


    #Storage the data into mesh class 
    MeshData=DFN_Mesh()
    
    MeshData.Points=pts
    MeshData.NumOfPts=len(MeshData.Points)
    MeshData.Elements = cells['triangle']
    MeshData.NumOfElements=len(MeshData.Elements)
    
    #Triangle Marker IDs, fracture plane
    MeshData.FracMarkerID = cell_info['triangle']['gmsh:physical']

    #Create MarkerName->MarkerID pair and count total number of fracs
    MeshData.FracName2Marker={}
    MeshData.Marker2Name = phys_names
    MeshData.NumFracs=0
    FracID=0
    for key, value in MeshData.Marker2Name.items():
        if(np.isin(key,MeshData.FracMarkerID)):
            MeshData.FracName2Marker[value]=key
        
        FracName = 'FRAC_' + str(FracID)
        if(FracName==value): 
            MeshData.NumFracs+=1
            FracID+=1

    #Get the pts id list for each fracture
    MeshData.FracPts = [None] * MeshData.NumFracs
    MeshData.FracElements = [None] * MeshData.NumFracs
    for fi in range(MeshData.NumFracs):
        FracName = 'FRAC_' + str(fi)
        FracMarker = MeshData.FracName2Marker[FracName]
        Pts_set = set()  # Using dict to remove duplicated point
        Eles_set = []
        for e in range(MeshData.NumOfElements):
            # Find the fracture elements
            if (FracMarker == MeshData.FracMarkerID[e]):
                #print('Element', e)
                #print(MeshData.Elements[e][0], MeshData.Elements[e][1],MeshData.Elements[e][2])
                Eles_set.append(MeshData.Elements[e])
                for j in range(3):
                    PtsID = MeshData.Elements[e][j]
                    Pts_set.add(PtsID)
        Pts_set = list(sorted(Pts_set))
        #print(Pts_set)
        MeshData.FracPts[fi] = Pts_set
        MeshData.FracElements[fi] = Eles_set
    

    return MeshData