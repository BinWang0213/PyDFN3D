import numpy as np
import math
from .poly_point_isect import *
from numba import jit

@jit(nopython=True)
def calcDist(Pts0=(0,0),Pts1=(1,1)):
    '''Calculating distance of two points
    '''
    return np.sqrt((Pts1[0]-Pts0[0])**2+(Pts1[1]-Pts0[1])**2)


@jit(nopython=True)
def calcDist3D(Pts0=(0,0,0),Pts1=(1,1,1)):
    '''Calculating distance of two points
    '''
    return np.sqrt((Pts1[0]-Pts0[0])**2+(Pts1[1]-Pts0[1])**2+(Pts1[2]-Pts0[2])**2)

@jit(nopython=True)
def IsSamePoint3D(Pts0=(0,0,0),Pts1=(1,1,1)):
    '''Comparing two points
    '''
    if(abs(Pts1[0]-Pts0[0])+abs(Pts1[1]-Pts0[1])+abs(Pts1[2]-Pts0[2])<1e-10): return True
    return False


def centroid2D(Pts):
    '''Calculating centroid of a series of point cloud
    '''
    Pts = np.asarray(Pts)
    Npts=len(Pts)
    return np.sum(Pts[:,0])/Npts,np.sum(Pts[:,1])/Npts

def EndPointOnLine(Pts_a=(0,0),Pts_b=(0,1),Nseg=4,refinement="linspace",Endpoint=True):
    '''Genetrating endpoints along a line segment
       algorithm: point with a given distance along a line: x=x_Start+unit_vector_x*distance  y=y_Start+unit_vector_y*distance
    Arguments
        ---------
        Pts_a    -- The start-point.
        Pts_b    -- The end-point
        Npts     -- Number of endpoints 
        Nseg     -- Number of segments
        unit_vx  -- Unit vector for x coordinates
        unit_vy  -- Unit vector for y coordinates
        interval -- Segment interval
                    uniform    - - - - - - - - - -  (linspace)
                    refinement -- - -  -  -  - - -- (cosspace)
    '''
    Npts=Nseg+1
    Pts=np.zeros((Npts,2))
    length=calcDist(Pts_a,Pts_b)
    unit_vx=(Pts_b[0]-Pts_a[0])/length
    unit_vy=(Pts_b[1]-Pts_a[1])/length
    
    if (refinement=="linspace"):
        interval=np.linspace(0.0,length,Npts,endpoint=Endpoint)
        rinterval=np.linspace(length,0.0,Npts,endpoint=Endpoint)
    elif (refinement=="cosspace"):
        interval=cosspace(0.0,length,Npts,endpoint=Endpoint)
        rinterval=cosspace(length,0.0,Npts,endpoint=Endpoint)
    
    for i in range(Npts):
        Pts[i,0]=Pts_a[0]+interval[i]*unit_vx
        Pts[i,1]=Pts_a[1]+interval[i]*unit_vy
    
    return Pts.tolist()

def EndPointOnLineTwoSide(Pts_a=(0, 0), Pts_b=(0, 1), Nseg=4, refinement="linspace", Endpoint=True,offset=0.0005):
    '''Genetrating endpoints along two sides of a line segment
       The order of pts is  [Pts1_left,Pts1_right,Pts2_left.....]

       #https://stackoverflow.com/questions/10198747/python-numpy-how-can-i-simultaneously-select-all-odd-rows-and-all-even-column    
    
    Programmer: Bin Wang(binwang.0213@gmail.com)
    Date: June. 2017
    '''
    CenterLine=np.array(EndPointOnLine(Pts_a,Pts_b,Nseg,refinement,Endpoint))
    Pts = SplitLinePtsOnTwoSide(CenterLine, offset)

    return Pts


def SplitLinePtsOnTwoSide(Pts, offset=0.0005):
    '''Convert Line Pts along two sides of this line segment
       The order of pts is  [Pts1_left,Pts1_right,Pts2_left.....]
    
    Arguments
    ---------
    offset   -- The offset distance of two side points
    nx(y)    -- Unit normal vector of a line segments

    Programmer: Bin Wang(binwang.0213@gmail.com)
    Date: June. 2017
    '''
    NumPts = len(Pts)
    Pts_new = np.zeros((NumPts * 2, 2))
    Pts_a,Pts_b=Pts[0],Pts[-1]
    unit_nx, unit_ny = GetUnitNormalVec(Pts_a,Pts_b)

    Pts_new[::2, :] = Pts
    Pts_new[1::2, :] = Pts

    #Odd row, xy - offset*nx(y)
    Pts_new[::2, 0] = Pts_new[::2, 0] - offset * unit_nx
    Pts_new[::2, 1] = Pts_new[::2, 1] - offset * unit_ny

    #Even row, xy + offset*nx(y)
    Pts_new[1::2, 0] = Pts_new[1::2, 0] + offset * unit_nx
    Pts_new[1::2, 1] = Pts_new[1::2, 1] + offset * unit_ny

    return Pts_new

def SplitTwoSideArray(array):
    #Extend the properties array from center line to two side format
    #Used to streamline splitting streamlines at intersection
    #See SplitLinePtsOnTwoSide()
    arr_new = np.zeros((len(array) * 2))
    arr_new[::2] = array #fill Odd elements
    arr_new[1::2] = array#Fill Even elements
    return arr_new

def GetUnitNormalVec(Pts_a,Pts_b):
    #Compute unit normal vector based on two points on a line
    length = calcDist(Pts_a, Pts_b)
    unit_nx = (Pts_a[1] - Pts_b[1]) / length
    unit_ny = (Pts_b[0] - Pts_a[0]) / length

    return unit_nx,unit_ny


def Polygon2NodePair(PolygonPts):
    #round trip connect to a closed polygon
    #used for polygon format to line segment pair format
    lines = [(PolygonPts[i], PolygonPts[i+1]) for i in range(len(PolygonPts)-1)]
    lines = lines + [(PolygonPts[-1],PolygonPts[0])]
    return lines

def EndPointOnCircle(Origin=(0,0),Angle=(0,360),ab=(0,0),R=1,Nseg=4):
    '''Genetrating endpoints along a circle
    Arguments
        ---------
        Origin  -- The start-point.
        R       -- The end-point
        Npts    -- Number of endpoints 
        Nseg    -- Number of segments
    '''
    Npts=Nseg+1
    Pts=np.zeros((Npts,2))
    

    PI=3.141592653
    Start_Angle=Angle[0]*PI/180
    End_Angle=Angle[1]*PI/180

    major_R,minor_R=R,R
    if (ab[0]!=0 and ab[1]!=0): #ellipse shape
        major_R,minor_R=ab[0],ab[1]

    interval=np.linspace(Start_Angle, End_Angle, Npts)

    for i in range(Npts):
        Pts[i,0]=Origin[0]+np.cos(interval[i])*major_R
        Pts[i,1]=Origin[1]+np.sin(interval[i])*minor_R
    
    return Pts

def EndPointOnPolygon(Pts_poly,Nseg=20,closed=1):
    #create a series points along a polygon
    #closed=1 this is a closed polygon, closed=0 open curve
    
    Pts_segment=[]
    length_segment=[]
    NumSegment=len(Pts_poly)
    for i in range(NumSegment-1):
        Pts_segment.append([Pts_poly[i],Pts_poly[i+1]])
        length_segment.append(calcDist(Pts_poly[i],Pts_poly[i+1]))
        if(i==NumSegment-2 and closed==1):
            Pts_segment.append([Pts_poly[i+1],Pts_poly[0]])
            length_segment.append(calcDist(Pts_poly[i+1],Pts_poly[0]))
    
    Total_length=sum(length_segment)
    
    Pts=[]
    for i in range(len(length_segment)):
        Num_Pts=int(length_segment[i]/Total_length*Nseg)
        if(Num_Pts<2): Num_Pts=2
        temp=np.array(EndPointOnLine(Pts_segment[i][0],Pts_segment[i][1],Num_Pts))[:-1]
        Pts+=list(temp)
    
    return [list(s) for s in Pts]


def point_in_line(pts,A,B):
    #Test of point(pts) lies on line segment (AB)
    #https://stackoverflow.com/questions/328107/how-can-you-determine-a-point-is-between-two-other-points-on-a-line-segment?noredirect=1&lq=1
    
    epsilon=0.0000000001
    squaredlengthba=(A[0] - B[0])**2 + (A[1] - B[1])**2
    
    crossproduct = (pts[1] - A[1]) * (B[0] - A[0]) - (pts[0] - A[0]) * (B[1] - A[1])
    if abs(crossproduct) > epsilon : return False   # (or != 0 if using integers)

    dotproduct = (pts[0] - A[0]) * (B[0] - A[0]) + (pts[1] - A[1])*(B[1] - A[1])
    if dotproduct < 0 : return False

    squaredlengthba = (B[0] - A[0])*(B[0] - A[0]) + (B[1] - A[1])*(B[1] - A[1])
    if dotproduct > squaredlengthba: return False

    return True

def point_in_line3D(pts,A,B):
    #Check C is on infinite line AB
    #https://stackoverflow.com/questions/328107/how-can-you-determine-a-point-is-between-two-other-points-on-a-line-segment
    dist=Point2Line_3D(pts,A,B)
    if(dist<1e-10):
        return True
    return False

def Point2Line_3D(p0,p1,p2):
    #http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    """ 3D-Returns the perpendicular distance between point (p0) and line(p1,p2):
        p0-point 0 fmt(1,2,3)
        p1-start point of line fmt(1,2,3)
        p2-end point of line fmt(1,2,3)
    """
    pts0=np.asarray(p0)
    pts1=np.asarray(p1)
    pts2=np.asarray(p2)
    
    cross=np.cross(pts2-pts1,pts1-pts0)
    d=np.linalg.norm(cross)/np.linalg.norm(pts2-pts1)

    return d

def LineLineDist_3D(p1,p2,p3,p4):
    #https://vicrucann.github.io/tutorials/3d-geometry-algorithms/
    #http://paulbourke.net/geometry/pointlineplane/L3D.py
    #http://geomalgorithms.com/a07-_distance.html#dist3D_Segment_to_Segment
    #https://www.geometrictools.com/Documentation/DistanceLine3Line3.pdf
    p1,p2,p3,p4=np.asarray(p1),np.asarray(p2),np.asarray(p3),np.asarray(p4)

    u = p1 - p2
    v = p3 - p4
    w = p2 - p4
    
    a = np.dot(u,u)
    b = np.dot(u,v)
    c = np.dot(v,v)
    d = np.dot(u,w)
    e = np.dot(v,w)
    D = a*c - b*b
    sD = D
    tD = D

    tol=1e-8

    if(D<tol):#the lines are almost parallel
        sN=0.0
        sD=1.0
        tN=e
        tD=c
    else:
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if (sN < 0.0):   # sc < 0 => the s=0 edge is visible       
            sN = 0.0
            tN = e
            tD = c
        elif (sN > sD): # sc > 1 => the s=1 edge is visible
            sN = sD
            tN = e + b
            tD = c
    
    if (tN < 0.0):            # tc < 0 => the t=0 edge is visible
        tN = 0.0
        # recompute sc for this edge
        if (-d < 0.0): sN = 0.0
        elif (-d > a): sN = sD
        else:
            sN = -d
            sD = a
    elif (tN > tD):       #tc > 1 => the t=1 edge is visible
        tN = tD
        #recompute sc for this edge
        if ((-d + b) < 0.0):
            sN = 0
        elif ((-d + b) > a):
            sN = sD
        else:
            sN = (-d + b)
            sD = a
    
    #finally do the division to get sc and tc
    if(abs(sN) < tol): sc = 0.0
    else: sc = sN / sD
    
    if(abs(tN) < tol): tc = 0.0
    else: tc = tN / tD
    
    Pa=p2+sc*u
    Pb=p4+tc*v
    dist=calcDist3D(Pa,Pb)

    return dist,Pa,Pb


def angle_between(v1, v2):
    #http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
    """ 3D-Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# Corner refine space function-for boundary element discretation
def cosspace(st,ed,N,endpoint=True):
    """
    Auto line segment refinement at end point
    e.g. --- - -  -  -  - - ---
    """
    #N=N+1
    AngleInc=np.pi/(N-1)
    CurAngle = AngleInc
    space=np.linspace(0,1,N,endpoint=endpoint)
    space[0]=st
    for i in range(N-1):
        space[i+1] = 0.5*np.abs(ed-st)*(1 - np.cos(CurAngle))
        CurAngle += AngleInc
    if ed<st:
        space[0]=ed
        space=space[::-1]

    return space


#Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
@jit(nopython=True)
def Point2Segment (px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    #LineMag = lineMagnitude(x1, y1, x2, y2)
    LineMag = math.hypot(x2 - x1,y2 - y1)


    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine
 
    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)
 
    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        #ix = lineMagnitude(px, py, x1, y1)
        ix=math.hypot(x1-px,y1-py)
        #iy = lineMagnitude(px, py, x2, y2)
        iy=math.hypot(x2-px,y2-py)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        #DistancePointLine = lineMagnitude(px, py, ix, iy)
        DistancePointLine = math.hypot(ix-px,iy-py)
 
    return DistancePointLine

@jit(nopython=True)
def LineSegIntersect(xa,ya,xb,yb,xc,yc,xd,yd):
    #Algorithm from http://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    #Test whether 2 line segment are intersected
    ccw_ACD=(yd-ya)*(xc-xa) > (yc-ya)*(xd-xa)
    ccw_BCD=(yd-yb)*(xc-xb) > (yc-yb)*(xd-xb)
    ccw_ABC=(yc-ya)*(xb-xa) > (yb-ya)*(xc-xa)
    ccw_ABD=(yd-ya)*(xb-xa) > (yb-ya)*(xd-xa)
    return ccw_ACD != ccw_BCD and ccw_ABC != ccw_ABD

@jit(nopython=True)
def LineSegIntersect2(line1=((0,0),(1,0)),line2=((1,0),(1,1))):
    #test of line segment intersection
    xa,ya=line1[0][0],line1[0][1]
    xb,yb=line1[1][0],line1[1][1]
    xc,yc=line2[0][0],line2[0][1]
    xd,yd=line2[1][0],line2[1][1]
    return LineSegIntersect(xa,ya,xb,yb,xc,yc,xd,yd)

def LineIntersect(line1=((0,0),(1,0)),line2=((1,0),(1,1))):
    #Find the intersection of two lines AB-CD, even their segment doesn't intersect
    #Line inpur port version of Line2Line intersection
    xa,ya=line1[0][0],line1[0][1]
    xb,yb=line1[1][0],line1[1][1]
    xc,yc=line2[0][0],line2[0][1]
    xd,yd=line2[1][0],line2[1][1]
    return Line2Line(xa,ya,xb,yb,xc,yc,xd,yd)

def Line2Line(xa,ya,xb,yb,xc,yc,xd,yd):
    #http://alienryderflex.com/intersect/
    #Find the intersection of two lines AB-CD, even their segment doesn't intersect
    
    #(1) Translate the system so that point A(xa,ya) is on the origin.
    xb-=xa
    yb-=ya
    xc-=xa
    yc-=ya
    xd-=xa
    yd-=ya
    #(2) Distance Ab
    distAB=np.sqrt(xb*xb+yb*yb)
    #(3) Rotate the system so that point B is on the positive X axis.
    theCos=xb/distAB
    theSin=yb/distAB
    newX=xc*theCos+yc*theSin
    yc  =yc*theCos-xc*theSin
    xc=newX
    newX=xd*theCos+yd*theSin
    yd  =yd*theCos-xd*theSin
    xd=newX
    #determine parallel
    if (yc==yd):
        print("Error:Parallel line")
        return 0
    #(4) Intersection point
    ABpos=xd+(xc-xd)*yd/(yd-yc)
    x_intsect=xa+ABpos*theCos
    y_intsect=ya+ABpos*theSin
    
    return x_intsect,y_intsect


def IntersectLines(lines):
    debug=0
    #1. Convert list to tuple (if so)
    lines = [tuple(x) for x in lines]
    num_lines = len(lines)

    if(num_lines==1):
        if(debug):
            print("%s Intersections Found!" % (num_isect))
        return lines

    #2. Intersection finding
    intersection = isect_segments(lines)
    num_isect = len(intersection)
    if(debug):
        print("%s Intersections Found:" % (num_isect))
    return intersection

def Split_IntersectLines(lines):
        """Detect and Subdivide intersected line based on fast bentley-ottmann algorithm-bug fixed
    
            Reference Lib: https://github.com/ideasman42/isect_segments-bentley_ottmann version:01/2018
            
            Example:
            import poly_point_isect #input line segment must be tuple format
            
            lines=[
            ((0.000000, 1.000000), (0.000000, -1.000000)),
            ((1.000000, 0.300000), (-1.000000, 0.300000)),
            ((1.000000, 0.800000), (-1.000000, 0.800000)),
            ((-1, -1), (1, 1))
            ]
            isect = poly_point_isect.isect_segments(lines)
            
            Arguments
            ---------
            num_isect       -- Number of intersection points
            num_lines       -- Number of lines 
            isect_lines     -- Intersection points for each line [line index][intersection points]
            new_lines       -- Collection of subdived lines

            Programmer: Bin Wang (binwang.0213@gmail.com)
            Date: July. 2017
        """
        debug = 1
        #1. Convert list to tuple (if so)
        lines = [tuple(x) for x in lines]
        num_lines = len(lines)

        if(num_lines==1):
            if(debug):
                print("%s Intersections Found!" % (0))
            return lines

        #2. Intersection finding
        intersection = isect_segments(lines)
        num_isect = len(intersection)

        if(num_isect==0):
            if(debug):
                print("%s Intersections Found!" % (num_isect))
            return lines

        #3. Subdivid the intersected line into segments
        new_lines = []

        isect_lines = [[] for j in range(num_lines)]  # 2D list

        for i in range(num_isect):  # finding intersection for each line
            for j in range(num_lines):
                if (point_in_line(intersection[i], lines[j][0], lines[j][1]) == True):
                    isect_lines[j].append(intersection[i])

        # correct the intersction order, e.g. [(0.5,0.5),(0.1,0.1)]->[(0.1,0.1),(0.5,0.5)]

        def sort_by_sum(a):
            #https://stackoverflow.com/questions/7235785/sorting-numpy-array-according-to-the-sum
            #https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
            return a[np.sum(a, axis=1).argsort()]

        for j in range(len(isect_lines)):
            if (len(isect_lines[j]) >= 1):
                temp = np.array(isect_lines[j])
                isect_lines[j] = sort_by_sum(temp)

        for j in range(num_lines):  # add intersected lines to [new_lines]
            if (len(isect_lines[j]) >= 1):
                new_lines.append((lines[j][0], tuple(isect_lines[j][0])))
                for k in range(len(isect_lines[j]) - 1):  # intersection points
                    new_lines.append(
                        (tuple(isect_lines[j][k]), tuple(isect_lines[j][k + 1]))
                        )
                new_lines.append((tuple(isect_lines[j][-1]), lines[j][1]))

        # add stand alone lines at the end of [new_lines]
        for j in range(num_lines):
            if (len(isect_lines[j]) == 0):
                new_lines.append([lines[j][0], lines[j][1]])

        if(debug):
            print("%s Intersections Found:" % (num_isect))
            #print(intersection)
            #print("New Traces")
            #for t in range(len(new_lines)):
                #print("Trace", t + 1, new_lines[t])

        return new_lines

def region_line(xa,ya,xb,yb,dist):
    #Calculate the given distance bounding box around a line segment
    #This generally used with point_in_domain function
    PI=np.pi
    length=np.sqrt((xb-xa)**2+(yb-ya)**2)
    theta=np.arctan2((yb-ya),(xb-xa))
    
    X_outline=[xb+dist*np.cos(PI/4+theta),xa+dist*np.cos(3*PI/4+theta),xa+dist*np.cos(5*PI/4+theta),xb+dist*np.cos(7*PI/4+theta)]
    Y_outline=[yb+dist*np.sin(PI/4+theta),ya+dist*np.sin(3*PI/4+theta),ya+dist*np.sin(5*PI/4+theta),yb+dist*np.sin(7*PI/4+theta)]
    return X_outline,Y_outline

def point_in_domain(x,y,X,Y):
    #Test of whether a point is within a polygon
    #Format of X,Y
    #  X=[10,20,20,10]
    #  Y=[10,10,20,20]
    poly=[]
    for i in range(len(X)):
        poly.append((X[i],Y[i]))
    
    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


def point_in_panel(x,y,panels):
    #Test of a point in a boundary element panel, this is for plotting 
    X=np.zeros(2*len(panels)+1)
    Y=np.zeros(2*len(panels)+1)
    for i, pl in enumerate(panels):
        X[2*i]=pl.xa
        X[2*i+1]=pl.xc
        Y[2*i]=pl.ya
        Y[2*i+1]=pl.yc
    X[-1]=X[0]
    Y[-1]=Y[0]

    poly=[]
    for i in range(len(X)):
        poly.append((X[i],Y[i]))
    
    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


def GetPtsInPolygon(Pts_e,resolution=10):
    from matplotlib import path

    #Find the domain extend
    domain_min=(min(np.asarray(Pts_e)[:,0]),min(np.asarray(Pts_e)[:,1]))
    domain_max=(max(np.asarray(Pts_e)[:,0]),max(np.asarray(Pts_e)[:,1]))   

    #Define the polygon
    Polygon = path.Path(Pts_e+[Pts_e[0]])

    #Sampling points over the retangle region
    xmin, ymin = domain_min[0], domain_min[1]
    xmax, ymax = domain_max[0], domain_max[1]

    #Genetrate the internal sampling points
    pad=0.05

    xi, yi = np.linspace(xmin+pad, xmax-pad, resolution), np.linspace(ymin+pad, ymax-pad, resolution)
    X, Y = np.meshgrid(xi,yi)  # generates a mesh grid
    Pts = np.array([(x, y)for x, y in zip(X.flatten(), Y.flatten())]) # Convert to 1D array
    BoolIndex=np.array(Polygon.contains_points(Pts))  # No need check points anymore
    Pts = Pts[BoolIndex == True]

    #Genetrate the sampling pint along the boundary
    Pts = list(Pts) + EndPointOnPolygon(Pts_e, Nseg=resolution * 4)
    Pts = np.array(Pts)

    import matplotlib.tri as tri
    
    #Find the unwanted triangular
    # When triplot is used this should be activated
    x,y = np.array(Pts[:, 0]),np.array(Pts[:, 1])
    triang = tri.Triangulation(x, y)
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    Tri_Pts = np.array([(x, y)for x, y in zip(xmid, ymid)])
    mask = Polygon.contains_points(Tri_Pts)
    mask = np.invert(mask)
    triang.set_mask(mask)
    
    #triang=0
    return Pts, triang


@jit(nopython=True)
def ShapeFunc_Weight(Pts,Pts_a,Pts_b,order=1):
    #Interpolate the solution on a 1D element
    #order=0-constant,1-linear,2-quadratic
    #Pts-Query point Ele_Pts-Element node points
    
    phi=[] #weights of shape function
    local=-1+2*calcDist(Pts,Pts_a)/calcDist(Pts_a,Pts_b)
    
    if(order==1):#Two nodes
        phi.append(0.5*(1-local))
        phi.append(0.5*(1+local))
    if(order==2):#Three nodes
        phi.append(0.5*local*(local-1))
        phi.append((1-local)*(1+local))
        phi.append(0.5*local*(local+1))
    
    return phi

@jit(nopython=True) #Numba acclerated
def Interp_Nonconforming(Pts_query,Pts,Vals,order=1):
    """General 1D interpolation along a line segments(Pts) function using shape funciton
       Element-wise shape function is compatible with FEM/BEM discretation
        
    Arguments
    ---------
    order     -- Element order in interpolation side
                 order=0 Const element interpolation
                 order=1 linear interpolation between two nodes
                 order=2 quadratic interpolation between three nodes
    Pts       -- Coords of nodes in interpolation side e.g. [(0,0),(1,0),(2.0)...]
    Vals      -- Values of nodes in interpoaltion side, same size with Pts e.g [0,1,2,3...]
    Pts_query -- Query points for interpolation e.g. [(0.2,0),(0.5,0),(1.5.0)...]
        
    Author:Bin Wang(binwang.0213@gmail.com)
    Date: Aug. 2017
    """
    Num_query=len(Pts_query)
    Num_nodes=len(Pts)
    
    Vals_interp=np.zeros(Num_query)

    if(order==0):
        Num_element=(int)(Num_nodes)-1
        Vals_interp=np.zeros(Num_query-1)
    if(order==1):
        Num_element=(int)(Num_nodes/order)-1
    if(order==2):
        Num_element=(int)(Num_nodes/order)
    
    
    
    for i in range(Num_query):#For each query points
        for j in range(Num_element):
            if(order==0):#constant interpolation
                Pts_a=Pts[j]
                Pts_b=Pts[j+1]
                if(point_on_line(Pts_query[i],Pts_a,Pts_b)):#Found the element it belongs to
                    #ShapeFunc_Interpolation1([0.2,0.3],Ele_Pts)
                    Vals_interp[i]=Vals[j]

            if(order==1):#linear interpolation
                Pts_a=Pts[j]
                Pts_b=Pts[j+1]
                if(point_on_line(Pts_query[i],Pts_a,Pts_b)):#Found the element it belongs to
                    #ShapeFunc_Interpolation1([0.2,0.3],Ele_Pts)
                    phi=ShapeFunc_Weight(Pts_query[i],Pts_a,Pts_b,order=order)
                    Vals_interp[i]=phi[0]*Vals[j]+phi[1]*Vals[j+1]
                    
            if(order==2):#quadratic interpolation
                Pts_a=Pts[2*j]
                Pts_c=Pts[2*j+1]
                Pts_b=Pts[2*j+2]
                if(point_on_line(Pts_query[i],Pts_a,Pts_b)):#Found the element it belongs to
                    #ShapeFunc_Interpolation1([0.2,0.3],Ele_Pts)
                    phi=ShapeFunc_Weight(Pts_query[i],Pts_a,Pts_b,order=order)
                    Vals_interp[i]=phi[0]*Vals[2*j]+phi[1]*Vals[2*j+1]+phi[2]*Vals[2*j+2]
    return Vals_interp


@jit(nopython=True) #23X faster than original one
def point_on_line(pts,A,B):
    #Test of point(pts) lies on line segment (AB)-fast
    #https://stackoverflow.com/questions/328107/how-can-you-determine-a-point-is-between-two-other-points-on-a-line-segment?noredirect=1&lq=1
    
    epsilon=0.0000001
    squaredlengthba=(A[0] - B[0])**2 + (A[1] - B[1])**2
    
    crossproduct = (pts[1] - A[1]) * (B[0] - A[0]) - (pts[0] - A[0]) * (B[1] - A[1])
    if abs(crossproduct) > epsilon : return False   # (or != 0 if using integers)

    dotproduct = (pts[0] - A[0]) * (B[0] - A[0]) + (pts[1] - A[1])*(B[1] - A[1])
    if dotproduct < 0 : return False

    squaredlengthba = (B[0] - A[0])*(B[0] - A[0]) + (B[1] - A[1])*(B[1] - A[1])
    if dotproduct > squaredlengthba: return False

    return True


def line_leftright(A,B,dist):
    #Calculate the left and right central points with given distance(dist) for a line segment(A->B)
    #  B
    #  | 
    #. | .
    #  |
    #  A
    # used for finding the neighboring element of a line segments
    pts_left=np.zeros(2)
    pts_right=np.zeros(2)
    xc=(A[0]+B[0])/2
    yc=(A[1]+B[1])/2

    length=calcDist(A,B)

    pts_left[0]=xc+dist*(A[1]-B[1])/length
    pts_left[1]=yc+dist*(B[0]-A[0])/length
    pts_right[0]=xc-dist*(A[1]-B[1])/length
    pts_right[1]=yc-dist*(B[0]-A[0])/length
    
    return pts_left,pts_right


def GaussLib(Gaussorder):

    if (Gaussorder==2):
        GI=[0.0,-0.5773502691896257,0.5773502691896257]
        WC=[0.0, 1.,1.]
    if (Gaussorder==3):
        GI=[0.0,-0.7745966692414834 ,0.,0.7745966692414834]
        WC=[0.0, 0.5555555555555556,0.8888888888888889,0.5555555555555556] 
    if (Gaussorder==4):
        GI=[0.0,-0.8611363115940526,-0.3399810435848563,0.3399810435848563,0.8611363115940526]
        WC=[0.0,0.3478548451374538,0.6521451548625461,0.6521451548625461,0.3478548451374538]
    if (Gaussorder==5):
        GI=[0.0,-0.9061798459386640 ,-0.5384693101056831,  0.  ,-0.5384693101056831, 0.9061798459386640]
        WC=[0.0,0.2369268850561891 , 0.4786286704993665 , 0.5688888888888889 , 0.4786286704993665 , 0.2369268850561891] 
    if (Gaussorder==6):
        GI=[0.0,-0.9324695142031521 ,-0.6612093864662645, -0.2386191860831969 , 0.2386191860831969 , 0.6612093864662645 , 0.9324695142031521]
        WC=[0.0,0.1713244923791704 ,0.3607615730481386 ,0.4679139345726910 ,0.4679139345726910, 0.3607615730481386 , 0.1713244923791704]
    if (Gaussorder==7):
        GI=[0.0,-0.9491079123427585, -0.7415311855993945, -0.4058451513773972 , 0.     
            ,0.4058451513773972, 0.7415311855993945 ,0.9491079123427585]
        WC=[0.0,0.1294849661688697 , 0.2797053914892766 , 0.3818300505051189 , 0.4179591836734694 
            , 0.3818300505051189 , 0.2797053914892766,0.1294849661688697]
    if (Gaussorder==8):
        GI=[0.0,-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498 
            , 0.1834346424956498 , 0.5255324099163290, 0.7966664774136267 , 0.9602898564975363]
        WC=[0.0,0.1012285362903763 , 0.2223810344533745 ,0.3137066458778873 , 0.3626837833783620
            ,0.3626837833783620 ,0.3137066458778873,0.2223810344533745 , 0.1012285362903763]
    if (Gaussorder==9):
        GI=[0.0,-0.9681602395076261, -0.8360311073266358 ,-0.6133714327005904 ,-0.3242534234038089, 0.     
            ,0.3242534234038089,0.6133714327005904 , 0.8360311073266358 , 0.9681602395076261]
        WC=[0.0,0.0812743883615744 , 0.1806481606948574 , 0.2606106964029354 ,  0.3123470770400029 , 0.3302393550012598 
            , 0.3123470770400029, 0.2606106964029354  , 0.1806481606948574 , 0.0812743883615744]
    if (Gaussorder==10):
        GI=[0.0,-0.9739065285171717 ,-0.8650633666889845 ,-0.6794095682990244, -0.4333953941292472, -0.1488743389816312 
            , 0.1488743389816312,0.4333953941292472 , 0.6794095682990244 , 0.8650633666889845 , 0.9739065285171717]
        WC=[0.0,0.0666713443086881 , 0.1494513491505806 ,0.2190863625159820 , 0.2692667193099963 , 0.2955242247147529 
            , 0.2955242247147529,0.2692667193099963 , 0.2190863625159820 , 0.1494513491505806 , 0.0666713443086881]
    
    return GI,WC

def Global2Iso(length_sub,length,Nodes):
    #This function mapping global (0,s) line element to local(-1,1) coordinates
    #-------------
    #-1    0     1   Local
    #0  L1  L2 L3    Global
    #Nodes, 3-quadrautic element 2-linear element
    N=len(length_sub)
    local=np.zeros(N+1)
    temp=0
    local[0]=-1
    for i in range(N):
        temp=temp+length_sub[i]
        #print(temp)
        local[i+1]=-1+2*temp/length
    return local
    


def Subdivision(xi, yi, X1, Y1, X2, Y2, TOL, sub_trigger, bd_int):
    #The element subdivision algorithm is based on Gao et al. Page80
    '''
    This algorithm is based on adaptive gauss point selection with equally length element of subdivision
    ----------Variable definition-------
    length- the length of target element
    mindist- the minimum distance from point(xi,yi) to element segment(X1,Y1)-(X2,Y2)
    p_prime-geometory coefficient- 2D problem internal point lamada=2
    TOL-required accurancy for near singular integration
    bd_inter- Position ID 1-boundary 0-interal domain
    
    NG_required-required gauss point for accurate integration
    No_sub- number of subdivision in this element
    length_sub-minimum euqally distance of subdivision element 
    Sub_endpoint-series of end point for sub-element
    '''
    Subdiv_trigger = 0
    length = np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)
    #print(length)
    #print(X1,Y1,X2,Y2)
    mindist = Point2Segment(xi, yi, X1, Y1, X2, Y2)
    p_prime=0
    if bd_int == 'boundary':  #boundary
        p_prime = np.sqrt(
            1 * 2 / 3 + 2 / 5)  #1.316561 Gao et al book. 2D problem Internal lamada=2  Boundary lamada=1
    if bd_int == 'internal':  #internal domain
        p_prime = np.sqrt(
            2 * 2 / 3 + 2 / 5)  #1.316561 Gao et al book. 2D problem Internal lamada=2  Boundary lamada=1
    #print('length:%8f Rmin:%8f'%(length,mindist))
    if length > 3.9 * mindist:
        length_temp = 3.9 * mindist  #special trick to trigger subdivision
    else:
        length_temp = length
    NG_required = p_prime * np.log(TOL / 2) * 0.5 / np.log(length_temp / 4 / mindist)
    #print('Calculated GausPoint:%s'%(NG_required))
    if sub_trigger == 1:  #integration of subelement requires more gauss point than standard algorithm
        if bd_int == 'boundary':
            NG_required = int(NG_required) + 2
        if bd_int == 'internal':
            NG_required = int(NG_required) + 5
    elif sub_trigger == 0:  #standard gauss point requirement
        if bd_int == 'boundary':
            NG_required = math.ceil(NG_required) + 1
        if bd_int == 'internal':
            NG_required = math.ceil(NG_required) + 2
    #print('Required GausPoint:%s'%(NG_required))

    #sub-interval length and division
    if NG_required > 10:  #subdivision trigger, 10=maximum gauss point in subdivision
        NG_required = 10  #maximum number of gauss point 
        length_sub = 4. * mindist * (TOL / 2.) ** (0.5 * p_prime / NG_required)  #minimum length of subdivision element
        #print('Required sublength:%s'%(length_sub))
        #print('Element length:%s'%(length))
        if length_sub >= length:  #if the required length > remainning length
            length_sub = length
        #No_sub=int(length/length_sub+0.95)
        #length_sub=length/No_sub
        Subdiv_trigger = 1
    elif NG_required < 2:  #minimum gauss point=2 in sub division
        NG_required = 2
        length_sub = 4. * mindist * (TOL / 2.) ** (0.5 * p_prime / NG_required)
        No_sub = int(length / length_sub + 0.95)
        length_sub = length / No_sub
    else:  #No sub-division is trigger
        No_sub = 1
        length_sub = length

    #print('Sub length:%s No_Gauss:%s\n'%(length_sub,NG_required))
    return Subdiv_trigger, NG_required, length_sub



@jit(nopython=True)
def Line2Local(Pts):
    #Convert 2d line coords to 1D local coords[0,1]
    Npts=len(Pts)
    local=np.zeros(Npts)
    
    Length_total=calcDist(Pts[0],Pts[-1])
    
    for i in range(Npts):
        local[i]=calcDist(Pts[0],Pts[i])/Length_total
        
    
    return local


def RotatePlanePts(PlaneCoords_3D, Query_Points,TargetPlane='2D'):
    '''Roate points on a 3D plane into 2D and backforth
    
    Rotation between two unit axis
    Rotation 2D Plane back to 3D requires the original 3D coords
    https://math.stackexchange.com/questions/856666/how-can-i-transform-a-3d-triangle-to-xy-plane
    
    Author:Bin Wang(binwang.0213@gmail.com)
    Date: June. 2018
    '''
    
    #1. Compuate the axis vectors (U,V,W) of Plane 
    AB = PlaneCoords_3D[1] - PlaneCoords_3D[0]
    AC = PlaneCoords_3D[len(PlaneCoords_3D) - 1] - PlaneCoords_3D[0]
    N=np.cross(AB,AC)

    #print(AB,AC,N)
    
    U = AB/np.linalg.norm(AB)
    W = N/np.linalg.norm(N)
    V = -np.cross(U, W)

    #print(U,W,V)
    #2. Compute the Rotation matrix
    RotationMat_3D = np.array([U, V, W]).T
    RotationMat_2D = np.linalg.inv(RotationMat_3D)
    
    #print(RotationMat_3D)
    #print("\n")
    #print(RotationMat_2D)

    #3. Rotate 3D Coords and get the 2D Coords
    NewCoords = []
    size = len(Query_Points)
    
    if(TargetPlane == "2D"):
        #3.1 Normalized the coords by setting Coords[0] as origin
        NormCoords = []
        for i in range(size):
            NormCoords.append(Query_Points[i] - PlaneCoords_3D[0])

        #3.2 Rotate Coords
        for i in range(size):
                NewCoords.append(np.dot(RotationMat_2D, NormCoords[i]))

        
    if(TargetPlane == "3D"):#Convert 2D Plane back to 3D
        for i in range(size):
            NewCoords.append(np.dot(RotationMat_3D, Query_Points[i]))
            NewCoords[i] = NewCoords[i] + PlaneCoords_3D[0]


    return NewCoords

def RotateMat(plane3D):
    '''Simplified from RotatePlanePts

    Get the rotation matrix from 3D to 2D
    
    Author:Bin Wang(binwang.0213@gmail.com)
    Date: June. 2018
    '''

    #1. Compuate the axis vectors (U,V,W) of Plane 
    AB = plane3D[1] - plane3D[0]
    AC = plane3D[-1] - plane3D[0]
    N=np.cross(AB,AC)

    #print(AB,AC,N)
    
    U = AB/np.linalg.norm(AB)
    W = N/np.linalg.norm(N)
    V = -np.cross(U, W)

    #print(U,W,V)
    #2. Compute the Rotation matrix
    RotationMat_3D = np.array([U, V, W]).T
    RotationMat_2D = np.linalg.inv(RotationMat_3D)

    return RotationMat_2D,RotationMat_3D

def RotateTo2D(RotationMat_2D,plane,Query_Points):
    '''Simplified from RotatePlanePts

    Rotate 3D point into 2D based on given rotation matrix
    
    Author:Bin Wang(binwang.0213@gmail.com)
    Date: May. 2019
    '''
    NewCoords = []

    #3.1 Normalized the coords by setting Coords[0] as origin
    #3.2 Rotate Coords
    NewCoords=[np.dot(RotationMat_2D, p-plane[0]) for  p in Query_Points]

    return np.array(NewCoords)

def RotateTo3D(RotationMat_3D,plane,Query_Points):
    '''Simplified from RotatePlanePts

    Rotate 2D point back into 3D based on given rotation matrix
    
    Author:Bin Wang(binwang.0213@gmail.com)
    Date: May. 2019
    '''
    NewCoords = [np.dot(RotationMat_3D, p)+plane[0] for p in Query_Points]
    return np.array(NewCoords)


def intersectLinePlane(line,polygon):
    #LineHitPlane https://gist.github.com/TimSC/8c25ca941d614bf48ebba6b473747d72
    #Point in 3D Polygon
    planeNormal=np.cross(polygon[1]-polygon[0],polygon[-1]-polygon[0])
    planePoint=polygon[0]
    rayDirection=line[1]-line[0]
    rayPoint=line[0]
    
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < 1e-8:
        print("!!no intersection or line is within plane")
        return None

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    if(si<0 or si>1):#Intersection point is out of line segment
        return None
    return Psi



from matplotlib.path import Path

def intersectLinePolygon3D(line,poly):
    '''Intersect a 3d line and a 3d polygon

    line is a [2x3] array
    poly is a [Nx3] array

    1. find line plane intersection point
    2. project 3d into 2d
    3. do point in poly test

    example:
    p0=np.array((-1,0,5))
    p1=np.array((-0.5,0,-1))
    t0=(-2,1,3)
    t1=(-2,-1,0)
    t2=(1,-1,0)

    line=np.array([p0,p1])
    triangle=np.array([t0,t1,t2])
    IntPts=intersectLinePolygon3D(line,triangle)
    
    Author:Bin Wang(binwang.0213@gmail.com)
    Date: May. 2019
    '''
    #line plane intersection
    IntPts=intersectLinePlane(line,poly)
    #print(IntPts)

    if(IntPts is None):
        return None

    #Convert pts into 2D
    RotateMat2D,RotateMat3D=RotateMat(poly)
    poly2D=RotateTo2D(RotateMat2D,poly,poly)
    IntPts2D=RotateTo2D(RotateMat2D,poly,[IntPts])[0]

    #Test point if in polygon
    polygon2D=Path(poly2D[:,0:2])
    flag=polygon2D.contains_point(IntPts2D[0:2])
    #print(poly2D,IntPts2D,flag)
    if(flag==False):
        return None

    return IntPts
