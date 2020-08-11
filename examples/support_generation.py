""" Support generation script - 01/08/2020"""
import os
import subprocess
import random

import trimesh
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import skimage.measure

import shapely.geometry
import shapely.affinity
from shapely.ops import unary_union

from pyslm.core import Part
import pyslm.support


def getUpskinMesh(part: Part, overhangAngle: float, splitMesh = False) -> trimesh.Trimesh:
    """
    Gets the overhang mesh from a :class:`Part`. If the individual regions for the overhang mesh require separating,
    the parameter :code:`splitMesh` should be set to True. This will split regions by their network connectivity using
    Trimesh.

    :param part: The part to extract the overhang mesh from
    :param overhangAngle: The overhang angle in degrees
    :param: splitMesh: If the overhang mesh should be split into separate trimesh entities by network connnectivity
    :return: The overhang mesh
    """
    # Upward vector for support angles
    v0 = np.array([[0., 0., -1.0]])

    # Identify Support Angles
    v1 = part.geometry.face_normals
    theta = np.arccos(np.clip(np.dot(v0, v1.T), -1.0, 1.0))
    theta = np.degrees(theta).flatten()

    supportFaceIds = np.argwhere(theta >  95).flatten()

    overhangMesh = trimesh.Trimesh(vertices=part.geometry.vertices,
                                   faces=part.geometry.faces[supportFaceIds])

    if splitMesh:
        return overhangMesh.split(only_watertight=False)
    else:
        return overhangMesh

def test(myPart: Part, overhangAngle: float, splitMesh=False) -> trimesh.Trimesh:
    """
    Gets the overhang mesh from a :class:`Part`. If the individual regions for the overhang mesh require separating,
    the parameter :code:`splitMesh` should be set to True. This will split regions by their network connectivity using
    Trimesh.

    :param part: The part to extract the overhang mesh from
    :param overhangAngle: The overhang angle in degrees
    :param: splitMesh: If the overhang mesh should be split into separate trimesh entities by network connnectivity
    :return: The overhang mesh
    """

    myPart.geometry.face_adjacency_edges
    edgeIds = myPart.geometry.face_adjacency_edges


    # Upward vector for support angles
    v0 = np.array([[0., 0., -1.0]])

    # Identify Support Angles
    v1 = part.geometry.face_normals
    theta = np.arccos(np.clip(np.dot(v0, v1.T), -1.0, 1.0))
    theta = np.degrees(theta).flatten()

    supportFaceIds = np.argwhere(theta > 95).flatten()

    overhangMesh = trimesh.Trimesh(vertices=part.geometry.vertices,
                                   faces=part.geometry.faces[supportFaceIds])

    if splitMesh:
        return overhangMesh.split(only_watertight=False)
    else:
        return overhangMesh


def extrudeFace(supportFaceMesh, height=None, heightArray=None):
    # Locate boundary nodes/edges of the support face
    interiorEdges = supportFaceMesh.face_adjacency_edges
    aset = set([tuple(x) for x in supportFaceMesh.edges])
    bset = set([tuple(x) for x in interiorEdges])  # Interior edges
    # cset = aset.difference(bset)
    # boundaryEdges = np.array([x for x in aset.difference(bset)])

    meshInd = np.array([]).reshape((0, 3))
    meshVerts = np.array([]).reshape((0, 3))
    cnt = 0

    triHitLocCpy = supportFaceMesh.vertices.copy()

    if height is not None:
        triHitLocCpy[:, 2] = height
    elif heightArray is not None:
        triHitLocCpy[:, 2] = heightArray
    else:
        triHitLocCpy[:, 2] = -0.1

    # All projected faces are guaranteed to intersect with face
    for i in range(0, supportFaceMesh.faces.shape[0]):
        # extrude the triangle based on the ray length
        fid = supportFaceMesh.faces[i, :]
        tri_verts = np.array(supportFaceMesh.vertices[fid, :])

        # Create a tri from intersections
        meshVerts = np.vstack([meshVerts, tri_verts, triHitLocCpy[fid, :]])

        # Always create the bottom and top faces
        triInd = np.array([(0, 1, 2),  # Top Face
                           (4, 3, 5)  # Bottom Face
                           ])

        edgeA = {(fid[0], fid[1]), (fid[1], fid[0])}
        edgeB = {(fid[0], fid[2]), (fid[2], fid[0])}
        edgeC = {(fid[1], fid[2]), (fid[2], fid[1])}

        if len(edgeA & bset) == 0:
            triInd = np.vstack([triInd,
                                np.array([(0, 3, 4), (1, 0, 4)])  # Side Face (A)
                                ])

        if len(edgeB & bset) == 0:
            triInd = np.vstack([triInd,
                                np.array([
                                    (0, 5, 3), (0, 2, 5)])  # Side Face (B)
                                ])



        if len(edgeC & bset) == 0:
            triInd = np.vstack([triInd,
                                np.array([
                                    (2, 1, 4), (2, 4, 5)])  # Side Face (C)
                                ])

        triInd += cnt * 6
        cnt += 1

        meshInd = np.vstack((meshInd, triInd))

    # Generate the extrusion
    extMesh = trimesh.Trimesh(vertices=meshVerts, faces=meshInd, validate=True, process=True)
    extMesh.fix_normals()

    return extMesh


## CONSTANTS ####
CORK_PATH = '/home/lparry/Development/src/external/cork/bin/cork'

OVERHANG_ANGLE = 45 # deg - Overhang angle
MIN_AREA_THRESHOLD = 10 # mm2 (default = 10)
RAY_PROJECTION_RESOLUTION = 0.2 #mm (default = 0.5)
GRAD_THRESHOLD = 1.1* np.tan(np.deg2rad(OVERHANG_ANGLE)) * RAY_PROJECTION_RESOLUTION # DEFUALT = 0.5
SUPPORT_EDGE_GAP = 0.5  # mm  - offset between part supports and baseplate supports (default = 1.0)
INNER_SUPPORT_EDGE_GAP = 0.5 # mm (default = 0.1)
PART_SUPPORT_OFFSET_GAP = 0.5  # mm  - offset between part supports and baseplate supports
BASE_PLATE_SUPPORT_DISTANCE = 5  # mm  - Distance between lowest point of part and baseplate

SIMPLIFY_POLYGON_FACTOR = 3*RAY_PROJECTION_RESOLUTION
TRIANGULATION_SPACING = 0.5 # default = 1

myPart = Part('myPart')
myPart.setGeometry("../models/frameGuide.stl")
#myPart.rotation = [60.0, 50.0, -40.0]
myPart.rotation = [45, 5.0, 0.0]
myPart.dropToPlatform(20)



""" Extract the overhang mesh - don't explicitly split the mesh"""
overhangMesh = pyslm.support.getOverhangMesh(myPart, OVERHANG_ANGLE, False)
overhangMesh.visual.face_colors = [1.0, 0., 0., 1.0]

upskinMesh = getUpskinMesh(myPart, OVERHANG_ANGLE, False)
upskinMesh.visual.face_colors = [0.0, 1., 0., 1.0]

# split the mesh
overhangSubregions = overhangMesh.split(only_watertight=False)
myPart.geometry.process(validate=True)
#trimesh.constants.tol.merge = 1
#myPart.geometry.merge_vertices()
vAdjacency = myPart.geometry.vertex_neighbors

edges = myPart.geometry.edges_unique
meshVerts =  myPart.geometry.vertices
centroids = myPart.geometry.triangles_center

# Calculate the vertical face angles
v0 = np.array([[0., 0., 1.0]])

# Identify Support Angles
v1 = myPart.geometry.face_normals
theta = np.arccos(np.clip(np.dot(v0, v1.T), -1.0, 1.0))
theta = np.degrees(theta).flatten()
adjacentFaceAngles = np.rad2deg(myPart.geometry.face_adjacency_angles)


overhangEdges = []
overhangEdgeList = []
edgeVerts = meshVerts[edges]

rays = []

for i in range(len(edgeVerts)):
    edge = edgeVerts[i].reshape(2,3)
    delta = edge[0] - edge[1]

    mag = np.sqrt(np.sum(delta * delta))
    ang = np.rad2deg(np.arcsin(delta[2]/mag))

    if(np.abs(ang) < 10):
        adjacentFaces = myPart.geometry.face_adjacency[i]
        triVerts = meshVerts[myPart.geometry.faces[adjacentFaces]].reshape(-1, 3)

        if adjacentFaceAngles[i] > OVERHANG_ANGLE and np.all(theta[adjacentFaces] > 89):
            #if np.all(theta[adjacentFaces] > 89) and np.all(triVerts[:,2] - np.min(edge[:,2]) > -0.01):
            overhangEdges.append(edges[i])
            rays.append(edge)

rays = np.array(rays)
visualize_support_edges = trimesh.load_path((rays).reshape(-1, 2, 3))




pointOverhang = []
for i in range(len(vAdjacency)):
    v = meshVerts[i]
    verts = meshVerts[vAdjacency[i],:]
    delta = verts-v
    #mag = np.sqrt(np.sum(delta * delta, axis=1))
    #theta = np.arcsin(delta[:,2]/mag)
    #theta = np.rad2deg(theta)
    #if np.all(theta > -0.001):
        #pointOverhang.append(i)

    if np.all(delta[:,2] > -0.05):
        pointOverhang.append(i)

if overhangEdges:
    overhangEdges = np.array(overhangEdges)
    overhangEdges = np.sort(overhangEdges, axis=1)
    overhangEdges = np.unique(overhangEdges, axis=0)
pointSupports =[]

"""
Generate the geometry for the supports
"""
rays = []
for pnt in pointOverhang:
    coords = np.zeros((2,3))
    coords[:,:] = meshVerts[pnt]
    coords[1,2] = 0.0
    rays.append(coords)

supportExtrudes = []
for edge in overhangEdges:
    coords = np.vstack((meshVerts[edge,:],meshVerts[edge,:]))
    coords[2:,2] = 0.0

    extrudeFace= np.array([(0, 1, 3), (3, 2, 0)])
    supportExtrudes.append(trimesh.Trimesh(vertices=coords, faces=extrudeFace))

rays = np.array(rays)
ray_visualize = trimesh.load_path((rays).reshape(-1, 2, 3))

myPart.geometry.visual.vertex_colors = [80,80,80, 125]
s2 = trimesh.Scene([myPart.geometry, ray_visualize, visualize_support_edges, overhangMesh] + supportExtrudes)
s2.show()


d

myPart.geometry.visual.vertex_colors = [50,50,50,125]

myPart.geometry.visual.vertex_colors[pointOverhang] = [255, 255, 0.0, 255]

myPart.geometry.show()
de
rays = []
myExtrusions = []

myPart.geometry.export('part.off')

numcnt = 10

""" Process sub-regions"""
for subregion in overhangSubregions:

    print('processing subregion')
    supportRegion = subregion.copy()

    """ Extract the outline of the overhang mesh region"""
    poly = subregion.outline()

    """ Convert the line to a 2D polygon"""
    poly.vertices[:, 2] = 0.0

    flattenPath, bd = poly.to_planar()
    flattenPath.process()
    polygon = flattenPath.polygons_full[0]

    # Offset in 2D the support region projection
    offsetShape = polygon.buffer(-SUPPORT_EDGE_GAP)

    if offsetShape is None or offsetShape.area < MIN_AREA_THRESHOLD:
        continue

    offsetPoly = trimesh.load_path(offsetShape)
    offsetPoly.apply_translation(np.array([bd[0, 3], bd[1, 3]]))

    """
    Create an extrusion at the vertical extent of the part
    """
    extruMesh = extrudeFace(supportRegion.copy(), 0.0)
    extruMesh.vertices[:, 2] = extruMesh.vertices[:, 2] - 0.01

    print('\t - start intersecting mesh')
    # Intersect the projection of the support face with the original part using the Cork Library
    extruMesh.export('downProjExtr.off')
    subprocess.call([CORK_PATH, '-isct', 'part.off', 'downProjExtr.off', 'c.off'])
    print('\t - finished intersecting mesh')

    """
    Note the cutMesh is the project down from the support surface with the original mesh
    """
    cutMesh = trimesh.load_mesh('c.off')

    if len(cutMesh.faces) == 0:

        extruMesh.visual.face_colors = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255]
        myExtrusions.append(extruMesh)
        continue  # No intersection had taken place

    # Rasterise the surface of overhang to generate projection points
    supportArea = np.array(offsetPoly.rasterize(RAY_PROJECTION_RESOLUTION, offsetPoly.bounds[0, :])).T

    coords = np.argwhere(supportArea).astype(np.float32) * RAY_PROJECTION_RESOLUTION
    coords += offsetPoly.bounds[0, :] + 1e-5  # An offset is required due to rounding error
    print('\tnumber of rays with resolution ({:.3f}): {:d}'.format(RAY_PROJECTION_RESOLUTION, len(coords)))

    """
    Project upwards to intersect with the upper surface
    """
    # Set the z-coordinates for the ray origin
    coords = np.insert(coords, 2, values=-1e5, axis=1)
    rays = np.repeat([[0., 0., 1.]], coords.shape[0], axis=0)

    # Find the first location of any triangles which intersect with the part
    hitLoc, index_ray, index_tri = supportRegion.ray.intersects_location(ray_origins=coords,
                                                                         ray_directions=rays,
                                                                         multiple_hits=True)
    print('\tfinished projecting rays')

    hitLocCpy = hitLoc.copy()
    hitLocCpy[:, :2] -= offsetPoly.bounds[0, :]
    hitLocCpy[:, :2] /= RAY_PROJECTION_RESOLUTION

    hitLocIdx = np.ceil(hitLocCpy[:, :2]).astype(np.int32)

    coords2 = coords.copy()

    coords2[index_ray, 2] = 1e7
    rays[:, 2] = -1.0

    # If any verteces in triangle there is an intersection
    # Find the first location of any triangles which intersect with the part
    hitLoc2, index_ray2, index_tri2 = cutMesh.ray.intersects_location(ray_origins=coords2,
                                                                      ray_directions=rays,
                                                                      multiple_hits=False)

    hitLocCpy2 = hitLoc2.copy()
    # Update the xy coordinates
    hitLocCpy2[:, :2] -= offsetPoly.bounds[0, :]
    hitLocCpy2[:, :2] /= RAY_PROJECTION_RESOLUTION
    hitLocIdx2 = np.ceil(hitLocCpy2[:, :2]).astype(np.int32)

    # Create a height map of the projection rays
    heightMap = np.ones(supportArea.shape) * -1.0
    heightMap2 = np.ones(supportArea.shape) * -1.0
    # Assign the heights
    heightMap[hitLocIdx[:, 0], hitLocIdx[:, 1]] = hitLoc[:, 2]

    # Assign the heights based on the lower projection
    heightMap[hitLocIdx2[:, 0], hitLocIdx2[:, 1]] = hitLoc2[:, 2]

    vx, vy = np.gradient(heightMap)
    grads = np.sqrt(vx ** 2 + vy ** 2)

    """
    Find the outlines of any regions of the height map which deviate significantly
    This is used to separate both self-intersecting supports and those which are simply connected to the base-plate
    """
    outlines = skimage.measure.find_contours(grads, GRAD_THRESHOLD)

    if numcnt < 1000:
        pass
    else:
        plt.figure()
        plt.imshow(grads)
        plt.contour(grads, 0.5, linewidths=2.0, colors='white')
        plt.figure()
        plt.imshow(heightMap.T)
        plt.figure()
        plt.imshow(grads.T)

        for outline in outlines:
            plt.plot(outline[:, 0], outline[:, 1])


    numcnt += 1

    polygons = []

    for outline in outlines:

        """
        Process the outline by finding the boundaries
        """
        outline = outline * RAY_PROJECTION_RESOLUTION + offsetPoly.bounds[0, :]
        outline = skimage.measure.approximate_polygon(outline, tolerance=SIMPLIFY_POLYGON_FACTOR)

        if outline.shape[0] < 3:
            continue

        """
        Process the polygon
        ---------------------
        Create a shapley polygon  and offset the boundary
        """
        mergedPoly = trimesh.load_path(outline)

        if not mergedPoly.is_closed or len(mergedPoly.polygons_full) == 0 or mergedPoly.polygons_full[0] is None:
            continue

        bufferPoly = mergedPoly.polygons_full[0].buffer(-INNER_SUPPORT_EDGE_GAP)

        if isinstance(bufferPoly, shapely.geometry.MultiPolygon):
            polygons += bufferPoly.geoms
        else:
            polygons.append(bufferPoly)

    if False:
        plt.figure()
        plt.imshow(heightMap.T)
        for outline in outlines:
            plt.plot(outline[:, 0], outline[:, 1])
    for bufferPoly in polygons:

        if bufferPoly.area < MIN_AREA_THRESHOLD:
            continue

        """
        Triangulate the polygon into a planar mesh
        """
        poly_tri = trimesh.creation.triangulate_polygon(bufferPoly, triangle_args='pa{:.3f}'.format(TRIANGULATION_SPACING))

        """
        Project upwards to intersect with the upper surface
        Project the vertices downward (-z) to intersect with the cutMesh
        """
        coords = np.insert(poly_tri[0], 2, values=-1e-7, axis=1)
        ray_dir = np.repeat([[0.,0.,1.]], coords.shape[0], axis=0)

        # Find the first location of any triangles which intersect with the part
        hitLoc, index_ray, index_tri = supportRegion.ray.intersects_location(ray_origins=coords,
                                                                             ray_directions=ray_dir,
                                                                             multiple_hits=False)

        coords2 = coords.copy()
        coords2[index_ray, 2] = hitLoc[:, 2] - 0.2

        ray_dir[:, 2] = -1.0

        """
        Intersecting with cutmesh is more efficient when projecting downwards
        """
        hitLoc2, index_ray2, index_tri2 = cutMesh.ray.intersects_location(ray_origins=coords2,
                                                                          ray_directions=ray_dir,
                                                                          multiple_hits=False)
        if len(hitLoc) != len(coords) or len(hitLoc2) != len(hitLoc):
            # The projections up and down do not match indicating that there maybe some flaw
            print(hitLoc.shape, hitLoc2.shape, coords.shape)

            if len(hitLoc2) == 0:
                # Base plate
                hitLoc2 = coords2.copy()
                hitLoc2[:, 2] = 0.0

                print('CREATING BASE-PLATE SUPPORT')
            else:
                print('PROJECTIONS NOT MATCHING')
                continue
                raise ValueError('PROJECTIONS NOT MATCHING')

        # Create a surface from the Ray intersection
        surf2 = trimesh.Trimesh(vertices=coords2, faces=poly_tri[1])

        # Extrude the surface based on the heights from the second ray cast
        newShapes = extrudeFace(surf2, None, hitLoc2[:, 2] - 0.05)

        """
        Take the near net-shape support and obtain the difference with the original part to get clean boundaries for the support
        """
        newShapes.export('b.off')
        subprocess.call([CORK_PATH, '-diff', 'b.off', 'part.off', 'c.off'])
        cutMesh2 = trimesh.load_mesh('c.off')

        # Draw the support structures generated
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255]
        cutMesh2.visual.face_colors = color
        myExtrusions.append(cutMesh2)

    print('processed support face')

overhangMesh.vertices[:,2] -= 0.1
#myPart.geometry
s = trimesh.Scene([overhangMesh,upskinMesh] + myExtrusions)
# open the scene viewer and move a ball around
#s.show(viewer='gl')


#overhangMesh.export('testSupport.ply');
#myPart.draw()

supportGeom = myExtrusions[0]
bx = supportGeom.bounds[:,0]
by = supportGeom.bounds[:,1]

spacingX = 3.0
spacingY = 3.0

# Obtain the section through the STL extension using Trimesh Algorithm (Shapely)
sectionsX = supportGeom.section_multiplane(plane_origin=[0.0, 0, 0],
                                                     plane_normal=[1, 0, 0],
                                                     heights = np.arange(bx[0], bx[1], spacingX))

sectionsY = supportGeom.section_multiplane(plane_origin=[0, 0.0, 0],
                                                     plane_normal=[0.0, 1.0, 0],
                                                     heights = np.arange(by[0], by[1], spacingY))

xSectionMesh = trimesh.Trimesh()
ySectionMesh = trimesh.Trimesh()

for section in sectionsX:
    if section is None:
        continue

    vx,fx = section.triangulate()
    if(len(vx)  == 0):
        continue


    vx = np.insert(vx, 2, values=0.0, axis=1)
    secX = trimesh.Trimesh(vertices=vx, faces=fx)
    secX.apply_transform(section.metadata['to_3D'])
    xSectionMesh += secX

for section in sectionsY:

    if section is None:
        continue


    vy,fy = section.triangulate()

    if (len(vy) == 0):
        continue

    vy = np.insert(vy, 2, values=0.0, axis=1)
    secY = trimesh.Trimesh(vertices=vy, faces=fy)
    secY.apply_transform(section.metadata['to_3D'])
    ySectionMesh += secY


print('\t - start intersecting mesh')
# Intersect the projection of the support face with the original part using the Cork Library
xSectionMesh.export('secX.off')
ySectionMesh.export('secY.off')
subprocess.call([CORK_PATH, '-resolve', 'secY.off', 'secX.off', 'merge.off'])
print('\t - finished intersecting mesh')
isectMesh = trimesh.load_mesh('merge.off')
#sectionsX[0].show()
#sectionsY[0].show()

supportGeom.visual.face_colors = [0.0, 1., 0., 0.3]
s2 = trimesh.Scene([isectMesh, supportGeom])
s2.show()

# Obtain the 2D Planar Section at this Z-position
#planarSection, transform = sections.to_planar(transformMat, normal=[1,0,0])

