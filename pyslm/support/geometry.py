"""
Provides supporting functions to generate geometry for support structures
"""

try:
    import triangle

except BaseException as E:
    raise BaseException("Lib Triangle is Required to use this Support.geometry submodule")

from typing import List
import numpy as np
import trimesh

from typing import Optional, Tuple, List


def extrudeFace(extrudeMesh: trimesh.Trimesh, height: Optional[float] = None, heightArray: Optional[np.ndarray] = None) -> trimesh.Trimesh:
    """
    Extrudes a set of connected triangle faces into a prism. This is based on a constant height - or a height array
    for corresponding to extrusions to be added to each triangular facet.

    :param faceMesh: A mesh consisting of *n* triangular faces to extrude
    :param height: A constant height to use for the prism extrusion
    :param heightArray: Optional array consisting of *n* heights to extrude each triangular facet
    :return: The extruded prism mesh
    """
    faceMesh = extrudeMesh.copy()

    # Locate boundary nodes/edges of the support face
    interiorEdges = faceMesh.face_adjacency_edges
    aset = set([tuple(x) for x in faceMesh.edges])
    bset = set([tuple(x) for x in interiorEdges])  # Interior edges
    # cset = aset.difference(bset)
    # boundaryEdges = np.array([x for x in aset.difference(bset)])

    # Deep copy the vertices from the face mesh
    triVertCpy = faceMesh.vertices.copy()

    if height is not None:
        triVertCpy[:, 2] = height
    elif heightArray is not None:
        triVertCpy[:, 2] = heightArray
    else:
        triVertCpy[:, 2] = -0.1

    meshInd = np.array([]).reshape((0, 3))
    meshVerts = np.array([]).reshape((0, 3))

    # Count indicator increases the triangle index upon each loop iteration
    cnt = 0

    # All projected faces are guaranteed to intersect with face
    for i in range(0, faceMesh.faces.shape[0]):
        # extrude the triangle based on the ray length
        fid = faceMesh.faces[i, :]
        tri_verts = np.array(faceMesh.vertices[fid, :])

        # Create a tri from intersections
        meshVerts = np.vstack([meshVerts, tri_verts, triVertCpy[fid, :]])

        # Always create the bottom and top faces
        triInd = np.array([(0, 1, 2),  # Top Face
                           (4, 3, 5)  # Bottom Face
                           ])

        edgeA = {(fid[0], fid[1]), (fid[1], fid[0])}
        edgeB = {(fid[0], fid[2]), (fid[2], fid[0])}
        edgeC = {(fid[1], fid[2]), (fid[2], fid[1])}

        if len(edgeA & bset) == 0:
            triInd = np.vstack([triInd, ((0, 3, 4), (1, 0, 4))])  # Side Face (A)

        if len(edgeB & bset) == 0:
            triInd = np.vstack([triInd, ((0, 5, 3), (0, 2, 5)) ]) # Side Face (B)

        if len(edgeC & bset) == 0:
            triInd = np.vstack([triInd, ((2, 1, 4), (2, 4, 5))])  # Side Face (C)

        triInd += cnt * 6
        cnt += 1

        meshInd = np.vstack((meshInd, triInd))

    # Generate the extrusion
    extMesh = trimesh.Trimesh(vertices=meshVerts, faces=meshInd, validate=True, process=True)
    extMesh.fix_normals()

    return extMesh

