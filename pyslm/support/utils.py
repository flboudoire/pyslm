from typing import List
import numpy as np
import trimesh

from typing import Optional, Tuple, List

from ..core import Part

def getSupportAngles(part: Part) -> np.ndarray:
    """
    Returns the support angles for each triangular face normal. This is mainly used for the benefit of visualisng the
    support angles for a part.

    :param part:
    :return: The support angles acros tseh whole part
    """

    # Upward vector for support angles
    v0 = np.array([[0., 0., -1.0]])

    # Identify Support Angles
    v1 = part.geometry.face_normals
    theta = np.arccos(np.clip(np.dot(v0, v1.T), -1.0, 1.0))
    theta = np.degrees(theta).flatten()

    return theta


def getOverhangMesh(part: Part, overhangAngle: float) -> trimesh.Trimesh:
    """
    Gets the overhang mesh from a :class:`Part`.

    :param part: The part to extract the overhang mesh from
    :param overhangAngle: The overhang angle in degrees
    :return: The overhang mesh
    """
    # Upward vector for support angles
    v0 = np.array([[0., 0., -1.0]])

    # Identify Support Angles
    v1 = part.geometry.face_normals
    theta = np.arccos(np.clip(np.dot(v0, v1.T), -1.0, 1.0))
    theta = np.degrees(theta).flatten()

    supportFaceIds = np.argwhere(theta > 180 - overhangAngle).flatten()

    overhangMesh = trimesh.Trimesh(vertices=part.geometry.vertices,
                                   faces=part.geometry.faces[supportFaceIds])

    return overhangMesh

def approximateSupportMomentArea(part: Part, overhangAngle: float) -> float:
    """
    The support moment area is a metric, which projects the distance from the base-plate (:math:`z=0`) for
    each support surface multiplied by the area. It gives a two parameter component cost function for the support area.

    .. note::
        This is an approximation that does not account for any self-intersections. It does not use ray queries to project
        the distance towards the mesh, therefore is more useful estimating the overall cost of the support structures,
        during initial support optimisation.

    :param part:
    :param overhangAngle: The overhang angle in degrees
    :return: The approximate cost function
    """
    overhangMesh = getOverhangMesh(part, overhangAngle)

    zHeights = overhangMesh.triangles_center[:,2]

    # Use the projected area by flattening the support faces
    overhangMesh.vertices[:,2] = 0.0
    faceAreas = overhangMesh.area_faces
    
    return np.sum(faceAreas*zHeights)

def approximateSupportMapByCentroid(part: Part, overhangeAngle: float,
                                    includeTriangleVertices: Optional[bool]=False) -> Tuple[np.ndarray]:
    """
    This method to approximate the surface area, projects  a single ray :math:`(0,0,-1)`, form each triangle in the
    overhang mesh -originating from the centroid or optionally each triangle vertex by setting the
    :code:`includeTriangleVertices` parameter. A self-intersection test with the mesh is performed  and this is used to
    calculate the distance from the hit location or if no intersection is made the base-plate (:math:`z=0.0`)
    which may be used later to generate a support heightmap.

    :param part:
    :param overhangeAngle: The overhang angle in degrees
    :param includeTriangleVertices: Optional parameter projects also from the triangular vertices
    :return: A tuple with the support map and
    """

    overhangMesh = getOverhangMesh(part, overhangeAngle)

    coords = overhangMesh.triangles_center

    if includeTriangleVertices:
        coords = np.vstack([coords, overhangMesh.vertices])

    ray_dir = np.tile(np.array([[0.,0.,-1.0]]), (coords.shape[0],1))

    # Find the first intersection hit of rays project from the triangle.
    hitLoc, index_ray, index_tri = part.geometry.ray.intersects_location(ray_origins=coords,
                                                                         ray_directions=ray_dir,
                                                                         multiple_hits = False)

    heightMap =  np.zeros((coords.shape[0],1), dtype=np.float)
    heightMap[index_ray] = hitLoc[:,2].reshape(-1,1)
    
    heightMap = np.abs(heightMap - coords[:,2])

    return heightMap


def approximateProjectionSupportCost(part: Part, overhangeAngle: float,
                                     includeTriangleVertices: Optional[bool]=False) -> float:
    """
    Provides a support structure cost using ray projection from the overhang regions which allows for self-intersection
    checks.

    :param part:
    :param overhangeAngle: The overhang angle in degree
    :param includeTriangleVertices: Optional parameter projects also from the triangular vertices
    :return: The cost function for support generation
    """

    overhangMesh = getOverhangMesh(part, overhangeAngle)

    heightMap = approximateSupportMapByCentroid(part, overhangeAngle, includeTriangleVertices)

    # Project the overhang area
    overhangMesh.vertices[:, 2] = 0.0
    faceAreas = overhangMesh.area_faces

    return np.sum(faceAreas * heightMap), heightMap