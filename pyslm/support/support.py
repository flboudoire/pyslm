"""
Provides classes  and methods for the creation of support structures in Additive Manufacturing.
"""

import abc
import time, random
from typing import Any, Optional, Tuple, List
import subprocess

from skimage.measure import approximate_polygon, subdivide_polygon, find_contours
import shapely.geometry

import numpy as np

from pyslm import pyclipper

from ..core import Part
from .utils import *
from .geometry import *

from shapely.geometry import Polygon as ShapelyPolygon


class BaseSupportGenerator(abc.ABC):
    """
    The BaseSupportGeneration class provides common methods used for generating the 'support' structures typically used
    in Additive Manufacturing.
    """

    PYCLIPPER_SCALEFACTOR = 1e4
    """ 
    The scaling factor used for polygon clipping and offsetting in `PyClipper <http://pyclipper.com>`_ for the decimal
    component of each polygon coordinate. This should be set to inverse of the required decimal tolerance i.e. 0.01 
    requires a minimum scalefactor of 1e2. This scaling factor is used in :meth:`~BaseHatcher.scaleToClipper` 
    and :meth:`~BaseHatcher.scaleFromClipper`. 
    """

    POINT_OVERHANG_TOLERANCE = 0.05

    def __init__(self):
        pass

    def __str__(self):
        return 'BaseGenerator <{:s}>'.format(self.name)

    @staticmethod
    def findOverhangPoints(part: Part) -> np.ndarray:

        meshVerts = part.geometry.vertices
        vAdjacency = part.geometry.vertex_neighbors

        pointOverhangs = []
        for i in range(len(vAdjacency)):

            # Find the edge deltas between the points
            v = meshVerts[i]
            neighborVerts = meshVerts[vAdjacency[i], :]
            delta = neighborVerts - v
            # mag = np.sqrt(np.sum(delta * delta, axis=1))
            # theta = np.arcsin(delta[:,2]/mag)
            # theta = np.rad2deg(theta)
            # if np.all(theta > -0.001):
            # pointOverhang.append(i)

            """
            If all neighbouring connected vertices lie above the point, this indicates the vertex lies below and 'may'
            not have underlying connectivity. There are two cases that exist: on upwards or downwards pointing surface.
            """
            if np.all(delta[:, 2] > -BaseSupportGenerator.POINT_OVERHANG_TOLERANCE):

                # Check that the vertex normal is pointing downwards (-ve Z) showing that the no material is underneath
                if part.geometry.vertex_normals[i][2] < 0.0:
                    pointOverhangs.append(i)

        return pointOverhangs

    @staticmethod
    def findOverhangEdges(part: Part, overhangAngle: float = 45, edgeOverhangAngle: float = 10):

        meshVerts = part.geometry.vertices
        edges = part.geometry.edges_unique
        edgeVerts = meshVerts[edges]

        """ 
        Calculate the face angles with respect to the +z vector  and the inter-face angles
        """
        theta = getSupportAngles(part, np.array([[0., 0., 1.0]]))
        adjacentFaceAngles = np.rad2deg(part.geometry.face_adjacency_angles)

        overhangEdges = []
        # Iterate through all the edges in the model
        for i in range(len(edgeVerts)):

            """
            Calculate the 'vertical' angle of the edge pointing in the z-direction by using the z component.
            First calculate vector, magnitude and the vertical angle of the vector
            """
            edge = edgeVerts[i].reshape(2, 3)
            delta = edge[0] - edge[1]
            mag = np.sqrt(np.sum(delta * delta))
            ang = np.rad2deg(np.arcsin(delta[2] / mag))

            # Identify if the vertical angle of the edge is less than the edgeOverhangAngle irrespective of the actual
            # direction of the vector (bi-directional)
            if np.abs(ang) < edgeOverhangAngle:

                """
                Locate the adjacent faces in the model using the face-adjacency property to identify if the edge 
                belongs to a sharp corner which tends to be susceptible areas. This is done by calculating the angle
                between faces.
                """
                adjacentFaces = part.geometry.face_adjacency[i]
                # triVerts = meshVerts[part.geometry.faces[adjacentFaces]].reshape(-1, 3)

                if adjacentFaceAngles[i] > overhangAngle and np.all(theta[adjacentFaces] > 89):
                    # if np.all(theta[adjacentFaces] > 89) and np.all(triVerts[:,2] - np.min(edge[:,2]) > -0.01):
                    overhangEdges.append(edges[i])

        return overhangEdges

    @staticmethod
    def flattenSupportRegion(region):
        supportRegion = region.copy()

        """ Extract the outline of the overhang mesh region"""
        poly = supportRegion.outline()

        """ Convert the line to a 2D polygon"""
        poly.vertices[:, 2] = 0.0

        flattenPath, polygonTransform = poly.to_planar()
        flattenPath.process()

        flattenPath.apply_translation(polygonTransform[:2, 3]) #np.array([polygonTransform[0, 3],
                                               # polygonTransform[1, 3]]))
        polygon = flattenPath.polygons_full[0]



        return polygon #, polygonTransform


class BlockSupportGenerator(BaseSupportGenerator):
    """
    The BlockSupportGenerator class provides common methods used for generating the 'support' structures typically used
    in Additive Manufacturing for block polygon regions
    """

    CORK_PATH = '/home/lparry/Development/src/external/cork/bin/cork'
    """
    Path to the Cork Boolean Library
    """

    def __init__(self):

        self._supportEdgeGap = 0.5
        self._minimumAreaThreshold = 5.0  # mm2 (default = 10)
        self._rayProjectionResolution = 0.2  # mm (default = 0.5)

        self._innerSupportEdgeGap = 0.2  # mm (default = 0.1)
        self._partSupportOffsetGap = 0.5  # mm  - offset between part supports and baseplate supports

        self._triangulationSpacing = 2  # mm (default = 1)
        self._simplifyPolygonFactor = 3.0 * self._rayProjectionResolution

        self._overhangAngle = 45.0  # [deg]

    def __str__(self):
        return 'BlockSupportGenerator <{:s}>'.format(self.name)

    @staticmethod
    def gradThreshold(rayProjectionDistance: float, overhangAngle: float) -> float:
        return 5.0 * np.tan(np.deg2rad(overhangAngle)) * rayProjectionDistance

    @property
    def overhangAngle(self) -> float:
        return self._overhangAngle

    @overhangAngle.setter
    def overhangAngle(self, value):
        self._overhangAngle = value

    @property
    def supportEdgeGap(self) -> float:
        return self._supportEdgeGap

    @supportEdgeGap.setter
    def supportEdgeGap(self, spacing: float):
        self._supportEdgeGap = spacing

    @property
    def innerSupportEdgeGap(self) -> float:
        return self._innerSupportEdgeGap

    @innerSupportEdgeGap.setter
    def innerSupportEdgeGap(self, spacing: float):
        self._innerSupportEdgeGap = spacing

    @property
    def minimumAreaThreshold(self) -> float:
        """
        The minimum support area threshold used to identify disconnected support regions.
        """
        return self._minimumAreaThreshold

    @minimumAreaThreshold.setter
    def minimumAreaThreshold(self, areaThresholdValue: float):
        self._minimumAreaThreshold = areaThresholdValue

    @property
    def simplifyPolygonFactor(self) -> float:
        return 3. * self._rayProjectionResolution

    @property
    def triangulationSpacing(self) -> float:
        return self._triangulationSpacing

    @triangulationSpacing.setter
    def triangulationSpace(self, spacing: float):
        self._triangulationSpacing = spacing

    @property
    def rayProjectionResolution(self) -> float:
        return self._rayProjectionResolution

    @rayProjectionResolution.setter
    def rayProjectionResolution(self, resolution: float):
        self._rayProjectionResolution = resolution

    def filterSupportRegion(self, region):
        pass

    def generateIntersectionHeightMap(self):
       pass

    def _identifySelfIntersectionHeightMap(self, subregion, offsetPoly, cutMesh):

        # Rasterise the surface of overhang to generate projection points
        supportArea = np.array(offsetPoly.rasterize(self.rayProjectionResolution, offsetPoly.bounds[0, :])).T

        coords = np.argwhere(supportArea).astype(np.float32) * self.rayProjectionResolution
        coords += offsetPoly.bounds[0, :] + 1e-5  # An offset is required due to rounding error

        print('\t - start projecting rays')
        print('\t - number of rays with resolution ({:.3f}): {:d}'.format(self.rayProjectionResolution, len(coords)))

        """
        Project upwards to intersect with the upper surface
        """
        # Set the z-coordinates for the ray origin
        coords = np.insert(coords, 2, values=-1e5, axis=1)
        rays = np.repeat([[0., 0., 1.]], coords.shape[0], axis=0)

        # Find the first location of any triangles which intersect with the part
        hitLoc, index_ray, index_tri = subregion.ray.intersects_location(ray_origins=coords,
                                                                         ray_directions=rays,
                                                                         multiple_hits=False)


        hitLocCpy = hitLoc.copy()
        hitLocCpy[:, :2] -= offsetPoly.bounds[0, :]
        hitLocCpy[:, :2] /= self.rayProjectionResolution

        hitLocIdx = np.ceil(hitLocCpy[:, :2]).astype(np.int32)

        coords2 = coords.copy()

        coords2[index_ray, 2] = 1e7
        rays[:, 2] = -1.0

        # If any verteces in triangle there is an intersection
        # Find the first location of any triangles which intersect with the part
        hitLoc2, index_ray2, index_tri2 = cutMesh.ray.intersects_location(ray_origins=coords2,
                                                                          ray_directions=rays,
                                                                          multiple_hits=False)

        print('\t - finished projecting rays')

        hitLocCpy2 = hitLoc2.copy()
        # Update the xy coordinates
        hitLocCpy2[:, :2] -= offsetPoly.bounds[0, :]
        hitLocCpy2[:, :2] /= self.rayProjectionResolution
        hitLocIdx2 = np.ceil(hitLocCpy2[:, :2]).astype(np.int32)

        # Create a height map of the projection rays
        heightMap = np.ones(supportArea.shape) * -1.0

        heightMapUpper = np.zeros(supportArea.shape)
        heightMapLower = np.zeros(supportArea.shape)

        # Assign the heights
        heightMap[hitLocIdx[:, 0], hitLocIdx[:, 1]] = hitLoc[:, 2]
        heightMapUpper[hitLocIdx[:, 0], hitLocIdx[:, 1]] = hitLoc[:,2]

        # Assign the heights based on the lower projection
        heightMap[hitLocIdx2[:, 0], hitLocIdx2[:, 1]] = hitLoc2[:, 2]
        heightMapLower[hitLocIdx2[:, 0], hitLocIdx2[:, 1]] = hitLoc2[:, 2]

        print('\tgenerated support height map')
        return heightMap, heightMapUpper, heightMapLower

    def identifySupportRegions(self, part: Part, overhangAngle: float,
                               findSelfIntersectingSupport: Optional[bool] = True):
        """
        Extracts the overhang mesh and generates block regions given a part and target overhang angle. The algorithm
        uses a combination of boolean operations and ray intersection/projection to discriminate support regions.
        If :code:`findSelfIntersectingSuppoort` is to set :code:`True` (default), the algorithm will process and
        separate overhang regions that by downward projection self-intersect with the part.
        This provides more refined behavior than simply projected support material downwards into larger support
        block regions and separates an overhang surface between intersecting and non-intersecting regions.

        :param part: The part to generate block supports for
        :param overhangAngle: Overhang angle in degrees
        :param findSelfIntersectingSupport:
        """

        overhangSubregions = getOverhangMesh(part, overhangAngle, True)

        """
        The geometry of the part requires exporting as a '.off' file to be correctly used with the Cork Library
        """
        part.geometry.export('part.off')

        supportBlockRegions = []

        """ Process sub-regions"""
        for subregion in overhangSubregions:

            print('Processing subregion')
            polygon = self.flattenSupportRegion(subregion)

            # Offset in 2D the support region projection
            offsetShape = polygon.buffer(-self.supportEdgeGap)

            if offsetShape is None or offsetShape.area < self.minimumAreaThreshold:
                print('skipping shape')
                continue

            offsetPoly = trimesh.load_path(offsetShape)

            """
            Create an extrusion at the vertical extent of the part
            """
            extruMesh = extrudeFace(subregion, 0.0)
            extruMesh.vertices[:, 2] = extruMesh.vertices[:, 2] - 0.01

            print('\t - start intersecting mesh')
            # Intersect the projection of the support face with the original part using the Cork Library
            extruMesh.export('downProjExtr.off')
            subprocess.call([self.CORK_PATH, '-isct', 'part.off', 'downProjExtr.off', 'c.off'])
            print('\t - finished intersecting mesh')

            """
            Note the cutMesh is the project down from the support surface with the original mesh
            """
            cutMesh = trimesh.load_mesh('c.off')

            if len(cutMesh.faces) == 0:
                extruMesh.visual.face_colors[:,:3] = np.random.randint(254, size=3)
                supportBlockRegions.append(extruMesh)
                continue  # No intersection had taken place
            elif not findSelfIntersectingSupport:
                continue

            v0 = np.array([[0., 0., 1.0]])

            # Identify Support Angles
            v1 = cutMesh.face_normals
            theta = np.arccos(np.clip(np.dot(v0, v1.T), -1.0, 1.0))
            theta = np.degrees(theta).flatten()

            cutMeshUpper = cutMesh.copy()
            cutMeshUpper.update_faces(theta < 89.95)
            cutMeshUpper.remove_unreferenced_vertices()

            # Toggle to use full intersecting mesh
            # cutMeshUpper = cutMesh

            # Use a ray-tracing approach to identfy self-intersections. This provides a method to isolate regions that
            # either are self-intersecting or not.

            heightMap, heightMapUpper, heightMapLower = self._identifySelfIntersectionHeightMap(subregion,offsetPoly,cutMeshUpper)
            vx, vy = np.gradient(heightMap)
            grads = np.sqrt(vx ** 2 + vy ** 2)

            """
            Find the outlines of any regions of the height map which deviate significantly
            This is used to separate both self-intersecting supports and those which are simply connected to the base-plate
            """
            outlines = find_contours(grads, self.gradThreshold(self.rayProjectionResolution, self.overhangAngle))

            polygons = []

            for outline in outlines:

                """
                Process the outline by finding the boundaries
                """
                outline = outline * self.rayProjectionResolution + offsetPoly.bounds[0, :]
                outline = approximate_polygon(outline, tolerance=self.simplifyPolygonFactor)

                if outline.shape[0] < 3:
                    continue

                """
                Process the polygon  by creating a shapley polygon and offseting the boundary
                """
                mergedPoly = trimesh.load_path(outline)

                if not mergedPoly.is_closed or len(mergedPoly.polygons_full) == 0 or mergedPoly.polygons_full[0] is None:
                    continue

                bufferPoly = mergedPoly.polygons_full[0].buffer(-self.innerSupportEdgeGap)

                if isinstance(bufferPoly, shapely.geometry.MultiPolygon):
                    polygons += bufferPoly.geoms
                else:
                    polygons.append(bufferPoly)

            for bufferPoly in polygons:

                if bufferPoly.area < self.minimumAreaThreshold:
                    continue

                """
                Triangulate the polygon into a planar mesh
                """
                poly_tri = trimesh.creation.triangulate_polygon(bufferPoly,
                                                                triangle_args='pa{:.3f}'.format(self.triangulationSpacing))

                """
                Project upwards to intersect with the upper surface
                Project the vertices downward (-z) to intersect with the cutMesh
                """
                coords = np.insert(poly_tri[0], 2, values=-1e-7, axis=1)
                ray_dir = np.repeat([[0., 0., 1.]], coords.shape[0], axis=0)

                # Find the first location of any triangles which intersect with the part
                hitLoc, index_ray, index_tri = subregion.ray.intersects_location(ray_origins=coords,
                                                                                 ray_directions=ray_dir,
                                                                                 multiple_hits=False)

                coords2 = coords.copy()
                coords2[index_ray, 2] = hitLoc[:, 2] - 0.2

                ray_dir[:, 2] = -1.0

                """
                Intersecting with cutmesh is more efficient when projecting downwards
                """
                hitLoc2, index_ray2, index_tri2 = cutMeshUpper.ray.intersects_location(ray_origins=coords2,
                                                                                       ray_directions=ray_dir,
                                                                                        multiple_hits=False)
                if len(hitLoc) != len(coords) or len(hitLoc2) != len(hitLoc):
                    # The projections up and down do not match indicating that there maybe some flaw
                    print(hitLoc.shape, hitLoc2.shape, coords.shape)

                    if len(hitLoc2) == 0:
                        # Base plate
                        hitLoc2 = coords2.copy()
                        hitLoc2[:, 2] = 0.0

                        print('Creating Base-plate support')
                    else:
                        print('PROJECTIONS NOT MATCHING')
                        raise ValueError('SUPPORT PROJECTIONS NOT MATCHING - Increase the outer edge gap or increase resolution')

                # Create a surface from the Ray intersection
                surf2 = trimesh.Trimesh(vertices=coords2, faces=poly_tri[1])

                # Extrude the surface based on the heights from the second ray cast
                extrudedBlock = extrudeFace(surf2, None, hitLoc2[:, 2] - 0.05)
                extrudedBlock.export('b.off')

                """
                Take the near net-shape support and obtain the difference with the original part to get clean 
                boundaries for the support
                """
                subprocess.call([self.CORK_PATH, '-diff', 'b.off', 'part.off', 'c.off'])
                blockSupport = trimesh.load_mesh('c.off')

                # Draw the support structures generated
                blockSupport.visual.face_colors[:,:3] = np.random.randint(254, size=3)

                supportBlockRegions.append(blockSupport)

            print('processed support face')

        return supportBlockRegions
