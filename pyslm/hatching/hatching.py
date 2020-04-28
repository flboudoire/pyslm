import pyclipper
import numpy as np

from skimage.measure import approximate_polygon, subdivide_polygon

from ..geometry import LayerGeometry, ContourGeometry, HatchGeometry, Layer


class BaseHatcher:

    PYCLIPPER_SCALEFACTOR = 1e4
    """ 
    The scaling factor used for polygon clipping and offsetting in PyClipper for the decimal component of each polygon
    coordinate. This should be set to inverse of the required decimal tolerance e.g. 0.01 requires a minimum 
    scalefactor of 1e2. 
    """

    def __init__(self):
        pass

    def __str__(self):
        return 'BaseHatcher <{:s}>'.format(self.name)

    def scaleToClipper(self, feature):
        return pyclipper.scale_to_clipper(feature, BaseHatcher.PYCLIPPER_SCALEFACTOR)

    def scaleFromClipper(self, feature):
        return pyclipper.scale_from_clipper(feature, BaseHatcher.PYCLIPPER_SCALEFACTOR)

    @classmethod
    def error(cls):
        """
        Returns the accuracy of the polygon clipping depending on the chosen scale factor :attribute:`~hatching.BaseHatcher.PYCLIPPER_SCALEFACTOR`"
        """
        return 1./cls.PYCLIPPER_SCALEFACTOR

    @staticmethod
    def plot(layer, zPos=0, plotContours=True, plotHatches=True, plotPoints=True, plot3D=True, handle=None):

        import matplotlib.pyplot as plt
        import matplotlib.colors
        import matplotlib.collections as mc

        if handle:
            fig = handle[0]
            ax = handle[1]

        else:
            if plot3D:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = plt.axes(projection='3d')
            else:
                fig, ax = plt.subplots()

        ax.axis('equal')
        plotNormalize = matplotlib.colors.Normalize()

        if plotHatches:
            hatches = [hatchGeom.coords for hatchGeom in layer.hatches]

            if len(hatches) > 0:

                hatches = np.vstack([hatchGeom.coords for hatchGeom in layer.hatches])

                lc = mc.LineCollection(hatches,
                                       colors=plt.cm.rainbow(plotNormalize(np.arange(len(hatches)))),
                                       linewidths=0.5)
                if plot3D:
                    ax.add_collection3d(lc, zs=zPos)
                else:
                    ax.add_collection(lc)
                    midPoints = np.mean(hatches, axis=1)
                    idx6 = np.arange(len(hatches))
                    ax.plot(midPoints[idx6][:, 0], midPoints[idx6][:, 1])

        if plotContours:
            for contourGeom in layer.contours:

                if contourGeom.type == 'inner':
                    lineColor = '#f57900';
                    lineWidth = 1
                elif contourGeom.type == 'outer':
                    lineColor = '#204a87';
                    lineWidth = 1.4
                else:
                    lineColor = 'k';
                    lineWidth = 0.7

                if plot3D:
                    ax.plot(contourGeom.coords[:, 0], contourGeom.coords[:, 1], zs=zPos, color=lineColor,
                            linewidth=lineWidth)
                else:
                    ax.plot(contourGeom.coords[:, 0], contourGeom.coords[:, 1], color=lineColor,
                            linewidth=lineWidth)

        if plotPoints:
            for pointsGeom in layer.points:
                ax.scatter(pointsGeom.coords[:, 0], pointsGeom.coords[:, 1], 'x')

        return fig, ax


    def offsetPolygons(self, polygons, offset: float):
        """
        Offsets the boundaries across a collection of polygons

        :param polygons:
        :param offset: The offset applied to the poylgon
        :return:
        """
        return [self.offsetBoundary(poly, offset) for poly in polygons]


    def offsetBoundary(self, paths, offset: float):
        """
        Offsets a single path for a single polygon

        :param paths:
        :param offset: The offset applied to the poylgon
        :return:
        """
        pc = pyclipper.PyclipperOffset()

        clipperOffset = self.scaleToClipper(offset)

        # Append the paths to libClipper offsetting algorithm
        for path in paths:
            pc.AddPath(self.scaleToClipper(path),
                       pyclipper.JT_ROUND,
                       pyclipper.ET_CLOSEDPOLYGON)

        # Perform the offseting operation
        boundaryOffsetPolys = pc.Execute2(clipperOffset)

        offsetContours = []
        # Convert these nodes back to paths
        for polyChild in boundaryOffsetPolys.Childs:
            offsetContours += self._getChildPaths(polyChild)

        return offsetContours


    def _getChildPaths(self, poly):

        offsetPolys = []

        # Create single closed polygons for each polygon
        paths = [path.Contour for path in poly.Childs]  # Polygon holes
        paths.append(poly.Contour)  # Path holes

        # Append the first point to the end of each path to close loop
        for path in paths:
            path.append(path[0])

        paths = self.scaleFromClipper(paths)

        offsetPolys.append(paths)

        for polyChild in poly.Childs:
            if len(polyChild.Childs) > 0:
                for polyChild2 in polyChild.Childs:
                    offsetPolys += self._getChildPaths(polyChild2)

        return offsetPolys

    def polygonBoundingBox(self, obj) -> np.ndarray:
        """
        Returns the bounding box of the polygon

        :param obj:
        :return: The bounding box of the polygon
        """
        # Path (n,2) coords that

        if not isinstance(obj, list):
            obj = [obj]

        bboxList = []

        for subObj in obj:
            path = np.array(subObj)[:,:2] # Use only coordinates in XY plane
            bboxList.append(np.hstack([np.min(path, axis=0), np.max(path, axis=0)]))

        bboxList = np.vstack(bboxList)
        bbox = np.hstack([np.min(bboxList[:, :2], axis=0), np.max(bboxList[:, -2:], axis=0)])

        return bbox

    def clipLines(self, paths, lines):
        """
        This function clips a series of lines (hatches) across a closed polygon using Pyclipper. Note, the order is NOT
        guaranteed from the list of lines used, so these must be sorted. If order requires preserving this must be
        sequentially performed at a significant computational expense.

        :param paths:
        :param lines: The un-trimmed lines to clip from the boundary

        :return: A list of trimmed lines (paths)
        """

        pc = pyclipper.Pyclipper()

        for path in paths:
            pc.AddPath(self.scaleToClipper(path), pyclipper.PT_CLIP, True)

        # Reshape line list to create n lines with 2 coords(x,y,z)
        lineList = lines.reshape(-1, 2, 3)
        lineList = tuple(map(tuple, lineList))
        lineList = self.scaleToClipper(lineList)

        pc.AddPaths(lineList, pyclipper.PT_SUBJECT, False)

        # Note open paths (lines) have to used PyClipper::Execute2 in order to perform trimming
        result = pc.Execute2(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        # Cast from PolyNode Struct from the result into line paths since this is not a list
        lineOutput = pyclipper.PolyTreeToPaths(result)

        return self.scaleFromClipper(lineOutput)

    def generateHatching(self, paths, hatchSpacing: float, hatchAngle: float = 90.0):
        """
        Generates un-clipped hatches which is guaranteed to cover the entire polygon region base on the maximum extent
        of the polygon bounding box

        :param paths:
        :param hatchSpacing: Hatch Spacing to use
        :param hatchAngle: Hatch angle (degrees) to rotate the scan vectors

        :return: Returns the list of unclipped scan vectors

        """

        # Hatch angle
        theta_h = np.radians(hatchAngle)  # 'rad'

        # Get the bounding box of the paths
        bbox = self.polygonBoundingBox(paths)

        print('bounding box bbox', bbox)
        # Expand the bounding box
        bboxCentre = np.mean(bbox.reshape(2, 2), axis=0)

        # Calculates the diagonal length for which is the longest
        diagonal = bbox[2:] - bboxCentre
        bboxRadius = np.sqrt(diagonal.dot(diagonal))

        # Construct a square which wraps the radius
        x = np.tile(np.arange(-bboxRadius, bboxRadius, hatchSpacing, dtype=np.float32).reshape(-1, 1), (2)).flatten()
        y = np.array([-bboxRadius, bboxRadius]);
        y = np.resize(y, x.shape)
        z = np.arange(0, x.shape[0]/2, 0.5).astype(np.int64)

        coords = np.hstack([x.reshape(-1, 1),
                            y.reshape(-1, 1),
                            z.reshape(-1,1)]);

        print('coords.', coords.shape)
        # Create the rotation matrix
        c, s = np.cos(theta_h), np.sin(theta_h)
        R = np.array([(c, -s, 0),
                      (s, c, 0),
                      (0, 0, 1.0)])

        # Apply the rotation matrix and translate to bounding box centre
        coords = np.matmul(R, coords.T)
        coords = coords.T + np.hstack([bboxCentre, 0.0])

        return coords
if False:

    class InnerHatchRegion(BaseHatcher):

        def __init__(self, parent):

            self._parent = parent
            self._region = []

        def __str__(self):
            return 'InnerHatchRegion <{:s}>'

        @property
        def boundary(self):
            return self._boundary

        def intersection(selfs):
            pc = pyclipper.Pyclipper()

            for path in sliceA:
                pc.AddPath(pyclipper.scale_to_clipper(path, 1000), pyclipper.PT_CLIP, True)

            for path in sliceB:
                pc.AddPath(pyclipper.scale_to_clipper(path, 1000), pyclipper.PT_SUBJECT, True)


            result = pc.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        def isClipped(self):
            pass


class Hatcher(BaseHatcher):
    """
    Provides a generic SLM Hatcher 'recipe' with standard parameters for defining the hatch across regions. This
    includes generating multiple contour offsets and the generic infill (without) pattern. This class may be derived from
    to provide additional or customised behavior.
    """

    def __init__(self):

        # TODO check that the polygon boundary feature type

        # Hatch parametrs

        # Contour information
        self._numInnerContours = 1
        self._numOuterContours = 1
        self._spotCompensation = 0.08  # mm
        self._contourOffset = 1 * self._spotCompensation
        self._volOffsetHatch = self._spotCompensation
        self._clusterDistance = 5  # mm

        # Hatch Information
        self._layerAngleIncrement = 0 # 66 + 2 / 3
        self._hatchDistance = 0.08  # mm
        self._hatchAngle = 45
        self._hatchSortMethod = 'alternate'

    """
    Properties for the Hatch Feature
    """

    @property
    def hatchDistance(self):
        return self._hatchDistance

    @hatchDistance.setter
    def hatchDistance(self, value):
        self._hatchDistance = value

    @property
    def hatchAngle(self):
        return self._hatchAngle

    @hatchAngle.setter
    def hatchAngle(self, value):
        self._hatchAngle = value

    @property
    def layerAngleIncrement(self):
        return self._layerAngleIncrement

    @layerAngleIncrement.setter
    def layerAngleIncrement(self, value):
        self._layerAngleIncrement = value

    @property
    def hatchSortMethod(self):
        return self._hatchSortMethod

    @hatchSortMethod.setter
    def hatchSortMethod(self, value):
        self._hatchSortMethod = value

    @property
    def numInnerContours(self):
        """
        Total number of inner contours to generate for a region (default: 1)
        """
        return self._numInnerContours

    @numInnerContours.setter
    def numInnerContours(self, value):
        self._numInnerContours = value

    @property
    def numOuterContours(self):
        return self._numOuterContours

    @numOuterContours.setter
    def numOuterContours(self, value):
        self._numOuterContours = value

    @property
    def clusterDistance(self):
        return self._clusterDistance

    @clusterDistance.setter
    def clusterDistance(self, value):
        self._clusterDistance = value

    @property
    def spotCompensation(self):
        return self._spotCompensation

    @spotCompensation.setter
    def spotCompensation(self, value):
        self._spotCompensation = value

    @property
    def volumeOffsetHatch(self):
        return self._volOffsetHatch

    @volumeOffsetHatch.setter
    def volumeOffsetHatch(self, value):
        self._volOffsetHatch = value

    def hatch(self, boundaryFeature):

        layer = Layer(0.0)
        # First generate a boundary with the spot compensation applied

        offsetDelta = 0.0
        offsetDelta -= self._spotCompensation

        for i in range(self._numOuterContours):
            offsetDelta -= self._contourOffset
            offsetBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

            for poly in offsetBoundary:
                for path in poly:
                    contourGeometry = ContourGeometry()
                    contourGeometry.coords = np.array(path)[:,:2]
                    contourGeometry.type = "outer"
                    layer.contours.append(contourGeometry)  # Append to the layer

        # Repeat for inner contours
        for i in range(self._numInnerContours):

            offsetDelta -= self._contourOffset
            offsetBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

            for poly in offsetBoundary:
                for path in poly:
                    contourGeometry = ContourGeometry()
                    contourGeometry.coords = np.array(path)[:,:2]
                    contourGeometry.type = "inner"
                    layer.contours.append(contourGeometry)  # Append to the layer

        # The final offset is applied to the boundary

        offsetDelta -= self._volOffsetHatch

        curBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

        scanVectors = []
        scanVectorMidpoints = []

        # Iterate through each closed polygon region in the slice. The currently individually sliced.
        for contour in curBoundary:
            # print('{:=^60} \n'.format(' Generating hatches '))

            paths = contour

            # Hatch angle will change per layer
            # TODO change the layer angle increment
            layerHatchAngle = np.mod(self._hatchAngle + self._layerAngleIncrement, 180)

            # The layer hatch angle needs to be bound by +ve X vector (i.e. -90 < theta_h < 90 )
            if layerHatchAngle > 90:
                layerHatchAngle = layerHatchAngle - 180

            # Generate the un-clipped hatch regions based on the layer hatchAngle and hatch distance
            hatches = self.generateHatching(paths, self._hatchDistance, layerHatchAngle)

            # Clip the hatch fill to the boundary
            clippedPaths = self.clipLines(paths, hatches)

            # Merge the lines together
            if len(clippedPaths) == 0:
                continue

            clippedLines = np.transpose(np.dstack(clippedPaths), axes=[2, 0, 1])

            # Extract only x-y coordinates
            clippedLines = clippedLines[:,:,:3]
            id = np.argsort(clippedLines[:,0,2])
            clippedLines = clippedLines[id,:,:]
            print('clipped lines shapes', clippedLines.shape)
            scanVectors.append(clippedLines)


        if len(clippedLines) > 0:
            # Scan vectors have been

            if False:

                # The greedy sort algorithm strugles the angle is less than 40
                if np.abs(layerHatchAngle) > 45:

                    # Since origin is used, the angle needs to be modified for vector to point in +Y direction
                    if layerHatchAngle < 0:
                        scanSortLayerHatchAngle = layerHatchAngle + 180
                    else:
                        scanSortLayerHatchAngle = layerHatchAngle
                    sortedHatchIdx = self.greedySortScanVectors(np.vstack(scanVectorMidpoints),
                                                                scanSortLayerHatchAngle,
                                                                self._hatchDistance * 5,
                                                                True)
                    # sortedHatchIdx = self.sortScanVectors(np.vstack(scanVectors), layerHatchAngle )
                else:
                    sortedHatchIdx = self.greedySortScanVectors(np.vstack(scanVectorMidpoints),
                                                                layerHatchAngle,
                                                                self._hatchDistance * 5)

                # Finally sort the scan vectors for the layer
                if self._hatchSortMethod == 'alternate':
                    hatchVectors = self.sortScanOrder(np.vstack(scanVectors)[sortedHatchIdx])
                else:
                    hatchVectors = np.vstack(scanVectors)[sortedHatchIdx]

            hatchVectors = np.vstack(scanVectors)
            # Construct a HatchGeometry containg the list of points
            hatchGeom = HatchGeometry()
            hatchGeom.coords = hatchVectors[:,:,:2]

            layer.hatches.append(hatchGeom)

        return layer

class StripeHatcher(Hatcher):

    def __init__(self):
        self._stripeWidth = 5.0

    @property
    def stripeWidth(self) -> float:
        return self._stripeWidth

    @stripeWidth.setter
    def stripeWidth(self, width):
        self._stripeWidth = width

    def hatch(self, boundaryFeature):

        layer = Layer(0.0)
        # First generate a boundary with the spot compensation applied

        offsetDelta = 0.0
        offsetDelta -= self._spotCompensation

        for i in range(self._numOuterContours):
            offsetDelta -= self._contourOffset
            offsetBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

            for poly in offsetBoundary:
                for path in poly:
                    contourGeometry = ContourGeometry()
                    contourGeometry.coords = np.array(path)
                    contourGeometry.type = "outer"
                    layer.contours.append(contourGeometry)  # Append to the layer

        # Repeat for inner contours
        for i in range(self._numInnerContours):

            offsetDelta -= self._contourOffset
            offsetBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

            for poly in offsetBoundary:
                for path in poly:
                    contourGeometry = ContourGeometry()
                    contourGeometry.coords = np.array(path)
                    contourGeometry.type = "inner"
                    layer.contours.append(contourGeometry)  # Append to the layer

        # The final offset is applied to the boundary

        offsetDelta -= self._volOffsetHatch

        curBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

        scanVectors = []
        scanVectorMidpoints = []

        # Iterate through each closed polygon region in the slice. The currently individually sliced.
        for contour in curBoundary:
            # print('{:=^60} \n'.format(' Generating hatches '))

            paths = contour


            # Hatch angle will change per layer
            # TODO change the layer angle increment
            layerHatchAngle = np.mod(self._hatchAngle + self._layerAngleIncrement, 180)

            # The layer hatch angle needs to be bound by +ve X vector (i.e. -90 < theta_h < 90 )
            if layerHatchAngle > 90:
                layerHatchAngle = layerHatchAngle - 180

            # Generate the un-clipped hatch regions based on the layer hatchAngle and hatch distance
            hatches = self.generateHatching(paths, self._hatchDistance, layerHatchAngle)

            # Clip the hatch fill to the boundary
            clippedPaths = self.clipLines(paths, hatches)

            # Merge the lines together
            if len(clippedPaths) == 0:
                continue

            clippedLines = np.transpose(np.dstack(clippedPaths), axes=[2, 0, 1])

            scanVectors.append(clippedLines)

            hatchVectors = np.vstack(scanVectors)
            # Construct a HatchGeometry containg the list of points
            hatchGeom = HatchGeometry()
            hatchGeom.coords = hatchVectors

            layer.hatches.append(hatchGeom)

        return layer

    def generateStripeHatching(self, paths, hatchSpacing: float, hatchAngle: float = 90.0):
        """
        Generates un-clipped hatches which is guaranteed to cover the entire polygon region base on the maximum extent
        of the polygon bounding box

        :param paths:
        :param hatchSpacing: Hatch Spacing to use
        :param hatchAngle: Hatch angle (degrees) to rotate the scan vectors

        :return: Returns the list of unclipped scan vectors

        """

        # Hatch angle
        theta_h = np.radians(hatchAngle)  # 'rad'

        # Get the bounding box of the paths
        bbox = self.polygonBoundingBox(paths)

        # Expand the bounding box
        bboxCentre = np.mean(bbox.reshape(2, 2), axis=0)

        # Calculates the diagonal length for which is the longest
        diagonal = bbox[2:] - bboxCentre
        bboxRadius = np.sqrt(diagonal.dot(diagonal))

        # Construct a square which wraps the radius
        x = np.tile(np.arange(-bboxRadius, bboxRadius, hatchSpacing, dtype=np.float32).reshape(-1, 1), (2)).flatten()
        y = np.array([-bboxRadius, bboxRadius]);
        y = np.resize(y, x.shape)

        coords = np.hstack([x.reshape(-1, 1),
                            y.reshape(-1, 1)]);

        # Create the rotation matrix
        c, s = np.cos(theta_h), np.sin(theta_h)
        R = np.array([(c, -s),
                      (s, c)])

        # Apply the rotation matrix and translate to bounding box centre
        coords = np.matmul(R, coords.T)
        coords = coords.T + bboxCentre

        return coords

    def generateCheckerHatching(self, paths, hatchSpacing: float, hatchAngle: float = 90.0):
        """
        Generates un-clipped hatches which is guaranteed to cover the entire polygon region base on the maximum extent
        of the polygon bounding box

        :param paths:
        :param hatchSpacing: Hatch Spacing to use
        :param hatchAngle: Hatch angle (degrees) to rotate the scan vectors

        :return: Returns the list of unclipped scan vectors

        """

        # Hatch angle
        theta_h = np.radians(hatchAngle)  # 'rad'

        # Get the bounding box of the paths
        bbox = self.polygonBoundingBox(paths)

        # Expand the bounding box
        bboxCentre = np.mean(bbox.reshape(2, 2), axis=0)

        # Calculates the diagonal length for which is the longest
        diagonal = bbox[2:] - bboxCentre
        bboxRadius = np.sqrt(diagonal.dot(diagonal))

        # Construct a square which wraps the radius
        x = np.tile(np.arange(-bboxRadius, bboxRadius, hatchSpacing, dtype=np.float32).reshape(-1, 1), (2)).flatten()
        y = np.array([-bboxRadius, bboxRadius]);
        y = np.resize(y, x.shape)

        coords = np.hstack([x.reshape(-1, 1),
                            y.reshape(-1, 1)]);

        # Create the rotation matrix
        c, s = np.cos(theta_h), np.sin(theta_h)
        R = np.array([(c, -s),
                      (s, c)])

        # Apply the rotation matrix and translate to bounding box centre
        coords = np.matmul(R, coords.T)
        coords = coords.T + bboxCentre

        return coords
