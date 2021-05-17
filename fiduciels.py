import numpy as np
from connect import *
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from scipy.ndimage import label, generate_binary_structure
from skimage.draw import polygon2mask
import SimpleITK as sitk
import os


def check_roi(case, roi_to_check):
    # this method checks if a toi exists
    roi_check = False
    rois = case.PatientModel.RegionsOfInterest
    for roi in rois:
        if roi.Name == roi_to_check:
            roi_check = True
    return roi_check


def has_contour(case, examination, roi_to_check):
    """ Check if a structure is empty or not"""
    return case.PatientModel.StructureSets[examination].RoiGeometries[roi_to_check].HasContours()


def get_bounding_box(case, examination, roi):
    # get coordinates of two point that create a box around the roi_name
    bound = case.PatientModel.StructureSets[examination].RoiGeometries[roi].GetBoundingBox()
    bounds = [[bound[0]['x'], bound[0]['y'], bound[0]['z']], [bound[1]['x'], bound[1]['y'], bound[1]['z']]]
    return bounds


class Patient:
    def __init__(self):
        self.case = get_current("Case")
        self.examination = get_current("Examination")
        self.patient = get_current("Patient")
        self.examination_names = []
        self.roi_list = []

    def get_examination_list(self):
        for i in self.case.Examinations:
            self.examination_names.append(i.Name)
        return self.examination_names

    def get_roi_list(self):
        rois = self.case.PatientModel.RegionsOfInterest
        self.roi_list = [roi.Name for roi in rois]
        return self.roi_list


class Image(Patient):
    def __init__(self, exam_name, roi_name=None):
        super().__init__()
        self.exam_name = exam_name
        self.data = self.case.Examinations[exam_name].Series[0].ImageStack.PixelData
        self.n_data = self.data.size
        self.n_voxels = int(self.n_data / 2)

        # Image Shape
        self.rows = self.case.Examinations[self.exam_name].Series[0].ImageStack.NrPixels.x
        self.columns = self.case.Examinations[self.exam_name].Series[0].ImageStack.NrPixels.y
        self.depth = int(self.n_data / 2 / self.rows / self.columns)

        # Depth
        self.slice_position = self.case.Examinations[self.exam_name].Series[0].ImageStack.SlicePositions
        self.slice_thickness = abs(self.slice_position[1] - self.slice_position[0])

        # Direction
        column_direction_x = self.case.Examinations[self.exam_name].Series[0].ImageStack.ColumnDirection.x
        column_direction_y = self.case.Examinations[self.exam_name].Series[0].ImageStack.ColumnDirection.y
        column_direction_z = self.case.Examinations[self.exam_name].Series[0].ImageStack.ColumnDirection.z

        row_direction_x = self.case.Examinations[self.exam_name].Series[0].ImageStack.RowDirection.x
        row_direction_y = self.case.Examinations[self.exam_name].Series[0].ImageStack.RowDirection.y
        row_direction_z = self.case.Examinations[self.exam_name].Series[0].ImageStack.RowDirection.z

        slice_direction_x = self.case.Examinations[self.exam_name].Series[0].ImageStack.SliceDirection.x
        slice_direction_y = self.case.Examinations[self.exam_name].Series[0].ImageStack.SliceDirection.y
        slice_direction_z = self.case.Examinations[self.exam_name].Series[0].ImageStack.SliceDirection.z

        self.direction = (column_direction_x, column_direction_y, column_direction_z,
                          row_direction_x, row_direction_y, row_direction_z,
                          slice_direction_x, slice_direction_y, slice_direction_z)

        # Image Corners
        self.x_corner = self.case.Examinations[self.exam_name].Series[0].ImageStack.Corner.x
        self.y_corner = self.case.Examinations[self.exam_name].Series[0].ImageStack.Corner.y
        self.z_corner = self.case.Examinations[self.exam_name].Series[0].ImageStack.Corner.z
        self.origin = (self.x_corner, self.y_corner, self.z_corner)

        # Pixel size
        self.x_pixel_size = self.case.Examinations[self.exam_name].Series[0].ImageStack.PixelSize.x
        self.y_pixel_size = self.case.Examinations[self.exam_name].Series[0].ImageStack.PixelSize.y
        self.spacing = (self.x_pixel_size, self.y_pixel_size, self.slice_thickness)

        # Conversion Parameters
        self.intercept = self.case.Examinations[self.exam_name].Series[
            0].ImageStack.ConversionParameters.RescaleIntercept
        self.slope = self.case.Examinations[self.exam_name].Series[0].ImageStack.ConversionParameters.RescaleSlope

        # Image Creation
        # BE CAREFUL : the first axis is inf/sup in numpy format
        self.image_itk = None
        self.image_npy = None

        # Roi
        self.roi_name = roi_name
        self.roi_mask_npy = None

    def get_itk_image(self, to_save=False):
        """Return a simpleITK image object"""
        print("----> Starting CT conversion ...")
        result = np.zeros(self.n_voxels)
        b1 = self.data[0:-1:2]
        b2 = self.data[1::2]
        result[b2 > 128] = b1[b2 > 128] + 256 * (b2[b2 > 128] - 128) - 65536
        result[b2 <= 128] = b1[b2 <= 128] + 256 * b2[b2 <= 128]
        result = self.slope * result + self.intercept
        result = np.reshape(result, [self.depth, self.columns, self.rows])

        # numpy file saving
        if to_save:
            # If needed, this saves the numpy array to file for external use
            path = os.getcwd()
            with open(os.path.join(path, "results.npy"), 'wb') as f:
                np.save(f, result)

        # Simple ITK conversion
        itk_image = sitk.GetImageFromArray(result)
        itk_image.SetDirection(self.direction)
        itk_image.SetOrigin(self.origin)
        itk_image.SetSpacing(self.spacing)

        print(itk_image.GetSize())
        print(itk_image.GetOrigin())
        print(itk_image.GetSpacing())
        print(itk_image.GetDirection())
        return itk_image

    def get_mask_from_roi(self):
        # This method needs a roi_name
        if self.roi_name is None:
            raise Exception("Please, give a roi name to the Image Object if you want to use this method")

        # This method needs the creation of the itk image if not already done
        if self.image_itk is None:
            self.image_itk = self.get_itk_image(to_save=False)
            self.image_npy = sitk.GetArrayFromImage(self.image_itk)
        try:
            check_roi(self.case, self.roi_name)
            has_contour(self.case, self.exam_name, self.roi_name)
        except:
            raise Exception("This roi does not exist")

        print(f"----> Starting mask creation for {self.roi_name} on {self.exam_name} ...")
        # Creation of an empty array that has the same size as the original image
        mask = np.zeros_like(self.image_npy)

        # sl is a list of all the dots for one single slice
        for sl in self.case.PatientModel.StructureSets[self.exam_name].RoiGeometries[
            self.roi_name].PrimaryShape.Contours:
            # for each slice, one creates an array that will contain all the dots coordinates.
            # This array is initialized by using np.ones like this -> coordinates = [[1,1,1] , [1,1,1] ,...]
            # it will be filled with coordinates like this -> [[x1,y1,z1],[x2,y2,z2], ....]

            n_dots = len(sl)
            coordinates = np.ones([n_dots, 3])

            slice_number = 0
            # dot contains three coordinates (in mm) for one dot of the contour. The three coordinates are needed
            # for the conversion from positions in mm to index
            for index, dot in enumerate(sl):
                coordinates[index] = self.image_itk.TransformPhysicalPointToIndex([dot.x, dot.y, dot.z])
                if index == 0:
                    slice_number = coordinates[index][2]
            # polygon2mask creates a mask only for 2D images. So ones needs to suppress the third coordinate (number
            # of the slice)
            image_shape = [self.columns, self.rows]
            coordinates_xy = [c[0:2] for c in coordinates]
            temp_mask = polygon2mask(image_shape, coordinates_xy)
            print('mask creation ok!')

            mask[int(slice_number), :, :] = temp_mask
        mask.astype(int)
        self.roi_mask_npy = mask
        return mask


class Fidu(Image):
    def __init__(self, exam_name, roi_name=None):
        super().__init__(exam_name, roi_name)

        # Fidu parameter: distance in voxel between two spots
        self.fidu_size = 10

        # Automatic Fidu seeking
        self.look_for_fidu()

    def find_local_max(self, image_to_process):
        s = self.fidu_size

        footprint = np.ones([s, s, s])
        coordinates = peak_local_max(image_to_process, min_distance=s, footprint=footprint)

        positions = []
        for coord in coordinates:
            # conversion of coordinates into int values
            coord = tuple(map(float, coord))
            # replacing z by x
            # todo: commentaire
            coord = [coord[1], coord[2], coord[0]]
            print(coord)
            position = self.image_itk.TransformContinuousIndexToPhysicalPoint(coord)
            print(position)
            positions.append(position)

        # sorting coordinates by z
        positions = sorted(positions, key=lambda x: x[2])

        return positions

    def poi_creation(self, coordinates):
        for i, coords in enumerate(coordinates):
            name = "Fidu " + str(i + 1)
            try:
                # POI list creation in RS
                self.case.PatientModel.CreatePoi(Name=name, Color="Yellow", VisualizationDiameter=1,
                                                 Type="Undefined")
            except:
                # todo : what if the points are already there?
                print('error')
            # POI assignation
            x, z, y = coords
            self.case.PatientModel.StructureSets[self.exam_name].PoiGeometries[name].Point = {
                'x': x, 'y': z, 'z': y}

    def look_for_fidu(self):
        if check_roi(self.case, self.roi_name) and has_contour(self.case, self.exam_name, self.roi_name):
            # Image Creation
            # BE CAREFUL : the first axis is inf/sup in numpy format
            self.image_itk = self.get_itk_image(to_save=False)
            self.image_npy = sitk.GetArrayFromImage(self.image_itk)

            # creating a copy of the numpy array containing the image
            image_to_process = np.copy(self.image_npy)

            # thresholding
            maximum = np.amax(self.image_npy)
            print(f'max = {maximum}')
            image_to_process[self.image_npy < 0.7 * maximum] = 0

            # Applying mask using structure:
            self.get_mask_from_roi()
            image_to_process = image_to_process * self.roi_mask_npy

            # Seeking the fiducials
            position = self.find_local_max(image_to_process)

            # creation of POIs in RS
            self.poi_creation(position)
        else:
            print(f"Please, make a contour for the roi {self.roi_name}")


if __name__ == '__main__':
    # ----- Patient -----
    # Creating patient object
    patient = Patient()

    # Creating a list containing all the examination names
    examinations = patient.get_examination_list()
    roi = patient.get_roi_list()

    # Creating Fidu object
    fidu = Fidu('4D  60%', 'Foie')
