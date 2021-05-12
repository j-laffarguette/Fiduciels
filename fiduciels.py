import numpy as np
from connect import *
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from scipy.ndimage import label, generate_binary_structure
import SimpleITK as sitk
import os


class Patient:
    def __init__(self):
        self.case = get_current("Case")
        self.examination = get_current("Examination")
        self.patient = get_current("Patient")
        self.examination_names = []

    def get_examination_list(self):
        for i in self.case.Examinations:
            self.examination_names.append(i.Name)
        return self.examination_names


class Ct_image(Patient):
    def __init__(self, exam_name):
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


class ROI(Ct_image):
    def __init__(self, exam_name):
        super().__init__(exam_name)
        self.roi_list = self.get_roi_list()
        self.exam_name = exam_name

    def get_roi_list(self):
        rois = self.case.PatientModel.RegionsOfInterest
        return [roi.Name for roi in rois]

    def check_roi(self, roi_name_test):
        roi_check = False
        rois = self.case.PatientModel.RegionsOfInterest
        for roi in rois:
            if roi.Name == roi_name_test:
                roi_check = True
        return roi_check

    def has_contour(self, roi_name_test):
        """ Check if a structure is empty or not"""
        return self.case.PatientModel.StructureSets[self.exam_name].RoiGeometries[roi_name_test].HasContours()

    def get_bounding_box(self, roi_name):
        bound = self.case.PatientModel.StructureSets[self.exam_name].RoiGeometries[roi_name].GetBoundingBox()
        bounds = [[bound[0]['x'], bound[0]['y'], bound[0]['z']], [bound[1]['x'], bound[1]['y'], bound[1]['z']]]
        return bounds

    def look_for_fidu(self, roi_name="Foie"):
        if self.check_roi(roi_name) and self.has_contour(roi_name):
            # Image Creation
            # BE CAREFUL : the first axis is inf/sup in numpy format
            self.image_itk = self.get_itk_image(to_save=False)
            self.image_npy = sitk.GetArrayFromImage(self.image_itk)

            # looking for limits of the roi in which the fiducials are
            bounds = self.get_bounding_box(roi_name)
            limits = [self.image_itk.TransformPhysicalPointToIndex([i[0], i[1], i[2]]) for i in bounds]
            print(f"limits : {limits}")

            # creating a copy of the numpy array containing the image
            image_to_process = np.copy(self.image_npy)

            # thresholding
            maximum = np.amax(self.image_npy)
            print(f'max = {maximum}')
            image_to_process[self.image_npy < 0.7 * maximum] = 0

            # removing values outside the roi bounds
            # z direction
            image_to_process[0:limits[0][2], :, :] = 0
            image_to_process[limits[1][2]:, :, :] = 0
            # x direction
            image_to_process[:, 0:limits[0][0], :] = 0
            image_to_process[:, limits[1][0]:, :] = 0
            # y direction
            image_to_process[:, :, 0:limits[0][1]] = 0
            image_to_process[:, :, limits[1][1]:] = 0

            position = self.find_local_max(image_to_process)
            self.poi_creation(position)
        else:
            raise("Please, make a contour for the liver")

    def find_local_max(self, image_to_process):
        # todo: mettre s dans le init

        # size parameter
        s = 10
        # image_max is the dilation of im with a 10x10 structuring element
        # It is used within peak_local_max function
        image_max = ndi.maximum_filter(image_to_process, size=s, mode='constant')

        # Comparison between image_max and im to find the coordinates of local maxima
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


if __name__ == '__main__':
    # ----- Patient -----
    # Creating patient object
    patient = Patient()
    # Creating a list containing all the examination names
    examinations = patient.get_examination_list()
    roi =
    # exam is one element from the list above

    loop = False

    if loop:
        for exam in examinations:
            # ----- Roi -----
            roi = ROI(exam)
            roi.look_for_fidu()
    else:
        exam = '4D  60%'
        # ----- Roi -----
        roi = ROI(exam)
        roi.look_for_fidu()






