import numpy as np
from connect import *
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
        self.image_ct = self.get_itk_image(to_save=False)

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
        # todo: repair the conversion between position to index to be able to use the bounds
        if self.check_roi(roi_name) and self.has_contour(roi_name):
            bounds = self.get_bounding_box(roi_name)
            limits = [self.image_ct.TransformPhysicalPointToIndex([i[0], i[1], i[2]]) for i in bounds]
            print('Limits :')
            print(limits)
            # self.find_local_max()
        else:
            print("Please, make a contour for the liver")

        self.get_index_from_physical_point(self.origin)

    def get_index_from_physical_point(self, physical_point):
        (x, y, z) = physical_point
        (i, j, k) = self.image_ct.TransformPhysicalPointToContinuousIndex((x, y, z))
        print(f"x : {x}, y : {y}, z : {z} correspond to point {i / 10} , {j / 10}, {k / 10}")
        return i, j, k

    # def get_physical_point_from_index(self, column, row, slice):
    #     [i, j, k] = self.image_ct.TransformPhysicalPointToContinuousIndex([x, y, z])
    #     print(f"x : {x}, y : {y}, z : {z} correspond to point {i} , {j}, {k}")
    #     return
    #
    # def find_local_max(self):
    #     # thresholding
    #     self.image_to_process[self.image_to_process < 2000] = 0
    #     self.image_to_process[self.image_to_process > 2000] = 1
    #
    #     coordinates = peak_local_max(self.image_to_process)
    #
    #     s = np.ones([3, 3, 3])
    #     labeled_array, num_features = label(self.image_to_process, structure = s)
    #     print("There are " + str(num_features) + " fiducials")


if __name__ == '__main__':
    # ----- Patient -----
    # Creating patient object
    patient = Patient()

    # Creating a list containing all the examination names
    examinations = patient.get_examination_list()
    # exam is one element from the list above
    # exam = examinations[0]
    exam = '2021.04.08  50% Contourage Foie'

    # ----- Ct -----
    # ct = Ct_image(exam)

    # ----- Roi -----
    roi = ROI(exam)
    roi.look_for_fidu()

    # ----- Fidu -----

    # foie = roi.has_contour('Foie')
    # bounding = roi.get_bounding_box('Foie')
    #
    # print(ct.get_index_from_position(-199.609375, -419.109375, -149.5))
    # print(ct.get_position_from_index(0, 0, 0))
    #
    # # path = os.path.dirname(os.path.realpath(__file__))
    # # print(path)
    # # with open(os.path.join(path, 'data.npy'), 'wb') as f:
    # #     np.save(f, image_ct)

# #GARBAGE PART
# # POI creation
# # =============================================================================
#
# name = []
#
# for i in range(n_unique_POI):
#     name.append("Fidu " + str(i + 1))
#     case.PatientModel.CreatePoi(Name=name[i], Color="Yellow", VisualizationDiameter=1, Type="Undefined")
#
# # =============================================================================
# # POI update for each exam
# # =============================================================================
#
# exam_0 = coordfidus[0][3]
#
# for i in range(n_fidus):
#     fidu = coordfidus[i]
#
#     x = fidu[0];
#     y = fidu[1];
#     z = fidu[2];
#     exam = fidu[3]
#     case.PatientModel.StructureSets[exam].PoiGeometries[name[i % n_unique_POI]].Point = {'x': x, 'y': y, 'z': z}
