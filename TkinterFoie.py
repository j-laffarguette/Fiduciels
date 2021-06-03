import os
import tkinter
from tkinter import *
from math import sqrt
import SimpleITK as sitk
import numpy as np
from connect import *
from skimage.draw import polygon2mask
from skimage.feature import peak_local_max
from skimage.filters import gaussian


# todo : use the center of mass and not only the maximum
# ----------------------------------------------------
# Functions
# ----------------------------------------------------

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


# ----------------------------------------------------
# Classes
# ----------------------------------------------------

class Patient:
    def __init__(self):

        self.case = get_current("Case")
        self.examination = get_current("Examination")
        self.patient = get_current("Patient")
        self.examination_names = []
        self.irm_names = []
        self.roi_list = []

    def get_ct_list(self):
        for exam in self.case.Examinations:
            if exam.EquipmentInfo.Modality == 'CT':
                self.examination_names.append(exam.Name)
        return self.examination_names

    def get_irm_list(self):
        name, modality = [], []
        for exam in self.case.Examinations:
            if exam.EquipmentInfo.Modality == 'MR':
                name.append(exam.Name)
                modality.append(exam.GetProtocolName())
        self.irm_names = list(zip(name, modality))
        return self.irm_names

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
        self.image_to_process = None

        # Roi
        self.roi_name = roi_name
        self.roi_mask_npy = None

        # IRM
        # name of the dixon IRM
        self.dixon_name = None
        # self.get_dixon_name()

    def get_itk_image(self):
        """Return a simpleITK image object"""
        print("----> Starting CT conversion ...")
        result = np.zeros(self.n_voxels)
        b1 = self.data[0:-1:2]
        b2 = self.data[1::2]
        result[b2 > 128] = b1[b2 > 128] + 256 * (b2[b2 > 128] - 128) - 65536
        result[b2 <= 128] = b1[b2 <= 128] + 256 * b2[b2 <= 128]
        result = self.slope * result + self.intercept
        result = np.reshape(result, [self.depth, self.columns, self.rows])

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

    def get_mask_from_roi(self, roi_name=None):

        # This method needs a roi_name
        if roi_name is None:
            if self.roi_name is None:
                raise Exception("Please, give a roi name to the Image Object if you want to use this method")
        else:
            self.roi_name = roi_name

        # This method needs the creation of the itk image if not already done
        if self.image_itk is None:
            self.image_itk = self.get_itk_image()
            self.image_npy = sitk.GetArrayFromImage(self.image_itk)
        try:
            check_roi(self.case, self.roi_name)
            has_contour(self.case, self.exam_name, self.roi_name)
        except:
            raise Exception("This roi does not exist")

        # Converting the Roi from voxel type to contours type
        self.case.PatientModel.StructureSets[self.exam_name].RoiGeometries[self.roi_name].SetRepresentation(
            Representation='Contours')

        # Starting creation
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

            mask[int(slice_number), :, :] = temp_mask

        print('mask creation ok!')
        mask.astype(int)

        self.roi_mask_npy = mask
        return mask

    def get_dixon_name(self):
        # todo : check if there might be more than one dixon
        irm_list = self.get_irm_list()
        for irm_modality in irm_list:
            if "DIXON" in irm_modality[1]:
                self.dixon_name = irm_modality[0]

    def create_IRM_external(self):

        if not check_roi(self.case, 'External'):
            self.case.PatientModel.CreateRoi(Name=r"External", Color="Green", Type="External", TissueName=r"",
                                             RbeCellTypeName=None, RoiMaterial=None)

        self.case.PatientModel.RegionsOfInterest['External'].CreateExternalGeometry(
            Examination=self.case.Examinations[self.dixon_name],
            ThresholdLevel=15)


class Fidu(Image):
    def __init__(self, exam_name, roi_name=None, threshold_abs=None, threshold_relative=None):
        super().__init__(exam_name, roi_name)

        # Fidu parameter: distance in voxel between two spots
        self.fidu_size = 10
        self.diameter = 5  # in mm

        # thresholds
        # if nothing is inserted :  threshold abs with value 1600
        # if threshold type == relative : threshold value = threshold_relative * max
        self.threshold_value = None
        self.maximum = None

        if threshold_abs:
            self.threshold_value = threshold_abs
            self.threshold_type = 'absolute'

        if threshold_relative:
            self.threshold_relative = threshold_relative
            self.threshold_type = 'relative'

        else:
            self.threshold_type = 'relative'
            self.threshold_relative = .48
            # self.threshold_value = 1600

        # Fidu names in RS
        self.fidu_prefix_names = "Fidu "

        # Automatic Fidu seeking
        # self.look_for_fidu()

        # Box size (cm)
        self.box_size = 1
        self.radius = 1.25

    def find_local_max(self, image_to_process):
        s = self.fidu_size
        footprint = np.ones([s, s, s])
        coordinates = peak_local_max(image_to_process, min_distance=s, footprint=footprint)
        return coordinates

    def look_for_fidu(self, filtering=True):

        # If a detection is not possible, one keeps in mind the exam name and retry it at the end
        keep_in_mind = None
        roi_list = None

        if check_roi(self.case, self.roi_name) and has_contour(self.case, self.exam_name, self.roi_name):
            print('---------------------------------------')
            print(f'\nRECHERCHE DE FIDU POUR --> {self.exam_name}\n')
            print('---------------------------------------')
            # Image Creation
            # BE CAREFUL : the first axis is inf/sup in numpy format
            self.image_itk = self.get_itk_image()
            self.image_npy = sitk.GetArrayFromImage(self.image_itk)

            if filtering:
                if self.slice_thickness < 0.3:
                    print(f'filtering with a high sigma value')
                    self.image_npy = gaussian(self.image_npy, sigma=0.5)
                elif self.slice_thickness == 0.3:
                    print(f'filtering with a low sigma value')
                    self.image_npy = gaussian(self.image_npy, sigma=0.1)

            # creating a copy of the numpy array containing the image
            image_to_process = np.copy(self.image_npy)

            # Applying mask using structure:
            self.get_mask_from_roi()
            image_to_process = image_to_process * self.roi_mask_npy
            # this attribute is used by post processing
            self.image_to_process = np.copy(image_to_process)

            # thresholding
            self.maximum = np.amax(self.image_npy)

            if self.threshold_type == 'relative':
                self.threshold_value = self.threshold_relative * self.maximum

            print(f'max = {self.maximum}')
            print(f'threshold = {self.threshold_value}')
            image_to_process[self.image_npy < self.threshold_value] = 0

            # Seeking the fiducials
            coordinates = self.find_local_max(image_to_process)

            # post processing
            coordinates = self.post_processing(coordinates)

            # Converting coordinates to positions (in mm)
            positions = self.get_position_from_coordinates(coordinates)

            # sorting coordinates compared to the image origin
            d = [sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in positions]
            positions = [x for _, x in sorted(zip(d, positions))]
            print(positions)
            # creation of POIs in RS
            self.poi_creation(positions)

            # roi_list = self.create_sphere_roi()


        else:
            keep_in_mind = str(self.exam_name)
            print(f"Please, make a contour for the roi {self.roi_name}")

        return keep_in_mind, roi_list

    def post_processing(self, coordinates):
        print("\nPost Processing (looking for artefacts)...")
        s = self.fidu_size

        coord_new = []
        for index, coord in enumerate(coordinates):
            i = coord[0]
            j = coord[1]
            k = coord[2]
            matrix = self.image_to_process[i - s:i + s,
                     j - s:j + s,
                     k - s:k + s]
            hist, bin_edges = np.histogram(matrix, bins=[-1500, -200, self.threshold_value, self.maximum])
            print(hist)
            if (hist[0] > 0) and (hist[2] < 100):
                print("-> La tache numéro : " + str(index + 1) + " contient des artefacts!\n")
                coord_new.append(coord)
        return coord_new

    # def post_process_distance(self, coordinates):
    #     print("\nPost Processing (distance verification)...")

    def get_position_from_coordinates(self, coordinates):

        # position will contain coordinates of all the fidus (in mm)
        positions = []

        for coord in coordinates:
            # conversion of coordinates (x,y,z) into float values
            coord = tuple(map(float, coord))
            # replacing z by x
            coord = [coord[1], coord[2], coord[0]]
            # transforming coord in position
            position = self.image_itk.TransformContinuousIndexToPhysicalPoint(coord)
            positions.append(position)

        return positions

    def poi_creation(self, coordinates):
        for i, coords in enumerate(coordinates):
            name = self.fidu_prefix_names + str(i + 1)
            try:
                # POI list creation in RS
                self.case.PatientModel.CreatePoi(Name=name, Color="Yellow", VisualizationDiameter=1,
                                                 Type="Undefined")
            except:
                print('The POI already exists.')
            # POI assignation
            x, z, y = coords
            self.case.PatientModel.StructureSets[self.exam_name].PoiGeometries[name].Point = {
                'x': x, 'y': z, 'z': y}

    def look_in_irm(self, roi='Foie'):

        # Roi inside which one is looking for the fidus (usually -> 'Foie')
        input_roi = roi

        # The Name of DIXON IRM is self.dixon_name
        # 1- ones needs to register IRM and CT and to copy little boxes centered to the fidu and copy them to the IRM

        self.get_dixon_name()

        # External Creation on dixon acquisition
        self.create_IRM_external()

        # Setting CT as primary
        self.case.Examinations[self.exam_name].SetPrimary()
        self.case.Examinations[self.dixon_name].SetSecondary()

        # Rigid registration between IRM and CT
        self.rigid_registration(self.dixon_name, self.exam_name, self.roi_name)

        # Creating little spheres
        coord, roi_list = self.create_sphere_roi()

        # Copying Rois one by one
        for roi_name in roi_list:
            print(roi_name)
            print(self.exam_name)
            self.copy_roi(source=self.exam_name, target=self.dixon_name, roi=roi_name)

        # Copy of the liver roi
        self.copy_roi(source=self.exam_name, target=self.dixon_name, roi=roi)

        # -------------------------------------------------------
        # FIDU CREATION
        # -------------------------------------------------------
        # Creating a Fidu object with dixon image as main exam attribute
        obj_irm = Fidu(self.dixon_name)
        # Creating an ITK image and then an associated numpy array
        image_irm = obj_irm.get_itk_image()
        img_npy = sitk.GetArrayFromImage(image_irm)

        # Creating a mask with the input roi (usually -> Foie)
        roi_mask = obj_irm.get_mask_from_roi(input_roi)

        # Then, fidu creation for all the little rois
        positions = []
        for roi in roi_list:
            image_to_process = np.copy(img_npy)

            # creating mask for each roi and multiplying the image by the both masks
            mask = obj_irm.get_mask_from_roi(roi)

            image_to_process = image_to_process * mask * roi_mask

            # every voxel that have zero value get max value, then invert all the value and get them positive
            maximum = np.amax(image_to_process)
            image_to_process[image_to_process == 0] = maximum
            image_to_process = image_to_process * (-1) + maximum

            # Applying gaussian filter
            image_to_process = gaussian(image_to_process, sigma=0.5)

            # Deleting all low values
            maximum = np.amax(image_to_process)
            image_to_process[image_to_process < 0.6 * maximum] = 0

            # finding local max
            coord = obj_irm.find_local_max(image_to_process)
            position = obj_irm.get_position_from_coordinates(coord)
            positions.append(position[0])

            obj_irm.case.PatientModel.RegionsOfInterest[roi].DeleteRoi()

        print(positions)
        obj_irm.poi_creation(positions)
        # obj_irm.create_sphere_roi()

    def copy_roi(self, source, target, roi):
        try:
            self.case.PatientModel.CopyRoiGeometries(SourceExamination=self.case.Examinations[source],
                                                     TargetExaminationNames=[target],
                                                     RoiNames=[roi])
        except:
            print('Unable to copy the structure')

    def rigid_registration(self, floating_exam, reference_exam, focus_roi):
        # Rigid registration between floating and reference with focus_roi
        self.case.ComputeRigidImageRegistration(FloatingExaminationName=floating_exam,
                                                ReferenceExaminationName=reference_exam,
                                                UseOnlyTranslations=False, HighWeightOnBones=False,
                                                InitializeImages=True,
                                                FocusRoisNames=[focus_roi])

    def create_sphere_roi(self):
        """ poi is a part of case.PatientModel.PointsOfInterest"""
        coordinates, roi_list = [], []
        for poi in self.case.PatientModel.PointsOfInterest:
            # Work only with the fidus named 'Fidu 1" etc.
            if self.fidu_prefix_names in poi.Name:
                x = self.case.PatientModel.StructureSets[self.exam_name].PoiGeometries[poi.Name].Point.x
                y = self.case.PatientModel.StructureSets[self.exam_name].PoiGeometries[poi.Name].Point.y
                z = self.case.PatientModel.StructureSets[self.exam_name].PoiGeometries[poi.Name].Point.z
                coordinates.append([x, y, z])
                roi_name = poi.Name + '_roi'
                roi_list.append(roi_name)

                try:
                    self.case.PatientModel.CreateRoi(Name=roi_name, Color="Yellow", Type="Marker", TissueName=None,
                                                     RbeCellTypeName=None, RoiMaterial=None)
                except:
                    print(f'{roi_name} already exists!')

                self.case.PatientModel.RegionsOfInterest[roi_name].CreateSphereGeometry(
                    Radius=self.radius, Examination=self.case.Examinations[self.exam_name],
                    Center={'x': x, 'y': y, 'z': z}, Representation="TriangleMesh", VoxelSize=None)
        return coordinates, roi_list


class TkFOR(tkinter.Tk):
    def __init__(self, patient_object, roi):

        tkinter.Tk.__init__(self)
        
        ## Liste de rois
        self.roi = roi
        
        ## Info Patient 
        self.patient = patient_object
        self.list = self.patient.get_ct_list()
        
        
        self.main_exam = None
        self.roi_list = []
        
        ## Dictionnaire (nomexam + IntVar (tk 1/0) + init
        self.examDict = {}
        self.__createDict()
        
        ## Recherche IRM
        self.checkForIRM = IntVar()
        
        self.__createWidgets()

    def __createDict(self):
        # todo : roialgebra pour copy roi en Foie_rechercheFidu pour script et suppression a la fin
    
         # todo : trier le dictionnaire en fonction des fidus dédoubles 
        # for exam in self.list :
            # if 50% : 
                # list 50 = []
            # else : 
                # list.append(exam)
        # self.sublist = list50 + list
        
        # Pour le groupe 4D : u moins 1 4D avec contours
        # Projette sur tout le monde 
        # regarde qui n'est pas dedouble 
        
        for exam in self.list:
            self.examDict[exam] = IntVar()

    def __execute(self):
        compteur = 0
        kept_in_mind = []
        for image in self.examDict:
        
       

            if self.examDict[image].get() == 1:
                fid = Fidu(str(image), self.roi)

                # Recherche des fiduciels dans les CT
                # If a detection is not possible (no roi contoured), one keeps in mind the exam name -> res
                res, self.roi_list = fid.look_for_fidu()

                if res:
                    kept_in_mind.append(res)
                # -----------------------------------------------------
                # Recherche des fiduciels dans l'IRM à partir du CT 4D
                # -----------------------------------------------------
                # if the irm was already registered, it's not done an other time
                if '%' in image and compteur == 0:
                    compteur += 1
                    self.main_exam = str(image)
                    fid_irm = Fidu(image, self.roi)
                    try:
                        fid_irm.look_in_irm(self.roi)
                    except:
                        print("Impossible de trouver les fidus sur l'irm. Voir logs")

        for image in kept_in_mind:
            try:
                print(f'main exam {self.main_exam}')
                print(f'kept in mind {image}')

                fid = Fidu(str(image), self.roi)

                fid.rigid_registration(floating_exam=image, reference_exam=self.main_exam, focus_roi=self.roi)
                fid.copy_roi(source=self.main_exam, target=image, roi=self.roi)
                fid.look_for_fidu()

            except:
                print('problem with kept in mind')

        # if self.checkForIRM.get() == 1 :
            # print("Recherche sur IRM")
            # do IRM fidu
            
        # todo: remettre les delete
        # todo: trouver un moyen de recaler
        self.quit()

    def __end(self):
        self.quit()

    def __createWidgets(self):

        # Objects creation
        self.title("Recherche de fiduciels")
        
        # Initialisation des frames (sous structures de la fenêtre)
 
        # Frame pour la sélection des structures
        self.Frame1 = Frame(self, width = 800, height = 200, relief=GROOVE)
        self.Frame1.grid(row=0, column=0,sticky='n')
       
        # Frame pour la sélection des images en fonction des structures cochées
        self.Frame2 = Frame(self, width = 800, height = 10,  relief=GROOVE)
        self.Frame2.grid(row=1, column=0,sticky='n')
        

        # Frame pour l'IRM
        bgIRM = "Gray"
        self.Frame3 = Frame(self, width = 800, height = 20, relief=GROOVE,bg = bgIRM)
        self.Frame3.grid(row=2, column=0,sticky='s')
        
        # Frame pour les boutons
        self.Frame4 = Frame(self, width = 800, height = 20, relief=GROOVE)
        self.Frame4.grid(row=3, column=0,sticky='s')
        
        
        # Titre de la fenêtre
        self.label = Label(self.Frame1, text="Choisir le/les CTS à analyser\n (NB: le Foie doit être contouré sur un scan "
                                      "4D) \n",
                           font=('Arial', '16'))
        self.label.grid(row=0, column=0)

        compteur = 0
        rowNumber = 0
        for image in self.examDict:
            txt = ""
            if has_contour(self.patient.case, image, self.roi) and not any(word in image.lower() for word in ['dosi', '1mm']):
                color = 'red'
                size = "12"
                txt = '-> Examen de référence : ' + str(image)
                self.checkbutton = Checkbutton(self.Frame2, text=txt, variable=self.examDict[image], onvalue=1, offvalue=0,
                                           justify="center", font=('Arial', size), fg=color)
                self.checkbutton.grid(column=0, row=rowNumber, sticky=W, padx=20)
                self.checkbutton.select()
                rowNumber += 1
                # # primary or secondary setting for registration
                # print(f'Setting {image} as primary')
                # self.patient.case.Examinations[image].SetPrimary()

            elif has_contour(self.patient.case, image, self.roi) and ("%" not in image) and (
                    "dosi" not in image.lower()) and ("1mm" not in image.lower()):
                color = 'red'
                size = "12"
                txt = '-> Foie contouré (calcul rapide) : ' + str(image)
                compteur += 1
                self.checkbutton = Checkbutton(self.Frame2, text=txt, variable=self.examDict[image], onvalue=1, offvalue=0,
                                           justify="center", font=('Arial', size), fg=color)
                self.checkbutton.grid(column=0, row=rowNumber, sticky=W, padx=20)
                # print(f'Setting {image} as secondary')
                # self.patient.case.Examinations[image].SetSecondary()
                rowNumber += 1
                
            elif compteur == 0 and (any(word in image.lower() for word in ['tard', 'port', 'arter']) or not has_contour(
                    self.patient.case, image, self.roi)):
                compteur += 1
                color = 'green'
                size = "12"
                txt = '-> Conseillé : ' + str(image)
                self.checkbutton = Checkbutton(self.Frame2, text=txt, variable=self.examDict[image], onvalue=1, offvalue=0,
                                           justify="center", font=('Arial', size), fg=color)
                self.checkbutton.grid(column=0, row=rowNumber, sticky=W, padx=20)
                rowNumber += 1
                
            elif "dosi" in image.lower() or "1mm" in image.lower():
                color = 'grey'
                size = "8"
                txt = '-> Déconseillé : ' + str(image)
                self.checkbutton = Checkbutton(self.Frame2, text=txt, variable=self.examDict[image], onvalue=1, offvalue=0,
                                           justify="center", font=('Arial', size), fg=color)
                self.checkbutton.grid(column=0, row=rowNumber, sticky=W, padx=20)
                rowNumber += 1
                
            elif not (has_contour(self.patient.case, image, self.roi)) and ("%" in image):
                color = 'grey'
                size = "8"
                txt = '-> Impossible : ' + str(image)
                self.checkbutton = Checkbutton(self.Frame2, text=txt, variable=self.examDict[image], onvalue=1, offvalue=0,
                                           justify="center", font=('Arial', size), fg=color,state=DISABLED)
                self.checkbutton.grid(column=0, row=rowNumber, sticky=W, padx=20)
                rowNumber += 1
                
            else:
                color = 'black'
                size = "9"
                txt = str(image)
                self.checkbutton = Checkbutton(self.Frame2, text=txt, variable=self.examDict[image], onvalue=1, offvalue=0,
                                           justify="center", font=('Arial', size), fg=color)
                self.checkbutton.grid(column=0, row=rowNumber, sticky=W, padx=20)
                rowNumber += 1

        # Recherche IRM
        label_IRM = Label(self.Frame3, text="Réaliser la recherche sur l'IRM : ",
                   font=('Arial', '10'),justify="center",bg = bgIRM)
        label_IRM.grid(row = 0, column=0, sticky='nws',padx = 10)
        
        checkbutton_Foie = Checkbutton(self.Frame3, text = ' ', variable=self.checkForIRM ,
                   font=('Arial', '10'),justify="center",bg = bgIRM)                               
        checkbutton_Foie.grid(row = 0, column=1,sticky='nes')
            
        # Boutons
        self.runButton = Button(self.Frame4, text='Recherche des fiduciels', command=self.__execute)
        self.runButton.grid(column=0, row=0, padx=20 , pady=5)

        self.cancelButton = Button(self.Frame4, text='Annuler', command=self.__end)
        self.cancelButton.grid(column=1, row=0, padx=20 , pady=5)


if __name__ == '__main__':
    # ----- Patient -----
    # Creating patient object
    patient = Patient()

    # tkinter
    app = TkFOR(patient, 'Foie')
    app.mainloop()
