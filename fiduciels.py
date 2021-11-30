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

from easygui import *
from pathlib import Path
from PIL import Image, ImageDraw


# todo : use the center of mass and not only the maximum
# ----------------------------------------------------
# Functions
# ----------------------------------------------------
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


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


def create_PIL_image_from_npy(img, scale=4):
    # image normalization between 0 an 1
    img[img <= -400] = -400
    img[img >= 2000] = 2000
    img = (img + abs(img.min()))
    img = img / img.max()
    img = img * 255
    img = np.array(img, dtype=np.uint8)

    # create PIL object
    im = Image.fromarray(img)
    # im.convert('L')

    # resize
    width, height = im.size
    newsize = (scale * width, scale * height)
    im = im.resize(newsize)

    return im


def add_background(img, factor, strip_size, value=0):
    background_size = factor * 2 + strip_size
    background = np.ones([background_size, background_size]) * value
    size_img = img.shape
    indent1 = int(background_size / 2 - (size_img[0] / 2))
    indent2 = int(background_size / 2 - (size_img[1] / 2))

    from_to_1 = range(indent1, indent1 + size_img[0] + 1, 1)
    print(f'from_to_1  : {from_to_1}')
    from_to_2 = range(indent2, indent2 + size_img[1] + 1, 1)
    print(f'from_to_2  : {from_to_2}')

    background[from_to_1[0]:from_to_1[-1]:1, from_to_2[0]:from_to_2[-1]:1] = img[:, :]
    return background


def create_npy_thumbnails(npy_arr, coords, factor=100, stripe_size=30):
    f = factor
    i, j, k = coords

    si, sj, sk = npy_arr.shape

    img_ax = npy_arr[i,
             max(0, j - f): min(sj, j + f),
             max(0, k - f): min(sk, k + f)]
    print(f'img_ax  : {img_ax.shape}')
    img_ax = add_background(img_ax, f, stripe_size, 0)

    # todo: check if necessary to go reverse for i
    img_sag = npy_arr[min(i + f, si):max(0, i - f): - 1,
              min(j + f, sj): max(j - f, 0):-1,
              k]
    print(f'img_sag  : {img_sag.shape}')
    img_sag = add_background(img_sag, f, stripe_size, 0)

    return img_ax, img_sag


def warning_msg():
    msg = "Attention: ce script réalise une sauvegarde automatique du dossier.\n" \
          "\n- tous les recalages seront supprimés" \
          "\n- les frames of reference seront dissociés\n" \
          "\nCliquer sur Continue pour sauvegarder le dossier et continuer."
    title = "Script fiduciels"
    if ccbox(msg, title):  # show a Continue/Cancel dialog
        pass  # user chose Continue
    else:  # user chose Cancel
        sys.exit(0)


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
        self.dixon_name = []
        self.get_dixon_name()
        self.patient_id = self.patient.PatientID
        self.roi = None

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

    def get_dixon_name(self):
        irm_list = self.get_irm_list()
        for irm in irm_list:
            irm_name = irm[0]
            irm_modality = irm[1]
            if "DIXON" in irm_modality and "RESPI LIBRE GADO TARDIF" in irm_modality:
                self.dixon_name.append(irm_name)
                print(irm_name)
        return self.dixon_name

    def get_roi_list(self):
        rois = self.case.PatientModel.RegionsOfInterest
        self.roi_list = [roi.Name for roi in rois]
        return self.roi_list

    def get_description(self, imageName):
        description = self.case.Examinations[imageName].GetStoredDicomTagValueForVerification(Group=0x0008,
                                                                                              Element=0x103E)
        description = description.__getitem__("SeriesDescription")
        return description

    def sort_acquisitions(self, roi, ref, secondary, forbidden):
        """This method takes all the acquisition names and sorts them
            - to do = images of interest containing a roi contour
            - to do after = images of interest without contour
            - not do to = no interest"""
        self.roi = roi

        # First creates a list containing all the acquisitions
        images = self.get_ct_list()

        # lists initialization
        to_do_now = []
        to_do_after = []
        not_to_do = []

        # Au début, je fais une premiere boucle for pour trier les CT. Pour que le CT 50% soit le premier de la
        # liste, tant que ce n'est pas lui, je fais passer les autres CT en fin de liste. La nouvelle recherche se
        # fera avec lui en premier

        index = 0
        for index, image in enumerate(images):
            if not has_contour(self.case, image, self.roi):
                images.append(image)
            else:
                break

        # STARTING THE REAL LOOP
        new_images = images[index:]
        for image in new_images:
            print(f'image sorting ... {image}')
            if has_contour(self.case, image, self.roi) \
                    and any(word in image.lower() for word in ref):
                # and not any(word in image.lower() for word in forbidden)
                # to_do_now.append((image, self.get_description(image)))
                to_do_now.append(image)

            # elif has_contour(self.case, image, self.roi) and ("%" not in image) and (
            #         "dosi" not in image.lower()) and ("1mm" not in image.lower()):
            #     # to_do_now.append((image, self.get_description(image)))
            #     to_do_now.append(image)

            elif any(word in image.lower() for word in secondary):
                # to_do_after.append((image, self.get_description(image)))
                to_do_after.append(image)

            else:
                # not_to_do.append((image, self.get_description(image)))
                not_to_do.append(image)

        # at the end, if there are mri's, we take them
        if self.dixon_name:
            for dixon in self.dixon_name:
                to_do_after.append(dixon)
        return to_do_now, to_do_after, not_to_do


class Images(Patient):
    def __init__(self, exam_name, roi_name=None):
        super().__init__()
        self.exam_name = exam_name
        self.data = self.case.Examinations[exam_name].Series[0].ImageStack.PixelData
        self.n_data = self.data.size
        self.n_voxels = int(self.n_data / 2)

        # Images Shape
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

        # Images Corners
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

        # Images Creation
        # BE CAREFUL : the first axis is inf/sup in numpy format
        self.image_itk = None
        self.image_npy = None
        self.image_to_process = None

        # Roi
        self.roi_name = roi_name
        self.roi_mask_npy = None

        # # IRM
        # # name of the dixon IRM
        # self.dixon_name = None
        # # self.get_dixon_name()

        # Change frame of reference
        # self.change_frame_of_reference()

    def change_frame_of_reference(self):
        # removing existing registration
        n_reg = self.case.Registrations.Count
        if n_reg:
            for reg in range(n_reg):
                floating = self.case.Registrations[reg].FromFrameOfReference
                reference = self.case.Registrations[reg].ToFrameOfReference
                self.case.RemoveRegistration(FloatingFrameOfReference=floating,
                                             ReferenceFrameOfReference=reference)
        self.patient.Save()
        self.case.Examinations[self.exam_name].AssignToNewFrameOfReference()

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
                raise Exception("Please, give a roi name to the Images Object if you want to use this method")
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

    def create_IRM_external(self):

        if not check_roi(self.case, 'External'):
            self.case.PatientModel.CreateRoi(Name=r"External", Color="Green", Type="External", TissueName=r"",
                                             RbeCellTypeName=None, RoiMaterial=None)

        self.case.PatientModel.RegionsOfInterest['External'].CreateExternalGeometry(
            Examination=self.case.Examinations[self.exam_name], ThresholdLevel=15)


class Fidu(Images):
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
            # Images Creation
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

            # this attribute is used by post processing (it needs to have all the information for hist creation)
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
            coordinates, thrown_away = self.post_processing(coordinates)

            # Converting coordinates to positions (in mm)
            positions = self.get_position_from_coordinates(coordinates)

            # sorting coordinates compared to the image origin
            d = [sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in positions]
            positions = [x for _, x in sorted(zip(d, positions))]
            print(positions)
            # creation of POIs in RS
            self.poi_creation(positions)

        else:
            keep_in_mind = str(self.exam_name)
            print(f"Please, make a contour for the roi {self.roi_name}")

        return keep_in_mind, roi_list

    def post_processing(self, coordinates):

        print("\nPost processing (looking for artefacts)...")
        s = self.fidu_size
        coord_new = []
        thrown_away = []
        for index, coord in enumerate(coordinates):
            i, j, k = coord
            matrix = self.image_to_process[i - s:i + s,
                     j - s:j + s,
                     k - s:k + s]
            hist, bin_edges = np.histogram(matrix, bins=[-1500, -200, self.threshold_value, self.maximum])
            print(hist)

            if hist[0] > 0:  # and (hist[2] < 100):
                print("-> La tache numéro : " + str(index + 1) + " contient des artefacts!\n")
                auto_result = True
            else:
                auto_result = False
                continue

                # Adding some manual decision
            if auto_result:
                msg = "L'algorithme pense que c'est un fiduciel. Est ce réellement le cas?" \
                      "\n\n-> Attention: il peut d'agir d'un artefact" \
                      "\n-> Si le fiduciel est dédoublé, cliquer sur 'Dédoublé'"
            else:
                msg = "' ------------------------------------- /!\'" \
                      "L'algorithme pense que ce n'est PAS un fiduciel. Est ce réellement le cas?" \
                      "\n-> Si le fiduciel est dédoublé, cliquer sur 'Dédoublé'"

            # creating images from numpy arrays
            factor = 80  # number of voxel surrounding the central voxel (how width is the thumbnail image)
            stripe_size = 20

            img_ax, img_sag = create_npy_thumbnails(self.image_npy, coords=coord, factor=factor,
                                                    stripe_size=stripe_size)

            method1 = False
            if method1:
                # img = np.concatenate((img_ax, blank, im_sag), axis=1)  #with central rectangle
                img = np.concatenate((img_ax, img_sag), axis=1)
                # converting npy array to image (0->255 , scaling it, etc.)
                scale = 3  # enlarging factor (for more comfortable viewing)
                im_to_save = create_PIL_image_from_npy(img, scale=scale)

            else:
                scale = 3
                d = 40  # half size of the rectangle

                im1 = create_PIL_image_from_npy(img_ax, scale=scale)
                im2 = create_PIL_image_from_npy(img_sag, scale=scale)
                x, y = im1.size
                x0 = x / 2 - d
                x1 = x / 2 + d

                # im1.convert('RGB')
                # im2.convert('RGB')

                draw = ImageDraw.Draw(im1)
                draw2 = ImageDraw.Draw(im2)

                draw.ellipse([(x0, x0), (x1, x1)], outline='red', width=5)
                draw2.ellipse([(x0, x0), (x1, x1)], outline='red', width=5)

                im_to_save = get_concat_h(im1, im2)

            directory = r"\\Client\X$\DEV\Fiduciels\images"
            filename = str(self.patient_id + '_' + str(i) + str(j) + str(k) + '_' + str(index) + '.png')
            filename.replace("\\", "_")  # in case the name is xxx\xxx
            filename.replace("/", "_")
            filename.replace(" ", "")

            im_path = os.path.join(directory, filename)

            # if the file already exists, delete it
            path = Path(im_path)
            if path.is_file():
                path.unlink()

            # saving image
            im_to_save.save(im_path)

            # easyGUI message
            choices = ["C'est un fidu", "Ce n'est PAS un fidu", "Dédoublé"]
            reply = buttonbox(msg, image=im_path, choices=choices)

            if reply == "C'est un fidu":
                coord_new.append(coord)
            elif reply == "Ce n'est PAS un fidu":
                thrown_away.append(coord)
            elif reply == "Dédoublé":
                thrown_away.append(coord)
            # # Final decision
            # if auto_result:
            #     coord_new.append(coord)
            # else:
            #     thrown_away.append(coord)
        return coord_new, thrown_away

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

    def look_in_irm(self, main_exam):

        # Roi inside which one is looking for the fidus (usually -> 'Foie')
        input_roi = self.roi_name

        # The Name of DIXON IRM is self.dixon_name
        # 1- ones needs to register IRM and CT and to copy little boxes centered to the fidu and copy them to the IRM

        # self.get_dixon_name()

        # External Creation on dixon acquisition

        self.create_IRM_external()

        # Setting CT as primary
        self.case.Examinations[main_exam].SetPrimary()
        self.case.Examinations[self.exam_name].SetSecondary()

        # Rigid registration between IRM and CT
        self.rigid_registration(self.exam_name, main_exam, self.roi_name)

        # Creating little spheres
        coord, roi_list = self.create_sphere_roi(main_exam)

        # Copying Rois one by one
        for roi_name in roi_list:
            print(roi_name)
            print(self.exam_name)
            self.copy_roi(source=main_exam, target=self.exam_name, roi=roi_name)

        # Copy of the liver roi
        self.copy_roi(source=main_exam, target=self.exam_name, roi=self.roi_name)

        # -------------------------------------------------------
        # FIDU CREATION
        # -------------------------------------------------------
        # Creating a Fidu object with dixon image as main exam attribute
        obj_irm = Fidu(self.exam_name)
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

    def create_sphere_roi(self, main_exam):

        """ poi is a part of case.PatientModel.PointsOfInterest"""
        coordinates, roi_list = [], []
        for poi in self.case.PatientModel.PointsOfInterest:
            # Work only with the fidus named 'Fidu 1" etc.
            if self.fidu_prefix_names in poi.Name:
                x = self.case.PatientModel.StructureSets[main_exam].PoiGeometries[poi.Name].Point.x
                y = self.case.PatientModel.StructureSets[main_exam].PoiGeometries[poi.Name].Point.y
                z = self.case.PatientModel.StructureSets[main_exam].PoiGeometries[poi.Name].Point.z
                coordinates.append([x, y, z])
                roi_name = poi.Name + '_roi'
                roi_list.append(roi_name)

                try:
                    self.case.PatientModel.CreateRoi(Name=roi_name, Color="Yellow", Type="Marker", TissueName=None,
                                                     RbeCellTypeName=None, RoiMaterial=None)
                except:
                    print(f'{roi_name} already exists!')

                self.case.PatientModel.RegionsOfInterest[roi_name].CreateSphereGeometry(
                    Radius=self.radius, Examination=self.case.Examinations[main_exam],
                    Center={'x': x, 'y': y, 'z': z}, Representation="TriangleMesh", VoxelSize=None)
        return coordinates, roi_list

    def create_roi(self, roi_name, color='Black', Type="Undefined"):

        if not check_roi(self.case, roi_name):
            try:
                self.case.PatientModel.CreateRoi(Name=roi_name, Color=color, Type=Type, TissueName=None,
                                                 RbeCellTypeName=None, RoiMaterial=None)
            except:
                print('Unable to create this structure! ')

    def create_wall(self, OutwardDistance=1, InwardDistance=0):

        # create roi if needed
        out_roi = self.roi_name + '_wall'
        self.create_roi(out_roi, color='Pink')

        # create wall
        res = self.case.PatientModel.RegionsOfInterest[out_roi].SetWallExpression(SourceRoiName=self.roi_name,
                                                                                  OutwardDistance=OutwardDistance,
                                                                                  InwardDistance=InwardDistance)
        res.UpdateDerivedGeometry(Examination=self.examination, Algorithm="Auto")

    def create_margin(self, distance=1):

        # create roi if needed
        out_roi = self.roi_name + ' + ' + str(distance)
        self.create_roi(out_roi, color='Grey')

        res = self.case.PatientModel.RegionsOfInterest[out_roi].SetMarginExpression(SourceRoiName=self.roi_name,
                                                                                    MarginSettings={'Type': "Expand",
                                                                                                    'Superior': distance,
                                                                                                    'Inferior': distance,
                                                                                                    'Anterior': distance,
                                                                                                    'Posterior': distance,
                                                                                                    'Right': distance,
                                                                                                    'Left': distance})

        res.UpdateDerivedGeometry(Examination=self.examination, Algorithm="Auto")


if __name__ == '__main__':

    warning_msg()

    # roi to analyse
    roi = 'Foie'

    # ----- Patient -----
    # Creating patient object
    patient = Patient()

    # ----- Creating multiple choice box  -----
    # Multiple Choice
    ref_acquisition = ("resp", "bloque", "mid", "ventilation")
    secondary_injected_acquisition = ('tard', 'port', 'arter', 'inj')
    forbidden_words = ('dosi', '1mm')

    to_do_now, to_do_after, _ = patient.sort_acquisitions(roi, ref_acquisition, secondary_injected_acquisition,
                                                          forbidden_words)

    # Creating easyguiBox
    msgbox_title = "Recherche des fiduciels"
    msgbox_txt = "Sur quelles acquisitions souhaitez-vous lancer la recherche de fiduciels?" \
                 "\n\n-> Le foie doit avoir été contouré sur l'acquisition 4D 50% (à défaut la 4D 30%)" \
                 "\n-> Possibilité de choisir plusieurs acquisitions dont l'IRM Dixon"
    images_to_process = multchoicebox(msgbox_txt, msgbox_title, to_do_now + to_do_after)

    # ----- Fidu creation  -----
    # creating some lists containing all the informations needed
    fid_objects = [Fidu(str(im), roi) for im in images_to_process]
    modalities = [fid_objects[index].case.Examinations[im].EquipmentInfo.Modality for index, im in enumerate \
        (images_to_process)]  # incredible onelining. Don't ask
    contoured = [has_contour(fid_objects[index].case, str(im), roi) for index, im in enumerate \
        (images_to_process)]
    frames_of_reference = [fid_objects[index].case.Examinations[im].EquipmentInfo.FrameOfReference for index, im in enumerate \
        (images_to_process)]

    values, counts = np.unique(frames_of_reference, return_counts=True)
    print(values,counts)

    already_done = []
    main_exam = to_do_now[0]
    was_main_exam_done = fid_objects[0].case.PatientModel.StructureSets[main_exam].PoiGeometries.Count
    for index, im in enumerate(images_to_process):
        nb = fid_objects[index].case.PatientModel.StructureSets[im].PoiGeometries.Count
        if nb > 0:
            already_done.append(True)
        else:
            already_done.append(False)

    for index, im in enumerate(images_to_process):
        modality = modalities[index]
        contour = contoured[index]
        fid_object = fid_objects[index]

        if modality == 'CT':
            # If the roi of interest already exists, just look for fidus
            if contour:
                main_exam = im
                fid_object.look_for_fidu()
            # If the roi does not exist, one have to copy it from an other registered exam
            else:
                # First doing the registration
                fid_object.rigid_registration(floating_exam=im, reference_exam=main_exam, focus_roi=roi)
                # Then copying the roi
                fid_object.copy_roi(source=main_exam, target=im, roi=roi)
                # Finally, looking for fidus
                fid_object.look_for_fidu()

        elif modality == 'MR':
            # in order to achieve the search of fidus on MR image, one needs to have already found them on the main
            # CT scan. If not (else condition), one starts searching on the main exam
            if was_main_exam_done:
                fid_object.look_in_irm(main_exam)

            else:
                fid = Fidu(str(main_exam), roi)
                fid.look_for_fidu()  # do first the main exam
                was_main_exam_done = True
                fid_object.look_in_irm(main_exam)
