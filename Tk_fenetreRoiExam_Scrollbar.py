import os
import tkinter as tk
from tkinter import *
from tkinter import ttk

# import SimpleITK as sitk
# import numpy as np
# from connect import *
# from skimage.draw import polygon2mask
# from skimage.feature import peak_local_max


# # todo : ne pas sélectionner les irm et les scanners millimétriques
# # ----------------------------------------------------
# # Functions
# # ----------------------------------------------------

# def check_roi(case, roi_to_check):
    # # this method checks if a toi exists
    # roi_check = False
    # rois = case.PatientModel.RegionsOfInterest
    # for roi in rois:
        # if roi.Name == roi_to_check:
            # roi_check = True
    # return roi_check


# def has_contour(case, examination, roi_to_check):
    # """ Check if a structure is empty or not"""
    # return case.PatientModel.StructureSets[examination].RoiGeometries[roi_to_check].HasContours()


# def get_bounding_box(case, examination, roi):
    # # get coordinates of two point that create a box around the roi_name
    # bound = case.PatientModel.StructureSets[examination].RoiGeometries[roi].GetBoundingBox()
    # bounds = [[bound[0]['x'], bound[0]['y'], bound[0]['z']], [bound[1]['x'], bound[1]['y'], bound[1]['z']]]
    # return bounds


# # ----------------------------------------------------
# # Classes
# # ----------------------------------------------------

# class Patient:
    # def __init__(self):
        # self.case = get_current("Case")
        # self.examination = get_current("Examination")
        # self.patient = get_current("Patient")
        # self.examination_names = []
        # self.roi_list = []

    # def get_examination_list(self):
        # for i in self.case.Examinations:
            # self.examination_names.append(i.Name)
        # return self.examination_names

    # def get_roi_list(self):
        # rois = self.case.PatientModel.RegionsOfInterest
        # self.roi_list = [roi.Name for roi in rois]
        # return self.roi_list


# class Image(Patient):
    # def __init__(self, exam_name, roi_name=None):
        # super().__init__()
        # self.exam_name = exam_name
        # self.data = self.case.Examinations[exam_name].Series[0].ImageStack.PixelData
        # self.n_data = self.data.size
        # self.n_voxels = int(self.n_data / 2)

        # # Image Shape
        # self.rows = self.case.Examinations[self.exam_name].Series[0].ImageStack.NrPixels.x
        # self.columns = self.case.Examinations[self.exam_name].Series[0].ImageStack.NrPixels.y
        # self.depth = int(self.n_data / 2 / self.rows / self.columns)

        # # Depth
        # self.slice_position = self.case.Examinations[self.exam_name].Series[0].ImageStack.SlicePositions
        # self.slice_thickness = abs(self.slice_position[1] - self.slice_position[0])

        # # Direction
        # column_direction_x = self.case.Examinations[self.exam_name].Series[0].ImageStack.ColumnDirection.x
        # column_direction_y = self.case.Examinations[self.exam_name].Series[0].ImageStack.ColumnDirection.y
        # column_direction_z = self.case.Examinations[self.exam_name].Series[0].ImageStack.ColumnDirection.z

        # row_direction_x = self.case.Examinations[self.exam_name].Series[0].ImageStack.RowDirection.x
        # row_direction_y = self.case.Examinations[self.exam_name].Series[0].ImageStack.RowDirection.y
        # row_direction_z = self.case.Examinations[self.exam_name].Series[0].ImageStack.RowDirection.z

        # slice_direction_x = self.case.Examinations[self.exam_name].Series[0].ImageStack.SliceDirection.x
        # slice_direction_y = self.case.Examinations[self.exam_name].Series[0].ImageStack.SliceDirection.y
        # slice_direction_z = self.case.Examinations[self.exam_name].Series[0].ImageStack.SliceDirection.z

        # self.direction = (column_direction_x, column_direction_y, column_direction_z,
                          # row_direction_x, row_direction_y, row_direction_z,
                          # slice_direction_x, slice_direction_y, slice_direction_z)

        # # Image Corners
        # self.x_corner = self.case.Examinations[self.exam_name].Series[0].ImageStack.Corner.x
        # self.y_corner = self.case.Examinations[self.exam_name].Series[0].ImageStack.Corner.y
        # self.z_corner = self.case.Examinations[self.exam_name].Series[0].ImageStack.Corner.z
        # self.origin = (self.x_corner, self.y_corner, self.z_corner)

        # # Pixel size
        # self.x_pixel_size = self.case.Examinations[self.exam_name].Series[0].ImageStack.PixelSize.x
        # self.y_pixel_size = self.case.Examinations[self.exam_name].Series[0].ImageStack.PixelSize.y
        # self.spacing = (self.x_pixel_size, self.y_pixel_size, self.slice_thickness)

        # # Conversion Parameters
        # self.intercept = self.case.Examinations[self.exam_name].Series[
            # 0].ImageStack.ConversionParameters.RescaleIntercept
        # self.slope = self.case.Examinations[self.exam_name].Series[0].ImageStack.ConversionParameters.RescaleSlope

        # # Image Creation
        # # BE CAREFUL : the first axis is inf/sup in numpy format
        # self.image_itk = None
        # self.image_npy = None
        # self.image_to_process = None

        # # Roi
        # self.roi_name = roi_name
        # self.roi_mask_npy = None

    # def get_itk_image(self, to_save=False):
        # """Return a simpleITK image object"""
        # print("----> Starting CT conversion ...")
        # result = np.zeros(self.n_voxels)
        # b1 = self.data[0:-1:2]
        # b2 = self.data[1::2]
        # result[b2 > 128] = b1[b2 > 128] + 256 * (b2[b2 > 128] - 128) - 65536
        # result[b2 <= 128] = b1[b2 <= 128] + 256 * b2[b2 <= 128]
        # result = self.slope * result + self.intercept
        # result = np.reshape(result, [self.depth, self.columns, self.rows])

        # # numpy file saving
        # if to_save:
            # # If needed, this saves the numpy array to file for external use
            # path = os.getcwd()
            # with open(os.path.join(path, "results.npy"), 'wb') as f:
                # np.save(f, result)

        # # Simple ITK conversion
        # itk_image = sitk.GetImageFromArray(result)
        # itk_image.SetDirection(self.direction)
        # itk_image.SetOrigin(self.origin)
        # itk_image.SetSpacing(self.spacing)

        # print(itk_image.GetSize())
        # print(itk_image.GetOrigin())
        # print(itk_image.GetSpacing())
        # print(itk_image.GetDirection())
        # return itk_image

    # def get_mask_from_roi(self):
        # # This method needs a roi_name
        # if self.roi_name is None:
            # raise Exception("Please, give a roi name to the Image Object if you want to use this method")

        # # This method needs the creation of the itk image if not already done
        # if self.image_itk is None:
            # self.image_itk = self.get_itk_image(to_save=False)
            # self.image_npy = sitk.GetArrayFromImage(self.image_itk)
        # try:
            # check_roi(self.case, self.roi_name)
            # has_contour(self.case, self.exam_name, self.roi_name)
        # except:
            # raise Exception("This roi does not exist")

        # # Converting the Roi from voxel type to contours type
        # self.case.PatientModel.StructureSets[self.exam_name].RoiGeometries[self.roi_name].SetRepresentation(
            # Representation='Contours')

        # # Starting creation
        # print(f"----> Starting mask creation for {self.roi_name} on {self.exam_name} ...")
        # # Creation of an empty array that has the same size as the original image
        # mask = np.zeros_like(self.image_npy)

        # # sl is a list of all the dots for one single slice
        # for sl in self.case.PatientModel.StructureSets[self.exam_name].RoiGeometries[
            # self.roi_name].PrimaryShape.Contours:
            # # for each slice, one creates an array that will contain all the dots coordinates.
            # # This array is initialized by using np.ones like this -> coordinates = [[1,1,1] , [1,1,1] ,...]
            # # it will be filled with coordinates like this -> [[x1,y1,z1],[x2,y2,z2], ....]

            # n_dots = len(sl)
            # coordinates = np.ones([n_dots, 3])

            # slice_number = 0
            # # dot contains three coordinates (in mm) for one dot of the contour. The three coordinates are needed
            # # for the conversion from positions in mm to index
            # for index, dot in enumerate(sl):
                # coordinates[index] = self.image_itk.TransformPhysicalPointToIndex([dot.x, dot.y, dot.z])
                # if index == 0:
                    # slice_number = coordinates[index][2]
            # # polygon2mask creates a mask only for 2D images. So ones needs to suppress the third coordinate (number
            # # of the slice)
            # image_shape = [self.columns, self.rows]
            # coordinates_xy = [c[0:2] for c in coordinates]
            # temp_mask = polygon2mask(image_shape, coordinates_xy)

            # mask[int(slice_number), :, :] = temp_mask

        # print('mask creation ok!')
        # mask.astype(int)

        # self.roi_mask_npy = mask
        # return mask


# class Fidu(Image):
    # def __init__(self, exam_name, roi_name=None, maximum=None, threshold_abs=1600, threshold_relative=0.7,
                 # threshold_type='absolute'):
        # super().__init__(exam_name, roi_name)

        # # Fidu parameter: distance in voxel between two spots
        # self.fidu_size = 10

        # # threshold
        # self.threshold_type = threshold_type
        # self.threshold_abs = threshold_abs
        # self.threshold_relative = threshold_relative
        # self.threshold_value = None
        # self.maximum = maximum

        # # Automatic Fidu seeking
        # self.look_for_fidu()

    # def find_local_max(self, image_to_process):
        # s = self.fidu_size
        # footprint = np.ones([s, s, s])
        # coordinates = peak_local_max(image_to_process, min_distance=s, footprint=footprint)
        # return coordinates

    # def look_for_fidu(self):
        # if check_roi(self.case, self.roi_name) and has_contour(self.case, self.exam_name, self.roi_name):
            # # Image Creation
            # # BE CAREFUL : the first axis is inf/sup in numpy format
            # self.image_itk = self.get_itk_image(to_save=False)
            # self.image_npy = sitk.GetArrayFromImage(self.image_itk)

            # # creating a copy of the numpy array containing the image
            # image_to_process = np.copy(self.image_npy)

            # # Applying mask using structure:
            # self.get_mask_from_roi()
            # image_to_process = image_to_process * self.roi_mask_npy
            # # this attribute is used by post processing
            # self.image_to_process = np.copy(image_to_process)

            # # thresholding
            # self.maximum = np.amax(self.image_npy)

            # if self.threshold_type == 'absolute':
                # self.threshold_value = self.threshold_abs

            # elif self.threshold_type == 'relative':
                # self.threshold_value = self.threshold_relative * self.maximum

            # print(f'max = {self.maximum}')
            # print(f'threshold = {self.threshold_abs}')
            # image_to_process[self.image_npy < self.threshold_value] = 0

            # # Seeking the fiducials
            # coordinates = self.find_local_max(image_to_process)

            # # post processing
            # coordinates = self.post_processing(coordinates)

            # # Converting coordinates to positions
            # positions = self.get_position_from_coordinates(coordinates)

            # # sorting coordinates by z then by x
            # positions = sorted(positions, key=lambda x: (x[2], x[0]))

            # # creation of POIs in RS
            # self.poi_creation(positions)
        # else:
            # print(f"Please, make a contour for the roi {self.roi_name}")

    # def post_processing(self, coordinates):
        # print("\nPost Processing...")
        # s = self.fidu_size

        # coord_new = []
        # for index, coord in enumerate(coordinates):
            # i = coord[0]
            # j = coord[1]
            # k = coord[2]
            # matrix = self.image_to_process[i - s:i + s,
                     # j - s:j + s,
                     # k - s:k + s]
            # hist, bin_edges = np.histogram(matrix, bins=[-1500, -200, self.threshold_value, self.maximum])
            # print(hist)
            # if (hist[0] > 0) and (hist[2] < 100):
                # print("-> La tache numéro : " + str(index + 1) + " contient des artefacts!\n")
                # coord_new.append(coord)
        # return coord_new

    # def get_position_from_coordinates(self, coordinates):

        # # position will contain coordinates of all the fidus (in mm)
        # positions = []

        # for coord in coordinates:
            # # conversion of coordinates (x,y,z) into float values
            # coord = tuple(map(float, coord))
            # # replacing z by x
            # coord = [coord[1], coord[2], coord[0]]
            # # transforming coord in position
            # position = self.image_itk.TransformContinuousIndexToPhysicalPoint(coord)
            # positions.append(position)

        # return positions

    # def poi_creation(self, coordinates):
        # for i, coords in enumerate(coordinates):
            # name = "Fidu " + str(i + 1)
            # try:
                # # POI list creation in RS
                # self.case.PatientModel.CreatePoi(Name=name, Color="Yellow", VisualizationDiameter=1,
                                                 # Type="Undefined")
            # except:
                # print('The POI already exists.')
            # # POI assignation
            # x, z, y = coords
            # self.case.PatientModel.StructureSets[self.exam_name].PoiGeometries[name].Point = {
                # 'x': x, 'y': z, 'z': y}

class Scrollable(ttk.Frame):
    """
       Make a frame scrollable with scrollbar on the right.
       After adding or removing widgets to the scrollable frame, 
       call the update() method to refresh the scrollable area.
    """

    def __init__(self, frame, widthsize):

        vscrollbar = tk.Scrollbar(frame, width=20, orient="vertical")
        # vscrollbar.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        vscrollbar.grid(row=0, column=1, sticky='ns')

        # hscrollbar = tk.Scrollbar(frame, width=20,orient ="horizontal")
        # # hscrollbar.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
        # hscrollbar.grid(row=1, column=0, sticky='ew')
        
        # self.canvas = tk.Canvas(frame,width=widthsize,xscrollcommand=hscrollbar.set, yscrollcommand=vscrollbar.set)
        self.canvas = tk.Canvas(frame,width=widthsize, yscrollcommand=vscrollbar.set)
        # self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.grid(row=0, column=0) 
        
        vscrollbar.config(command=self.canvas.yview)
        # hscrollbar.config(command=self.canvas.xview)

        self.canvas.bind('<Configure>', self.__fill_canvas)

        # base class initialization
        tk.Frame.__init__(self, frame)         

        # assign this obj (the inner frame) to the windows item of the canvas
        self.windows_item = self.canvas.create_window(0,0, window=self, anchor=tk.NW)


    def __fill_canvas(self, event):
        "Enlarge the windows item to the canvas width"

        canvas_width = event.width
        self.canvas.itemconfig(self.windows_item, width = canvas_width)        

    def update(self):
        "Update the canvas and the scrollregion"

        self.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox(self.windows_item))


        
        
class TkFOR(tk.Tk):
    # def __init__(self, patient, roi):
    def __init__(self, roiList):
        print("Recherche pour toutes les structures")
        tk.Tk.__init__(self)
        
        self.rowNumber = 0
        self.rowNumberMax = 4
        self.fr2_rowNumber = 0
        
        
        ## Creer une instance Patient
        # self.patient = patient
        
        ## Récupère la liste des examens
        # self.examinationList = self.patient.get_examination_list()
        self.examinationList = ["4D  40%","4D  60%","2021.04.28  50% Contourage Foie","2021.04.28 PORTAL / MNI","dosimétrie"]
        
        ## Passer en argument pour pouvoir choisir les structures dans le main
        self.roiList = roi
        
        ## Dictionnaire pour les chekbutton des Roi
        self.roiDict = {}
        
        ## Liste pour stocker les dictionnaires des examens en fonction des Roi
        self.roiExamList = []
        
        ## Liste pour stocker les Roi cochées
        self.activatedRoiList = []
        
        ## Liste pour stocker les dictionnaires des examens en fonction des Roi
        self.checkRegistrationForLiver = IntVar()
        
        ## Creation des différents dictionnaires pour le stockage des variables de checkbutton
        self.__createDict()

        ## Creation des widgets
        self.__createMainWidgets()
        
    def __createDict(self):
        for roi in self.roiList :
            self.roiDict[roi] = IntVar()
            examDict = {}
            for exam in self.examinationList:
                examDict[exam] = IntVar()
            self.roiExamList.append(examDict)
 
    def __delete_frame(self, frame):
        for widget in frame.winfo_children():
            widget.grid_forget() 

    
    def __returnActivatedRoi(self) :
        roiList = [] 
        for r, (roi, v) in enumerate(self.roiDict.items()): 
            if self.roiDict[roi].get() == 1:
                roiList.append((r,roi))       
        return roiList 
        
    def __create_examChecked(self,roiNumber,widget):
    
    ## Initialisation des variables pour la mise en page
        self.fr2_rowNumber +=1
       
    ## Case à cocher pour la sélection des ROIs        
        for i, (image, image_value) in enumerate(self.roiExamList[roiNumber].items()): 

            color = 'black'
            size = "9"
            
            # if str(self.activatedRoiList[roiNumber][1] HAS CONTOUR :             
            self.checkbutton2 = Checkbutton(widget, text=image, variable=self.roiExamList[self.activatedRoiList[roiNumber][0]][image], onvalue=1, offvalue=0,
                                       justify="center", font=('Arial', size), fg=color)
            self.checkbutton2.grid(column=0, row=self.fr2_rowNumber, sticky=W, padx=20)
            self.fr2_rowNumber += 1
            # else : 
                # " Moi je n'afficherai meme pas la série"
            
    def __callBack_roiChecked(self):
    
        self.activatedRoiList = self.__returnActivatedRoi()       
        
        ### Detruit tout le frame 2 lorsqu'aucune case n'est coché
        if len(self.activatedRoiList) == 0 : 
            self.__delete_frame(self.Frame2)
            self.Frame2.grid_forget()
 
        ### Crée le Frame2 et le rempli en fonction de ROI activé
        else :
           
            ## Efface et recrée un frame 2 vide 
            self.__delete_frame(self.Frame2)
            self.Frame2.grid_forget()
            
            
            ## Creation de la place pour le Frame 2
            self.Frame2 = Frame(self, width = 800, height = 100,  relief=GROOVE)
            self.scrollable_frame2 = Scrollable(self.Frame2, 500)
            
            ## Boucle uniquement sur les structures activées
            fr2_columnNumber = 0
            fr2_rowNumberMax = 3
            count = 0
            
            self.fr2_rowNumber = 0
            
            for r in range (0,len(self.activatedRoiList)) : 
                    
                label2 = Label(self.scrollable_frame2, text="Sélectionner les images pour la structure "+ str(self.activatedRoiList[r][1]),
                           font=('Arial', '10'))
                label2.grid(row = self.fr2_rowNumber, column=0, sticky='w')
                
                self.__create_examChecked(r,self.scrollable_frame2)

                if (self.activatedRoiList[r][1] == "Foie") :
                    
                    label_Foie = Label(self.scrollable_frame2, text="Réaliser les recalages entre les différentes séries d'images sélectionnées : ",
                               font=('Arial', '10'),justify="center")
                    label_Foie.grid(row = self.fr2_rowNumber, column=0, sticky='nws',padx = 10)
                    
                    checkbutton_Foie = Checkbutton(self.scrollable_frame2, text = ' ', variable=self.checkRegistrationForLiver ,
                               font=('Arial', '10'),justify="center")                               
                    checkbutton_Foie.grid(row = self.fr2_rowNumber, column=0,sticky='nes')
                    self.fr2_rowNumber +=1

            self.Frame2.grid(row=1, column=0,sticky='n')
            self.scrollable_frame2.update()
            
    def __createMainWidgets(self):

        #### Titre de la fenêtre Tkinter
        self.title("Recherche de fiduciels")
        
        #### Initialisation des frames (sous structures de la fenêtre)
 
        ## Frame pour la sélection des structures
        self.Frame1 = Frame(self, width = 800, height = 200, relief=GROOVE)
        self.Frame1.grid(row=0, column=0,sticky='n')
        
        ## Frame pour la sélection des images en fonction des structures cochées
        self.Frame2 = Frame(self, width = 800, height = 10,  relief=GROOVE)
        self.Frame2.grid(row=1, column=0,sticky='n')
        

        ## Frame pour les boutons
        self.Frame3 = Frame(self, width = 800, height = 20, relief=GROOVE)
        self.Frame3.grid(row=2, column=0,sticky='s')
        
        
        #### Creation des widgets 
        
        ## Titre
        self.label = Label(self.Frame1, text="Sélectionner les ROIs puis les images pour la recherche de fiduciels",
                           font=('Arial', '16'))
        self.label.grid(row=self.rowNumber, column=0, columnspan=(len(self.roiList) // self.rowNumberMax + 1),ipady =10, ipadx = 10)

        ## Case à cocher pour la sélection des ROIs   
        ## Changer pour fixer à 3 colonnes ou 4 plutot qu'un nombre de ligne
        for i, (roi, v) in enumerate(self.roiDict.items()):  
            self.rowNumber = 1 + (i)% self.rowNumberMax
            self.columnNumber = (i) // self.rowNumberMax
            self.checkbutton  = Checkbutton(self.Frame1, text=roi, variable=self.roiDict[roi], onvalue=1, offvalue=0,
                                       justify="center", font=('Arial', "9"), fg='black', command = self.__callBack_roiChecked)
            self.checkbutton.grid(column=self.columnNumber, row=self.rowNumber, sticky=W, padx=20)
        
        ## Bouton d'exécution
        self.runButton = Button(self.Frame3, text='Recherche des fiduciels', command=self.__execute)
        self.runButton.grid(column=0, row=0, padx=20 , pady=5)

        ## Bouton d'annulation
        self.cancelButton = Button(self.Frame3, text='Annuler', command=self.__end)
        self.cancelButton.grid(column=1, row=0, padx=20 , pady=5)

    def __execute(self):
        for i, (roi, v) in enumerate(self.roiDict.items()):  
            if self.roiDict[roi].get() == 1 :
                # print(roi)
                for image in self.roiExamList[i] :
                    if self.roiExamList[i][image].get() == 1 :
                        # Fidu(str(image), self.roi)
                        # Fidu(str(image), roi)
                        print(roi," - ",str(image)) 
                        
                if roi == 'Foie' and self.checkRegistrationForLiver.get() == 1 :
                    print("Les recalages vont être effectués")
        self.quit()

    def __end(self):
        self.quit()

if __name__ == '__main__':
    # ----- Patient -----
    # Creating patient object
    # patient = Patient()

    # Creating a list containing all the examination names
    # examinations = patient.get_examination_list()
    # roi = patient.get_roi_list()
    
    examinations = ["4D  40%","4D  60%","2021.04.28  50% Contourage Foie","2021.04.28 PORTAL / MNI","dosimétrie"]
    # examinations = ["4D  40%","4D  60%","2021.04.28  50% Contourage Foie","2021.04.28 PORTAL / MNI","dosimétrie"]
    
    roi = ["Foie","Duodenum","Colon","Estomac","Coeur"]
    # roi = patient.get_roi_list()

    # tkinter
    # app = TkFOR(patient, ['Foie'])
    app = TkFOR(roi)
    app.mainloop()

    # Creating Fidu object
    # fidu = Fidu('4D  60%', 'Foie')
