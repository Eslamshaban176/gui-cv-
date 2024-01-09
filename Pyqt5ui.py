from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.uic import loadUi
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


os.path.dirname(os.path.abspath(__file__))

class LoadQt(QMainWindow):
    def __init__(self):
        super(LoadQt, self).__init__()
        loadUi('demo.ui', self)
        self.setWindowIcon(QtGui.QIcon("icons/python-icon.png"))
        self.setWindowTitle("Computer Vision | Eslam Shaban")
        self.setFixedSize(1659, 725)
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.image = None
        self.actionOpen.triggered.connect(self.open_img)
        self.actionSave.triggered.connect(self.save_img)
        self.actionPrint.triggered.connect(self.createPrintDialog)
        self.actionQuit.triggered.connect(self.QuestionMessage)
        self.actionBig.triggered.connect(self.big_Img)
        self.actionSmall.triggered.connect(self.small_Img)
        self.actionAuthor.triggered.connect(self.AboutDeveloper)


        # Buttons [ Image Segmentation, Erosion, Dilation, Opening, Closing, Adaptive Threshold, Contours, Corner Detection, Line Detection, Global Threshold, K-means ]
        self.erosionbtn.clicked.connect(self.erosion_fun)
        self.dilationbtn.clicked.connect(self.dilation_fun)
        self.adaptivethresholdbtn.clicked.connect(self.adaptive_threshold)
        self.linedetectionbtn.clicked.connect(self.contours)
        self.openingbtn.clicked.connect(self.opening)
        self.closingbtn.clicked.connect(self.closing)
        self.globalthresholdbtn.clicked.connect(self.global_threshold)
        self.kmeansbtn.clicked.connect(self.k_means)
        self.segmentationbtn.clicked.connect(self.segmentation)

        # Buttons [ Rotation, Shearing, Translation ]
        self.rotationbtn.clicked.connect(self.rotation)
        self.shearingbtn.clicked.connect(self.shearing)
        self.translationbtn.clicked.connect(self.translation)

        # Buttons [ Grayscale, Negative, Histogram Equalization, Log, Gamma ]
        self.greyscalebtn.clicked.connect(self.grey_scale)
        self.negativebtn.clicked.connect(self.negative)
        self.hostgranequalizationbtn.clicked.connect(self.histogram_Equalization)
        self.logbtn.clicked.connect(self.Log)
        self.gammabtn.clicked.connect(self.gamma)

        # Image Restoration 1
        self.actionAdaptive_Wiener_Filtering.triggered.connect(self.weiner_filter)
        self.actionMedian_Filtering.triggered.connect(self.median_filtering)
        self.actionAdaptive_Median_Filtering.triggered.connect(self.adaptive_median_filtering)

        #  Edge Detection
        self.edgdetectionbtn.clicked.connect(self.edge_detection)

        # Image Restoration 2
        self.actionInverse_Filter.triggered.connect(self.inv_filter)


        # Smoothing
        self.actionBlur.triggered.connect(self.blur)
        self.actionBox_Filter.triggered.connect(self.box_filter)
        self.actionMedian_Filter.triggered.connect(self.median_filter)
        self.actionGaussian_Filter.triggered.connect(self.gaussian_filter)

        # Filter
        self.actionMedian_threshold_2.triggered.connect(self.median_threshold)
        self.actionDirectional_Filtering_2.triggered.connect(self.directional_filtering)
        self.actionDirectional_Filtering_3.triggered.connect(self.directional_filtering2)
        self.actionDirectional_Filtering_4.triggered.connect(self.directional_filtering3)

        # Set input
        self.dial.valueChanged.connect(self.rotation2)
        self.horizontalSlider.valueChanged.connect(self.Gamma_)
        self.gaussian_QSlider.valueChanged.connect(self.gaussian_filter2)
        self.erosion.valueChanged.connect(self.erode)
        self.Qlog.valueChanged.connect(self.Log)
        self.size_Img.valueChanged.connect(self.SIZE)
        self.canny.stateChanged.connect(self.Canny)
        self.canny_min.valueChanged.connect(self.Canny)
        self.canny_max.valueChanged.connect(self.Canny)
        self.pushButton.clicked.connect(self.reset)

    @pyqtSlot()
    def loadImage(self, fname):
        self.image = cv2.imread(fname)
        self.tmp = self.image
        self.displayImage()

    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if(self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)

        # converts an RGB image to a BGR image.
        img = img.rgbSwapped() 
        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)# căn chỉnh vị trí xuất hiện của hình trên lable
        if window == 2:
            self.imgLabel2.setPixmap(QPixmap.fromImage(img))
            self.imgLabel2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def open_img(self):
        try:
            fname, _ = QFileDialog.getOpenFileName(self, 'Open File', self.path, "Image Files (*)")
        except:
            raise ValueError("Image is not loaded")
        
        self.loadImage(fname)
            

     
    def save_img(self):
        
        fname, _ = QFileDialog.getSaveFileName(self, 'Save File', self.path + "/outputs" , "Image Files (*.png)")
        if fname:
            cv2.imwrite(fname, self.image)
            print("Image Saved")
        print("Error: Image not Saved")
       
            

    def createPrintDialog(self):
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)

        if dialog.exec_() == QPrintDialog.Accepted:
            self.imgLabel2.print_(printer)

    def big_Img(self):
        self.image = cv2.resize(self.image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def small_Img(self):
        self.image = cv2.resize(self.image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def SIZE(self , c):
        self.image = self.tmp
        self.lblText.setText(self.label_3.text())
        self.image = cv2.resize(self.image, None, fx=c, fy=c, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def reset(self):
        self.image = self.tmp
        self.displayImage(2)


    def AboutDeveloper(self):
        QMessageBox.about(self, "About Developer", "Name :   Eslam Shaban \n" 
                                                "Email : eslam@gmail.com\n"
                                                 "University : FCAI BeniSuef"
                                                 )

    def QuestionMessage(self):
        message = QMessageBox.question(self, "Exit", "Are you sure you want to exit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if message == QMessageBox.Yes:
            print("Yes")
            self.close()
        else:
            print("No")

################################ Tab 3 ##############################################################################
    def rotation(self):
        rows, cols, _ = self.image.shape
        self.lblText.setText(self.rotationbtn.text())
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1) 
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.displayImage(2)

    def rotation2(self, angle): # Dial
        self.image = self.tmp
        self.lblText.setText(self.rotationbtn.text())
        rows, cols, steps = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.displayImage(2)

    def shearing(self):
        self.image = self.tmp
        self.lblText.setText(self.shearingbtn.text())
        rows, cols, ch = self.image.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

        M = cv2.getAffineTransform(pts1, pts2)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))

        self.displayImage(2)

    def translation(self):
        self.image = self.tmp
        self.lblText.setText(self.translationbtn.text())
        num_rows, num_cols = self.image.shape[:2]

        translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])
        img_translation = cv2.warpAffine(self.image, translation_matrix, (num_cols, num_rows))
        self.image = img_translation
        self.displayImage(2)

    def erode(self , iter):

        try : 
            self.image = self.tmp
        except :
            raise ValueError("Image is not loaded")
        
        if iter > 0 :
            kernel = np.ones((4, 7), np.uint8)
            self.image = cv2.erode(self.tmp, kernel, iterations=iter)
            self.lblText.setText(self.erosionbtn.text())
        else :
            kernel = np.ones((2, 6), np.uint8)
            self.image = cv2.dilate(self.image, kernel, iterations=iter*-1)
            self.lblText.setText(self.dilationbtn.text())
        self.displayImage(2)

    def Canny(self):
        self.image = self.tmp
        self.lblText.setText(self.actionCanny.text())
        if self.canny.isChecked():
            can = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.Canny(can, self.canny_min.value(), self.canny_max.value())
        self.displayImage(2)

################################ Greyscale ##############################################################################
    def grey_scale(self):
        self.image = self.tmp
        self.lblText.setText(self.greyscalebtn.text())
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.displayImage(2)

    def anh_Xam2(self):
        self.image = self.tmp
        self.lblText.setText(self.actionGrayscale.text())
        if self.gray.isChecked():
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        self.displayImage(2)
################################ Negative ##############################################################################
    def negative(self):
        self.image = self.tmp
        self.lblText.setText(self.actionNegative.text())
        self.image = ~self.image
        self.displayImage(2)
################################ Histogram Equalization ################################################################
    def histogram_Equalization(self):
        self.image = self.tmp
        self.lblText.setText(self.hostgranequalizationbtn.text())
        img_yuv = cv2.cvtColor(self.image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        self.image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        self.displayImage(2)
################################ Log ##################################################################################
    def Log(self):
        self.image = self.tmp
        self.lblText.setText(self.actionLog.text())
        img_2 = np.uint8(np.log(self.image))
        c = 2
        self.image = cv2.threshold(img_2, c, 225, cv2.THRESH_BINARY)[1]
        self.displayImage(2)
################################ Gamma ##################################################################################
    def Gamma_(self, gamma):
        self.image = self.tmp
        self.lblText.setText(self.actionGamma.text())
        gamma = gamma*0.1
        invGamma = 1.0 /gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        self.image = cv2.LUT(self.image, table)
        self.displayImage(2)
################################ Gamma ##################################################################################
    def gamma(self):
        self.image = self.tmp
        self.lblText.setText(self.actionGamma.text())
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        self.image = cv2.LUT(self.image, table)
        self.displayImage(2)

#################################### Image Restoration 1#################################################################
    def median_filtering(self):
        self.image = self.tmp
        self.lblText.setText(self.actionMedian_Filtering.text())
        self.image = cv2.medianBlur(self.image, 5)
        self.displayImage(2)

    def adaptive_median_filtering(self):
        self.image = self.tmp
        self.lblText.setText(self.actionAdaptive_Median_Filtering.text())
        temp = []
        filter_size = 5
        indexer = filter_size // 2
        for i in range(len(self.image)):

            for j in range(len(self.image[0])):

                for z in range(filter_size):
                    if i + z - indexer < 0 or i + z - indexer > len(self.image) - 1:
                        for c in range(filter_size):
                            temp.append(0)
                    else:
                        if j + z - indexer < 0 or j + indexer > len(self.image[0]) - 1:
                            temp.append(0)
                        else:
                            for k in range(filter_size):
                                temp.append(self.image[i + z - indexer][j + k - indexer])

                temp.sort()
                self.image[i][j] = temp[len(temp) // 2]
                temp = []
        self.displayImage(2)

    def weiner_filter(self):
        self.image = self.tmp
        self.lblText.setText(self.actionAdaptive_Wiener_Filtering.text())
        M = 256  # length of Wiener filter
        Om0 = 0.1 * np.pi  # frequency of original signal
        N0 = 0.1  # PSD of additive white noise

        # generate original signal
        s = np.cos(Om0 * np.ndarray(self.image))
        # generate observed signal
        g = 1 / 20 * np.asarray([1, 2, 3, 4, 5, 4, 3, 2, 1])
        n = np.random.normal(size=self.image, scale=np.sqrt(N0))
        x = np.convolve(s, g, mode='same') + n
        # estimate (cross) PSDs using Welch technique
        f, Pxx = sig.csd(x, x, nperseg=M)
        f, Psx = sig.csd(s, x, nperseg=M)
        # compute Wiener filter
        H = Psx / Pxx
        H = H * np.exp(-1j * 2 * np.pi / len(H) * np.arange(len(H)) * (len(H) // 2))  # shift for causal filter
        h = np.fft.irfft(H)
        # apply Wiener filter to observation
        self.image = np.convolve(x, h, mode='same')
        self.displayImage(2)

####################################Image Restoration 2#################################################################
    def inv_filter(self):
        self.image = self.tmp
        self.lblText.setText(self.actionInverse_Filter.text())
        for i in range(0, 3):
            g = self.image[:, :, i]
            G = (np.fft.fft2(g))

            # h = cv2.imread(self.image, 0)
            h_padded = np.zeros(g.shape)
            h_padded[:self.image.shape[0], :self.image.shape[1]] = np.copy(self.image)
            H = (np.fft.fft2(h_padded))

            # normalize to [0,1]
            H_norm = H / abs(H.max())
            G_norm = G / abs(G.max())
            F_temp = G_norm / H_norm
            F_norm = F_temp / abs(F_temp.max())

            # rescale to original scale
            F_hat = F_norm * abs(G.max())

            # 3. apply Inverse Filter and compute IFFT
            self.image = np.fft.ifft2(F_hat)
            self.image[:, :, i] = abs(self.image)
        self.displayImage(2)

################################## Edge Detection #################################################################
    def edge_detection(self):
        self.image = self.tmp
        self.lblText.setText(self.edgdetectionbtn.text())
        # Convert the img to grayscale
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Structure the element or filter or kernel or mask
        kernel = np.ones((3,3),np.uint8)
        # Apply the mask to the image
        dilation = cv2.dilate(self.image,kernel,iterations = 1)
        # subtract the dilated image from the original image
        self.image = cv2.subtract(dilation,self.image)
        # Display the image
        self.displayImage(2)

##################################### Smoothing ##########################################################################
    def blur(self):
        self.image = self.tmp
        self.lblText.setText(self.actionBlur.text())
        self.image = cv2.blur(self.image, (5, 5))
        self.displayImage(2)

    def box_filter(self):
        self.image = self.tmp
        self.lblText.setText(self.actionBox_Filter.text())
        self.image = cv2.boxFilter(self.image, -1,(20,20))
        self.displayImage(2)

    def median_filter(self):
        self.image = self.tmp
        self.lblText.setText(self.actionMedian_Filter.text())
        self.image = cv2.medianBlur(self.image,5)
        self.displayImage(2)

    def gaussian_filter(self):
        self.image = self.tmp
        self.lblText.setText(self.actionGaussian_Filter.text())
        self.image = cv2.GaussianBlur(self.image,(5,5),0)
        self.displayImage(2)

    def gaussian_filter2(self, g):
        self.image = self.tmp
        self.lblText.setText(self.actionGaussian_Filter.text())
        self.image = cv2.GaussianBlur(self.image, (5, 5), g)
        self.displayImage(2)
########################################Filter##########################################################################
    def median_threshold(self):
        self.image = self.tmp
        self.lblText.setText(self.actionMedian_threshold_2.text())
        grayscaled = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.medianBlur(self.image,5)
        _ , threshold = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.image = threshold
        self.displayImage(2)

    def directional_filtering(self):
        self.image = self.tmp
        self.lblText.setText(self.actionDirectional_Filtering_2.text())
        kernel = np.ones((3, 3), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)

    def directional_filtering2(self):
        self.image = self.tmp
        self.lblText.setText(self.actionDirectional_Filtering_3.text())
        kernel = np.ones((5, 5), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)

    def directional_filtering3(self):
        self.image = self.tmp
        self.lblText.setText(self.actionDirectional_Filtering_4.text())
        kernel = np.ones((7, 7), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)

######################################## Image Segmentation ####################################################
    def segmentation(self):
        self.image = self.tmp
        self.lblText.setText(self.segmentationbtn.text())
        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        _ , tresh = cv2.threshold(gray,np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        # GET CONTOURS
        contours , _ = cv2.findContours(tresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(contours, key=cv2.contourArea)[-1]
        mask = np.zeros( (750, 1038), dtype="uint8" )
        maskedRed = cv2.drawContours(mask,[cnt] , -1 , (0 , 0 , 255), -1)
        maskedFinal = cv2.drawContours(mask,[cnt] , -1 , (255 , 255 , 255), -1)
        self.image = cv2.bitwise_and(self.image, self.image, mask=maskedFinal)
        self.displayImage(2)
########################################  Erosion #######################################################
    def erosion_fun(self):
        self.lblText.setText(self.erosionbtn.text())
        self.image = self.tmp
        kernel = np.ones((4, 7), np.uint8)
        self.image = cv2.erode(self.tmp, kernel, iterations=1)
        self.displayImage(2)
########################################  Dilation #######################################################
    def dilation_fun(self):
        self.image = self.tmp
        self.lblText.setText(self.dilationbtn.text())
        kernel = np.ones((4, 7), np.uint8)
        self.image = cv2.dilate(self.tmp, kernel, iterations=1)
        self.displayImage(2)
########################################  Opening #######################################################
    def opening(self):
        self.image = self.tmp
        self.lblText.setText(self.openingbtn.text())
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _ , binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        self.image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
        self.displayImage(2)
########################################  Closing #######################################################
    def closing(self):
        self.image = self.tmp
        self.lblText.setText(self.closingbtn.text())
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _ , binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        self.image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=1)
        self.displayImage(2)
########################################  Adaptive Threshold ############################################
    def adaptive_threshold(self):
        self.image = self.tmp
        value = self.adaptivethresholdbtn.text()
        self.lblText.setText(value)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        self.displayImage(2)
########################################  Contours or Line detection ############################################
    def contours(self):
        self.image = self.tmp
        value = self.linedetectionbtn.text()
        self.lblText.setText(value)
        self.image =  cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #Now convert the grayscale image to binary image
        _ , self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #Now detect the contours
        self.image, _ = cv2.findContours(self.image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # draw contours on the original image
        image_copy = self.tmp.copy()
        self.image = cv2.drawContours(image_copy, self.image, -1, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        self.displayImage(2)
########################################  Global Threshold ############################################
    def global_threshold(self):
        self.image = self.tmp
        value = self.globalthresholdbtn.text()
        self.lblText.setText(value)
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, self.image = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        self.displayImage(2)
########################################  K-means ############################################
    def k_means(self):
        self.image = self.tmp
        value = self.kmeansbtn.text()
        self.lblText.setText(value)
        # Reshape the image to a 2D array of pixels
        pixels = self.image.reshape(-1, 3).astype(np.float32)
        # Convert the pixels to the required format for K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        # K value 
        k = 3
        # Perform K-means clustering
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)
        # Convert the centers to 8-bit integers
        centers = np.uint8(centers)
        # Replace each pixel value with its corresponding center value
        segmented_image = centers[labels.flatten()]
        # Reshape the segmented image back to the original shape
        segmented_image = segmented_image.reshape(self.image.shape)
        self.image = segmented_image
        self.displayImage(2)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LoadQt()
    win.show()
    sys.exit(app.exec())

