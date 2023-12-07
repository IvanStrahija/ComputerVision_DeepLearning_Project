import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QMdiArea, QMdiSubWindow, QMessageBox, QPlainTextEdit, QSpinBox, QSpacerItem, QSizePolicy, QMdiSubWindow, QLineEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QByteArray
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import sys
import time
import os
from io import StringIO
import subprocess
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torchsummary
import torchvision.models as models
import torch.optim as optim

fs1 = cv2.FileStorage('Data\\alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
fs2 = cv2.FileStorage('Data\\alphabet_lib_vertical.txt',cv2.FILE_STORAGE_READ)


class Iask1(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_paths = []
        self.current_image_index = 0
        self.timer = QTimer(self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('1')
        self.setGeometry(100, 100, 435, 900)
        self.setFixedSize(435, 900)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        
        self.text_label = QLabel("Task 1", self.central_widget)
        self.text_label.setStyleSheet("background-color: lightgreen;font-weight: bold;")
        self.text_label.setAlignment(Qt.AlignHCenter)
        self.text_label.setFixedHeight(18)
        self.layout.addWidget(self.text_label)

        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setText("Display Area")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.stop_button = QPushButton('Clear Display', self)
        self.stop_button.clicked.connect(self.stop_display)
        self.stop_button.setStyleSheet("color: red; font-weight: bold;")
        self.layout.addWidget(self.stop_button)

        self.load_button = QPushButton('Load Images', self)
        self.load_button.clicked.connect(self.load_images)
        self.load_button.setStyleSheet("color: green; font-weight: bold;")
        self.layout.addWidget(self.load_button)

        self.findcorners = QPushButton('1.1 Find Corners', self)
        self.findcorners.clicked.connect(self.find_corners)
        self.layout.addWidget(self.findcorners)

        self.intrinsic = QPushButton('1.2 Find intrinsic matrix', self)
        self.intrinsic.clicked.connect(self.find_intrinsic)
        self.layout.addWidget(self.intrinsic)

        spacer = QWidget()  
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        spacer.setFixedHeight(30)
        self.layout.addWidget(spacer)  

        self.counter = QSpinBox(self)
        self.counter.setRange(1, 100)  
        self.counter.setSingleStep(1)  
        self.counter.setPrefix("Image number: ")
        self.layout.addWidget(self.counter)
    
        self.extrinsic = QPushButton('1.3 Find extrinsic matrix', self)
        self.extrinsic.clicked.connect(self.find_extrinsic)
        self.layout.addWidget(self.extrinsic)

        spacer = QWidget()  
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        spacer.setFixedHeight(30)
        self.layout.addWidget(spacer)
        

        self.dist = QPushButton('1.4 Find distortion', self)
        self.dist.clicked.connect(self.find_dist)
        self.layout.addWidget(self.dist)

        self.undist = QPushButton('1.5 Show undistorted', self)
        self.undist.clicked.connect(self.find_undist)
        self.layout.addWidget(self.undist)

        self.output_text = QPlainTextEdit(self)
        self.output_text.setFixedSize(413, 100)
        self.layout.addWidget(self.output_text)
        self.output_text.setPlainText('Distortion, extrinsic and intrinsic matrixes will be displayed here')

        self.timer.timeout.connect(self.find_corners)

    def stop_display(self):
        self.timer.stop()
        self.image_label.clear()
        self.image_label.setText("Display Area")
        self.image_label.setAlignment(Qt.AlignCenter)

    def show_message(self, message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle('Notification')
        msg_box.setText(message)
        msg_box.exec_()

    def load_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.image_paths, _ = QFileDialog.getOpenFileNames(self, 'Open Image Files', '', 'Image Files (*.jpg *.png *.bmp *.gif *.jpeg *.ico);;All Files (*)', options=options)

        if self.image_paths:
            self.current_image_index = 0
            self.show_message('Images loaded successfully.')
        else:
            self.show_message('Error while loading images.')

    def find_corners(self):
        if self.image_paths:
            if self.current_image_index < len(self.image_paths):
                image_path = self.image_paths[self.current_image_index]
                image = cv2.imread(image_path)

                # Parameters and initialization
                pattern_size = (11,8)
                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                new_width = 768
                new_height = 768
                image_size =(new_width, new_height)
                obj_points = []  
                img_points = []
                objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
                objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

                # Image processing
                image = cv2.resize(image, image_size)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3,3), 3)
                sharp = cv2.addWeighted(blur, 0.5, blur, -0.4,0)
                rett, thresholded = cv2.threshold(sharp, 69, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ret, corners = cv2.findChessboardCornersSB(thresholded, pattern_size, None)
                if ret:
                    corners2 = cv2.cornerSubPix(thresholded, corners, winSize, zeroZone, criteria)
                    obj_points.append(objp)
                    img_points.append(corners2)
                    cv2.drawChessboardCorners(image, pattern_size, corners2, ret)
                
                #output
                self.timer.start(1000)
                # Display the processed image
                self.display_processed_image(image)
                self.current_image_index += 1
                
                if self.current_image_index >= len(self.image_paths):
                    self.current_image_index = 0
            else:
                self.timer.stop()

    def find_intrinsic (self):
        def format_element(element):
            return "{:.5f}".format(element)
        if self.image_paths:
            pattern_size = (11,8)
            winSize = (5, 5)
            zeroZone = (-1, -1)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            new_width = 768
            new_height = 768
            self.image_size =(new_width, new_height)
            obj_points = []  
            img_points = []
            objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

            for image_path in self.image_paths:
                image = cv2.imread(image_path)
                
                # Image processing
                image = cv2.resize(image, (new_width, new_height))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3,3), 3)
                sharp = cv2.addWeighted(blur, 0.5, blur, -0.4,0)
                rett, thresholded = cv2.threshold(sharp, 69, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ret, corners = cv2.findChessboardCornersSB(thresholded, pattern_size, None)
                if ret:
                    corners2 = cv2.cornerSubPix(thresholded, corners, winSize, zeroZone, criteria)
                    obj_points.append(objp)
                    img_points.append(corners2)

            # find intrinsic
            ret2, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera (obj_points, img_points, self.image_size ,None, None)
            fx = self.mtx[0, 0]
            fy = self.mtx[1, 1]
            cx = self.mtx[0, 2]
            cy = self.mtx[1, 2]
            skew = self.mtx[0, 1]
            intrinsic = np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1]])
            intrinsic = np.vectorize(format_element)(intrinsic)
            intrinsic_str = np.array2string(intrinsic, precision=5, suppress_small=True)
            display_text = "Intrinsic Matrix:\n" + intrinsic_str
            self.output_text.setPlainText(display_text)
            
            print ('Inrinsic:')
            print (intrinsic)
                 
    def display_processed_image(self, processed_image):
        h, w, c = processed_image.shape  # Get height, width, and number of channels (for color images)
        bytes_per_line = c * w  # Calculate bytes per line for color image
        q_image = QImage(processed_image.data, w, h, bytes_per_line, QImage.Format_RGB888)  # Use RGB888 format for color image
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def find_extrinsic (self):
        def format_element(element):
            return "{:.5f}".format(element)
        if self.image_paths:
            if self.counter.value() <= len(self.image_paths):
                pattern_size = (11,8)
                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                new_width = 768
                new_height = 768
                image_size =(new_width, new_height)
                obj_points = []  
                img_points = []
                objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
                objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
                
                image_path = self.image_paths[self.counter.value()-1]
                image = cv2.imread(image_path)
                image = cv2.resize(image, (new_width, new_height))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3,3), 3)
                sharp = cv2.addWeighted(blur, 0.5, blur, -0.4,0)
                rett, thresholded = cv2.threshold(sharp, 69, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ret, corners = cv2.findChessboardCornersSB(thresholded, pattern_size, None)
                if ret:
                    corners2 = cv2.cornerSubPix(thresholded, corners, winSize, zeroZone, criteria)
                    obj_points.append(objp)
                    img_points.append(corners2)

                    ret2, mtx, dist, rvecs, tvecs = cv2.calibrateCamera (obj_points, img_points, image_size ,None, None)
                    for rvec, tvec in zip(rvecs, tvecs):
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        extrinsic_matrix = np.column_stack((rotation_matrix, tvec))
                    extrinsic_matrix = np.vectorize(format_element)(extrinsic_matrix)
                    extrinsic_str = np.array2string(extrinsic_matrix, precision=5, suppress_small=True)
                    display_text = "Extrinsic Matrix:\n" + extrinsic_str
                    self.output_text.setPlainText(display_text)
                    print ('Extrinsic:')
                    print (extrinsic_matrix)

    def find_dist (self):
        if hasattr(self, 'mtx') and self.mtx is not None:
            def format_element(element):
                return "{:.5f}".format(element)
            dist = np.vectorize(format_element)(self.dist)
            dist_str = np.array2string(dist, precision=5, suppress_small=True)
            display_text = "Distortion:\n" + dist_str
            self.output_text.setPlainText(display_text)
            print ('Distortion:')
            print (dist)
        
    def find_undist (self):
        if hasattr(self, 'mtx') and self.mtx is not None:
            if self.image_paths:
                image_path = self.image_paths[self.counter.value()-1]
                image = cv2.imread(image_path)
                image = cv2.resize(image, self.image_size)
                newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, self.image_size, 1, self.image_size)
                undst = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)
                x, y, w, h = roi
                undst = undst[y:y+h, x:x+w]
                undst[10, 10] = (0, 0, 255)
                undst= cv2.cvtColor(undst, cv2.COLOR_BGR2RGB)
                self.timer.stop()
                self.image_label.clear()
                self.display_processed_image(undst)

class Task2(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_paths = []
        self.current_image_index = 0
        self.timer2 = QTimer(self)
        self.timer3 = QTimer(self)
        self.timer2.timeout.connect(self.stop_display)
        self.timer2.timeout.connect(self.write_text_hor)
        self.timer3.timeout.connect(self.write_text_ver)
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('2')
        self.setGeometry(100, 100, 435, 450)
        self.setFixedSize(435, 450)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        
        self.text_label = QLabel("Task 2", self.central_widget)
        self.text_label.setStyleSheet("background-color: lightgreen;font-weight: bold;")
        self.text_label.setAlignment(Qt.AlignHCenter)
        self.text_label.setFixedHeight(18)
        self.layout.addWidget(self.text_label)

        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setText("Display Area")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.stop_button = QPushButton('Clear Display', self)
        self.stop_button.clicked.connect(self.stop_display)
        self.stop_button.setStyleSheet("color: red; font-weight: bold;")
        self.layout.addWidget(self.stop_button)

        self.load_button = QPushButton('Load Images', self)
        self.load_button.clicked.connect(self.load_images)
        self.load_button.setStyleSheet("color: green; font-weight: bold;")
        self.layout.addWidget(self.load_button)

        self.input_text = QLineEdit(self)
        self.input_text.setPlaceholderText("Enter a 6 letter word: ")
        self.layout.addWidget(self.input_text)

        submit_button = QPushButton('2.1 Show word horizontaly', self)
        submit_button.clicked.connect(self.process_input)
        submit_button.clicked.connect(self.write_text_hor)
        self.layout.addWidget(submit_button)

        submit_button2 = QPushButton('2.2 Show word verticaly', self)
        submit_button2.clicked.connect(self.process_input)
        submit_button2.clicked.connect(self.write_text_ver)
        self.layout.addWidget(submit_button2)
    
    def show_message(self, message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle('Notification')
        msg_box.setText(message)
        msg_box.exec_()

    def load_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.image_paths, _ = QFileDialog.getOpenFileNames(self, 'Open Image Files', '', 'Image Files (*.jpg *.png *.bmp *.gif *.jpeg *.ico);;All Files (*)', options=options)

        if self.image_paths:
            self.current_image_index = 0
            self.show_message('Images loaded successfully.')
        else:
            self.show_message('Error while loading images.')

    def stop_display(self):
        self.timer2.stop()
        self.timer3.stop()
        self.image_label.clear()
        self.image_label.setText("Display Area")
        self.image_label.setAlignment(Qt.AlignCenter)

    def process_input(self):
        input_text = self.input_text.text()
        input_text = input_text.upper()
        if len(input_text) != 6:
            self.show_message('Enter a 6 letter word.')
        self.chars = []
        for char in input_text:
                self.chars.append(char)

    def display_processed_image(self, processed_image):
            h, w, c = processed_image.shape  # Get height, width, and number of channels (for color images)
            bytes_per_line = c * w  # Calculate bytes per line for color image
            q_image = QImage(processed_image.data, w, h, bytes_per_line, QImage.Format_RGB888)  # Use RGB888 format for color image
            pixmap = QPixmap.fromImage(q_image)

            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

    def write_text_hor(self):
        if len(self.chars)==6:
            self.timer2.stop()
            self.timer3.stop()
            def draw(image, imgpts):
                    letter= int (len (imgpts))
                    for i in range (0,letter,2):
                        image = cv2.line(image, tuple(imgpts[i].ravel()), tuple(imgpts[i+1].ravel()), (255, 87, 51), 15)
                    return image
            ch1 = fs1.getNode(self.chars[0]).mat()
            ch2 = fs1.getNode(self.chars[1]).mat()
            ch3 = fs1.getNode(self.chars[2]).mat()
            ch4 = fs1.getNode(self.chars[3]).mat()
            ch5 = fs1.getNode(self.chars[4]).mat()
            ch6 = fs1.getNode(self.chars[5]).mat()
            input_chars = [ch1, ch2, ch3, ch4, ch5, ch6]

            if self.image_paths:
                pattern_size = (11,8)
                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                new_width = 768
                new_height = 768
                self.image_size =(new_width, new_height)
                obj_points = []  
                img_points = []
                objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
                objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
                no = 0
                if self.current_image_index < len(self.image_paths):
                    image_path = self.image_paths[self.current_image_index]
                    image = cv2.imread(image_path)
                    
                    # Image processing
                    image = cv2.resize(image, (new_width, new_height))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (3,3), 3)
                    sharp = cv2.addWeighted(blur, 0.5, blur, -0.4,0)
                    rett, thresholded = cv2.threshold(sharp, 69, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    ret, corners = cv2.findChessboardCornersSB(thresholded, pattern_size, None)
                    if ret:
                        corners2 = cv2.cornerSubPix(thresholded, corners, winSize, zeroZone, criteria)
                        obj_points.append(objp)
                        img_points.append(corners2)

                        ret2, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera (obj_points, img_points, self.image_size ,None, None)

                        count = 0
                        for j in range (2):
                            for i in range (3):
                                ch = input_chars [count]
                                translation_vector = np.array([3+i*3, 2+j*3, 0], dtype=np.float32)
                                translation_matrix = np.eye(4)
                                translation_matrix[0:3, 3] = translation_vector
                                angle_in_degrees = 180
                                rotation_vector = np.array([0, 0, np.deg2rad(angle_in_degrees)], dtype=np.float32)
                                rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
                                rotation_matrix_homogeneous = np.eye(4)
                                rotation_matrix_homogeneous[0:3, 0:3] = rotation_matrix
                                transformation_matrix = translation_matrix.dot(rotation_matrix_homogeneous)

                                ch = cv2.transform(ch.reshape(1, -1, 3), transformation_matrix)[0]
                                ch = ch[:, :3]
                                axis = np.float32(ch).reshape(-1, 3)
                                imgpts, jac = cv2.projectPoints(axis, rvecs[0], tvecs[0], self.mtx, self.dist)
                                imgpts = imgpts.astype(int)
                                image =  draw(image, imgpts)
                                count = count+1   
                    

                    self.timer2.start(1000)
                    # Display the processed image
                    self.display_processed_image(image)
                    self.current_image_index += 1
                    
                    if self.current_image_index >= len(self.image_paths):
                        self.current_image_index = 0
                else:
                    self.timer2.stop()
                    
    def write_text_ver(self):
            if len(self.chars)==6:
                self.timer2.stop()
                self.timer3.stop()
                def draw(image, imgpts):
                        letter= int (len (imgpts))
                        for i in range (0,letter,2):
                            image = cv2.line(image, tuple(imgpts[i].ravel()), tuple(imgpts[i+1].ravel()), (255, 87, 51), 15)
                        return image
                ch1 = fs2.getNode(self.chars[0]).mat()
                ch2 = fs2.getNode(self.chars[1]).mat()
                ch3 = fs2.getNode(self.chars[2]).mat()
                ch4 = fs2.getNode(self.chars[3]).mat()
                ch5 = fs2.getNode(self.chars[4]).mat()
                ch6 = fs2.getNode(self.chars[5]).mat()
                input_chars = [ch1, ch2, ch3, ch4, ch5, ch6]

                if self.image_paths:
                    pattern_size = (11,8)
                    winSize = (5, 5)
                    zeroZone = (-1, -1)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    new_width = 768
                    new_height = 768
                    self.image_size =(new_width, new_height)
                    obj_points = []  
                    img_points = []
                    objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
                    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
                    no = 0
                    if self.current_image_index < len(self.image_paths):
                        image_path = self.image_paths[self.current_image_index]
                        image = cv2.imread(image_path)
                        
                        # Image processing
                        image = cv2.resize(image, (new_width, new_height))
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        blur = cv2.GaussianBlur(gray, (3,3), 3)
                        sharp = cv2.addWeighted(blur, 0.5, blur, -0.4,0)
                        rett, thresholded = cv2.threshold(sharp, 69, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        ret, corners = cv2.findChessboardCornersSB(thresholded, pattern_size, None)
                        if ret:
                            corners2 = cv2.cornerSubPix(thresholded, corners, winSize, zeroZone, criteria)
                            obj_points.append(objp)
                            img_points.append(corners2)

                            ret2, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera (obj_points, img_points, self.image_size ,None, None)

                            count = 0
                            for j in range (2):
                                for i in range (3):
                                    ch = input_chars [count]
                                    translation_vector = np.array([3+i*3, 2+j*3, 0], dtype=np.float32)
                                    translation_matrix = np.eye(4)
                                    translation_matrix[0:3, 3] = translation_vector
                                    angle_in_degrees = 180
                                    rotation_vector = np.array([0, 0, np.deg2rad(angle_in_degrees)], dtype=np.float32)
                                    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
                                    rotation_matrix_homogeneous = np.eye(4)
                                    rotation_matrix_homogeneous[0:3, 0:3] = rotation_matrix
                                    transformation_matrix = translation_matrix.dot(rotation_matrix_homogeneous)

                                    ch = cv2.transform(ch.reshape(1, -1, 3), transformation_matrix)[0]
                                    ch = ch[:, :3]
                                    axis = np.float32(ch).reshape(-1, 3)
                                    imgpts, jac = cv2.projectPoints(axis, rvecs[0], tvecs[0], self.mtx, self.dist)
                                    imgpts = imgpts.astype(int)
                                    image =  draw(image, imgpts)
                                    count = count+1   
                        

                        self.timer3.start(1000)
                        # Display the processed image
                        self.display_processed_image(image)
                        self.current_image_index += 1
                        
                        if self.current_image_index >= len(self.image_paths):
                            self.current_image_index = 0
                    else:
                        self.timer3.stop()

class Task3(QMainWindow):
    def __init__(self):
        super().__init__()

        self.timer4 = QTimer(self)
        self.selected_point = None
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('3')
        self.setGeometry(100, 100, 435, 450)
        self.setFixedSize(435, 450)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        
        self.text_label = QLabel("Task 3", self.central_widget)
        self.text_label.setStyleSheet("background-color: lightgreen;font-weight: bold;")
        self.text_label.setAlignment(Qt.AlignHCenter)
        self.text_label.setFixedHeight(18)
        self.layout.addWidget(self.text_label)

        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setText("Display Area")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.load_buttonL = QPushButton('Load Image_L', self)
        self.load_buttonL.clicked.connect(self.load_imageL)
        self.load_buttonL.setStyleSheet("color: green; font-weight: bold;")
        self.layout.addWidget(self.load_buttonL)

        self.load_buttonR = QPushButton('Load Image_R', self)
        self.load_buttonR.clicked.connect(self.load_imageR)
        self.load_buttonR.setStyleSheet("color: green; font-weight: bold;")
        self.layout.addWidget(self.load_buttonR)

        self.show_disparity = QPushButton('3.1 Show disparity map (Press ESC to exit)', self)
        self.show_disparity.clicked.connect(self.disparity)
        self.layout.addWidget(self.show_disparity)

        self.check_disparity = QPushButton('3.2 Show corresponding points (Press ESC to exit)', self)
        self.check_disparity.clicked.connect(self.cor_disparity)
        self.layout.addWidget(self.check_disparity)

    def show_message(self, message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle('Notification')
        msg_box.setText(message)
        msg_box.exec_()

    def load_imageL(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.imgL, _ = QFileDialog.getOpenFileNames(self, 'Open Image Files', '', 'Image Files (*.jpg *.png *.bmp *.gif *.jpeg *.ico);;All Files (*)', options=options)

        if len(self.imgL)==1:
            self.show_message('Image_L loaded successfully.')
        else:
            self.show_message('Error while loading. Select only one image.')

    def load_imageR(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.imgR, _ = QFileDialog.getOpenFileNames(self, 'Open Image Files', '', 'Image Files (*.jpg *.png *.bmp *.gif *.jpeg *.ico);;All Files (*)', options=options)

        if len(self.imgR)==1:
            self.show_message('Image_R loaded successfully.')
        else:
            self.show_message('Error while loading. Select only one image.')

    def display_processed_image(self, processed_image):
        if len(processed_image.shape) == 2:
            h, w = processed_image.shape  # Get height and width for gray image
            bytes_per_line = w  # Calculate bytes per line for gray image
            q_image = QImage(processed_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)  # Use Grayscale8 format for gray image
        else:
            h, w, c = processed_image.shape  # Get height, width, and number of channels (for color images)
            bytes_per_line = c * w  # Calculate bytes per line for color image
            q_image = QImage(processed_image.data, w, h, bytes_per_line, QImage.Format_RGB888)  # Use RGB888 format for color image

        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def stop_display(self):
        self.timer4.stop()
        self.image_label.clear()
        self.image_label.setText("Display Area")
        self.image_label.setAlignment(Qt.AlignCenter)

    def disparity (self):
        if hasattr(self, 'imgL') and self.imgL and hasattr(self, 'imgR') and self.imgR is not None:
            stereo = cv2.StereoBM_create(numDisparities=256, blockSize=9)
            image_pathL = self.imgL[0]
            imgL = cv2.imread(image_pathL)
            image_pathR = self.imgR[0]
            imgR = cv2.imread(image_pathR)
            self.imgLd = cv2.cvtColor (imgL, cv2.COLOR_BGR2GRAY)
            self.imgRd = cv2.cvtColor (imgR, cv2.COLOR_BGR2GRAY)
            self.disparity_map = stereo.compute(self.imgLd,self.imgRd).astype(np.float32)
            self.disparity_map = cv2.normalize(self.disparity_map, None, 0, 255, norm_type=cv2.NORM_MINMAX)
            self.display_processed_image(self.disparity_map)
            cv2.imshow ('disp', self.disparity_map)
            while True:
                key = cv2.waitKey(1)
                if key == 27:  # Check for the 'Esc' key (ASCII code 27)
                    break

            cv2.destroyAllWindows()

        

    def mouseCallback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouseX = x
            self.mouseY = y
            self.selected_point = (x, y) 
    
    def cor_disparity (self):
        if hasattr(self, 'disparity_map') and self.disparity_map is not None:
            
            image_pathL = self.imgL[0]
            imgL = cv2.imread(image_pathL)
            image_pathR = self.imgR[0]
            imgR = cv2.imread(image_pathR)
            

            disparity_map = self.disparity_map
            while True:
                cv2.namedWindow('Left Image')
                cv2.setMouseCallback('Left Image', self.mouseCallback)
                cv2.imshow('Left Image', imgL)
            
                if self.selected_point is not None:
                    print (self.selected_point)
                    disparity = disparity_map[self.selected_point[1], self.selected_point[0]]
                    if disparity > 0:
                        print ('Selected coordinates:' + str (self.selected_point) + '\n Disparity:' + str(disparity)+ '\n')
                        right_x = self.selected_point[0] - disparity
                        right_y = self.selected_point[1]
                        cv2.circle(imgR, (int(right_x), right_y), 5, 255, -1)  # Highlight the corresponding point in the right image
                    else:
                        print ('Failure\n')
                    self.selected_point = None

                cv2.namedWindow('Right Image')
                cv2.imshow('Right Image', imgR)
                if cv2.waitKey(1) & 0xFF == 27:  # Exit when ESC key is pressed
                    break

            cv2.destroyAllWindows()

class Task4(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('4')
        self.setGeometry(0, 0, 435, 900)
        self.setFixedSize(435, 900)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        
        self.text_label = QLabel("Task 4", self.central_widget)
        self.text_label.setStyleSheet("background-color: lightgreen;font-weight: bold;")
        self.text_label.setAlignment(Qt.AlignHCenter)
        self.text_label.setFixedHeight(18)
        self.layout.addWidget(self.text_label)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(415, 415)
        self.image_label.setText("Display Area")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.layout.addWidget(self.image_label)

        self.stop_button = QPushButton('Clear Display', self)
        self.stop_button.clicked.connect(self.stop_display)
        self.stop_button.setStyleSheet("color: red; font-weight: bold;")
        self.layout.addWidget(self.stop_button)

        self.load_buttonL = QPushButton('Load Image_L', self)
        self.load_buttonL.clicked.connect(self.load_imageL)
        self.load_buttonL.setStyleSheet("color: green; font-weight: bold;")
        self.layout.addWidget(self.load_buttonL)

        self.load_buttonR = QPushButton('Load Image_R', self)
        self.load_buttonR.clicked.connect(self.load_imageR)
        self.load_buttonR.setStyleSheet("color: green; font-weight: bold;")
        self.layout.addWidget(self.load_buttonR)

        self.keypointsL = QPushButton('4.1 Show keypoints (Left)', self)
        self.keypointsL.clicked.connect(self.get_keypointsL)
        self.layout.addWidget(self.keypointsL)

        self.keypointsR = QPushButton('4.1 Show keypoints (Right)', self)
        self.keypointsR.clicked.connect(self.get_keypointsR)
        self.layout.addWidget(self.keypointsR)

        self.keypoints_matched = QPushButton('4.2 Match keypoints (Press ESC to exit)', self)
        self.keypoints_matched.clicked.connect(self.match_keypoints)
        self.layout.addWidget(self.keypoints_matched)

        self.image_label2 = QLabel(self)
        self.image_label2.setFixedSize(415, 220)
        self.layout.addWidget(self.image_label2)

    def show_message(self, message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle('Notification')
        msg_box.setText(message)
        msg_box.exec_()

    def load_imageL(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.imgL, _ = QFileDialog.getOpenFileNames(self, 'Open Image Files', '', 'Image Files (*.jpg *.png *.bmp *.gif *.jpeg *.ico);;All Files (*)', options=options)

        if len(self.imgL)==1:
            self.show_message('Image_L loaded successfully.')
        else:
            self.show_message('Error while loading. Select only one image.')

    def load_imageR(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.imgR, _ = QFileDialog.getOpenFileNames(self, 'Open Image Files', '', 'Image Files (*.jpg *.png *.bmp *.gif *.jpeg *.ico);;All Files (*)', options=options)

        if len(self.imgR)==1:
            self.show_message('Image_R loaded successfully.')
        else:
            self.show_message('Error while loading. Select only one image.')

    def display_processed_image(self, processed_image):
        if len(processed_image.shape) == 2:
            h, w = processed_image.shape  # Get height and width for gray image
            bytes_per_line = w  # Calculate bytes per line for gray image
            q_image = QImage(processed_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)  # Use Grayscale8 format for gray image
        else:
            h, w, c = processed_image.shape  # Get height, width, and number of channels (for color images)
            bytes_per_line = c * w  # Calculate bytes per line for color image
            q_image = QImage(processed_image.data, w, h, bytes_per_line, QImage.Format_RGB888)  # Use RGB888 format for color image

        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)  

    def stop_display(self):
        self.image_label.clear()
        self.image_label.setText("Display Area")
        self.image_label.setAlignment(Qt.AlignCenter)

    def get_keypointsL (self):
        if hasattr(self, 'imgL') and self.imgL is not None:
            sift = cv2.SIFT_create()
            image_pathL = self.imgL[0]
            imgL = cv2.imread(image_pathL)
            imgL = cv2.resize(imgL, (512, 512))
            self.keypointsL, self.descriptorsL = sift.detectAndCompute(imgL, None)
            Left_keypoints = cv2.drawKeypoints(imgL, self.keypointsL, imgL, color=(0, 255, 0))
            self.display_processed_image(Left_keypoints)
    
    def get_keypointsR (self):
        if hasattr(self, 'imgR') and self.imgR is not None:
            sift = cv2.SIFT_create()
            image_pathR = self.imgR[0]
            imgR = cv2.imread(image_pathR)
            imgR = cv2.resize(imgR, (512, 512))
            self.keypointsR, self.descriptorsR = sift.detectAndCompute(imgR, None)
            Right_keypoints = cv2.drawKeypoints(imgR, self.keypointsR, imgR, color=(0, 255, 0))
            self.display_processed_image(Right_keypoints)

    def match_keypoints(self):
        if hasattr(self, 'keypointsR') and self.keypointsR and hasattr(self, 'keypointsL') and self.keypointsL is not None:
            if hasattr(self, 'imgR') and self.imgR and hasattr(self, 'imgL') and self.imgL is not None:
                bf = cv2.BFMatcher()
                image_pathR = self.imgR[0]
                imgR = cv2.imread(image_pathR)
                image_pathL = self.imgL[0]
                imgL = cv2.imread(image_pathL)
                imgL = cv2.resize(imgL, (512, 512))
                imgR = cv2.resize(imgR, (512, 512))
                matches = bf.knnMatch(self.descriptorsL, self.descriptorsR, k=2)
                good = []
                for m,n in matches:
                    if m.distance < 0.5*n.distance:
                        good.append([m])
                matched_image = cv2.drawMatchesKnn(imgL, self.keypointsL, imgR, self.keypointsR, good,None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                self.display_processed_image(matched_image)
                cv2.imshow('matched', matched_image)
                key = cv2.waitKey(0)
                if key == 27:
                    cv2.destroyAllWindows()

class Task5(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = models.vgg19_bn(pretrained=True)
        self.model.classifier[6] = torch.nn.Linear(4096, 10)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.plot = cv2.imread('Data\\60_epoch.png')
        nn_path = 'Data\\epoch_60.pt'
        nn = torch.load (nn_path)
        self.model.load_state_dict(nn['model_state_dict'])
        self.optimizer.load_state_dict(nn['optimizer_state_dict'])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize the image to the model's input size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('5')
        self.setGeometry(100, 100, 435, 900)
        self.setFixedSize(435, 900)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        

        self.text_label = QLabel("Task 5", self.central_widget)
        self.text_label.setStyleSheet("background-color: lightgreen;font-weight: bold;")
        self.text_label.setAlignment(Qt.AlignHCenter)
        self.text_label.setFixedHeight(18)
        self.layout.addWidget(self.text_label)

        self.load_images5 = QPushButton('Load Image', self)
        self.load_images5.clicked.connect(self.load_images5f)
        self.load_images5.setStyleSheet("color: green; font-weight: bold;")
        self.layout.addWidget(self.load_images5)

        self.show_aug = QPushButton('5.1 Show augmented images', self)
        self.show_aug.clicked.connect(self.show_augmented)
        self.layout.addWidget(self.show_aug)

        self.show_struct = QPushButton('5.2 Show model structure in terminal', self)
        self.show_struct.clicked.connect(self.show_structure)
        self.layout.addWidget(self.show_struct)

        self.show_acc = QPushButton('5.3 Show accuracy and loss', self)
        self.show_acc.clicked.connect(self.show_acc_loss)
        self.layout.addWidget(self.show_acc)

        self.show_inf = QPushButton('5.3 Show inference', self)
        self.show_inf.clicked.connect(self.show_inference)
        self.layout.addWidget(self.show_inf)

        self.image_label = QLabel(self)
        self.image_label.setText("Display Area")
        self.image_label.setFixedSize(414, 414)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.layout.addWidget(self.image_label)

        self.text_label2 = QLabel(self)
        self.text_label2.setAlignment(Qt.AlignCenter)
        self.text_label2.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(self.text_label2)

        

    def load_images5f(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.image5, _ = QFileDialog.getOpenFileNames(self, 'Open Image Files', '', 'Image Files (*.jpg *.png *.bmp *.gif *.jpeg *.ico);;All Files (*)', options=options)

        if len(self.image5) == 1:
            pixmap = QPixmap(self.image5[0])
            self.image_label.setPixmap(pixmap.scaled(128,128))
            self.text_label2.setText("")
            self.show_message('Image loaded successfully.')           
        else:
            self.image5 = []
            self.show_message('Select only one image.')
            
    def show_message(self, message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle('Notification')
        msg_box.setText(message)
        msg_box.exec_()
    
    def show_augmented (self):
        path = 'Images\\Q5_Image\\Q5_1'
        images = []

        for filename in os.listdir(path):
            if filename.endswith(".png"):
                file_path = os.path.join(path, filename)
                img = Image.open(file_path)
                images.append((img, filename))  # Store both the image and its filename

        transforms2 = v2.Compose([
            v2.RandomRotation(60),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        ])

        for i in range(len(images)):
            img, filename = images[i]  # Unpack the image and its filename
            img = transforms2(img)
            img = img.permute(1, 2, 0).numpy()
            img = img.clip(0, 1)
            images[i] = (img, filename)  # Update the list with the transformed image and filename

        n_rows = 3
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 6))

        for i, ax in enumerate(axes.ravel()):
            if i < len(images):
                img, filename = images[i]
                filename = os.path.splitext(filename)[0]  # Remove the ".png" extension
                ax.imshow(img)
                ax.set_title(filename)
                ax.axis('off')

        for i in range(len(images), n_rows * n_cols):
            axes.ravel()[i].axis('off')

        plt.tight_layout()
        plt.savefig('Data\\AugmentedImages.png')
        plot = cv2.imread('Data\\AugmentedImages.png')
        cv2.imshow('Data\\Augmented Images', plot)
        key = cv2.waitKey(0)
        if key == 27:
                cv2.destroyAllWindows()
    
    def show_structure (self):
        torchsummary.summary(self.model, (3, 32, 32))

    def show_acc_loss(self):
        cv2.imshow('Plot', self.plot)
        key = cv2.waitKey(0)
        if key == 27:
                cv2.destroyAllWindows()

    def show_inference (self):
        if hasattr(self, 'image5') and self.image5 is not None:
            input_image = Image.open(self.image5[0])
            input_tensor = self.transform(input_image)
            input_batch = input_tensor.unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_batch)
            output_array = output.numpy()
            _, predicted_class = output.max(1)
            predicted_class_name = self.classes[predicted_class.item()]
            self.text_label2.setText(f"Predicted class: {predicted_class_name}")
            min_value = output_array.min()
            max_value = output_array.max()
            def scale_value(val):
                return max(0, min(100, (val - min_value) / (max_value - min_value) * 100))
            output_percentages = np.vectorize(scale_value)(output_array)
            output_percentages = output_percentages.squeeze()

            plt.figure(figsize=(6, 6))
            plt.bar(self.classes, output_percentages)
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Classes')
            plt.ylabel('Percentage (%)')
            plt.title('Prediction Probability Distribution')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.ylim(0, 100)
            plt.savefig('Data\\PredictionProbability.png')
            plott = cv2.imread('Data\\PredictionProbability.png')
            cv2.imshow('Data\\Prediction Probability', plott)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()

class spaceH(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('space')
        self.setGeometry(100, 100, 20, 900)
        self.setFixedSize(20, 900)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        spacer = QWidget()  
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        spacer.setFixedWidth(5)
        spacer.setStyleSheet("background-color: green;")
        self.layout.addWidget(spacer)

class spaceV(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('space')
        self.setGeometry(100, 100, 500, 20)
        self.setFixedSize(500, 20)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        spacer = QWidget()  
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        spacer.setFixedHeight(5)
        spacer.setStyleSheet("background-color: green;")
        self.layout.addWidget(spacer)

def main():
    app = QApplication(sys.argv)
    window = QMainWindow()
    flags = window.windowFlags()
    flags |= Qt.MSWindowsFixedSizeDialogHint  # Fixed size
    flags &= ~Qt.WindowMaximizeButtonHint     # Remove maximize button
    window.setWindowFlags(flags)
    mdi_area = QMdiArea(window)
    window.setCentralWidget(mdi_area)

    sub_window1 = QMdiSubWindow()
    sub_window1.setWindowFlags(sub_window1.windowFlags() & ~Qt.WindowCloseButtonHint & ~Qt.WindowMaximizeButtonHint & ~Qt.WindowMinimizeButtonHint | Qt.FramelessWindowHint)
    mdi_area.addSubWindow(sub_window1)
    sub_window1.setWidget(Iask1())
    sub_window1.show()

    sub_window_space = QMdiSubWindow()
    sub_window_space.setWindowFlags(sub_window_space.windowFlags() & ~Qt.WindowCloseButtonHint & ~Qt.WindowMaximizeButtonHint & ~Qt.WindowMinimizeButtonHint | Qt.FramelessWindowHint)
    mdi_area.addSubWindow(sub_window_space)
    sub_window_space.setWidget(spaceH())
    sub_window_space.show()
    
    sub_window2 = QMdiSubWindow()
    sub_window2.setWindowFlags(sub_window2.windowFlags() & ~Qt.WindowCloseButtonHint & ~Qt.WindowMaximizeButtonHint & ~Qt.WindowMinimizeButtonHint | Qt.FramelessWindowHint)
    mdi_area.addSubWindow(sub_window2)
    sub_window2.setWidget(Task2())
    sub_window2.show()

    sub_window_space2 = QMdiSubWindow()
    sub_window_space2.setWindowFlags(sub_window_space2.windowFlags() & ~Qt.WindowCloseButtonHint & ~Qt.WindowMaximizeButtonHint & ~Qt.WindowMinimizeButtonHint | Qt.FramelessWindowHint)
    mdi_area.addSubWindow(sub_window_space2)
    sub_window_space2.setWidget(spaceH())
    sub_window_space2.show()

    sub_window3 = QMdiSubWindow()
    sub_window3.setWindowFlags(sub_window3.windowFlags() & ~Qt.WindowCloseButtonHint & ~Qt.WindowMaximizeButtonHint & ~Qt.WindowMinimizeButtonHint | Qt.FramelessWindowHint)
    mdi_area.addSubWindow(sub_window3)
    sub_window3.setWidget(Task3())
    sub_window3.show()

    sub_window4 = QMdiSubWindow()
    sub_window4.setWindowFlags(sub_window4.windowFlags() & ~Qt.WindowCloseButtonHint & ~Qt.WindowMaximizeButtonHint & ~Qt.WindowMinimizeButtonHint | Qt.FramelessWindowHint)
    mdi_area.addSubWindow(sub_window4)
    sub_window4.setWidget(Task4())
    sub_window4.show()

    sub_window_space3 = QMdiSubWindow()
    sub_window_space3.setWindowFlags(sub_window_space3.windowFlags() & ~Qt.WindowCloseButtonHint & ~Qt.WindowMaximizeButtonHint & ~Qt.WindowMinimizeButtonHint | Qt.FramelessWindowHint)
    mdi_area.addSubWindow(sub_window_space3)
    sub_window_space3.setWidget(spaceH())
    sub_window_space3.show()

    sub_window5 = QMdiSubWindow()
    sub_window5.setWindowFlags(sub_window5.windowFlags() & ~Qt.WindowCloseButtonHint & ~Qt.WindowMaximizeButtonHint & ~Qt.WindowMinimizeButtonHint | Qt.FramelessWindowHint)
    mdi_area.addSubWindow(sub_window5)
    sub_window5.setWidget(Task5())
    sub_window5.show()

    #layout
    sub_window1.setGeometry(0,0,435,900)
    sub_window_space.setGeometry(435,0,20,900)
    sub_window2.setGeometry(455,0,435,450)
    sub_window_space2.setGeometry(890,0,20,900)
    sub_window3.setGeometry(455,450,435,450)
    sub_window4.setGeometry(910,0,435,900)
    sub_window_space3.setGeometry(1345,0,20,900)
    

    window.setGeometry(100, 100, 1800, 900)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
