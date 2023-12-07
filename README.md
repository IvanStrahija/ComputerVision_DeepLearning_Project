You can open the GUI by running the code from Python Interpreter (VSCode was used). File name: 'main.py'

To run the code please open the directory 'ComputerVision_DeepLearning_Project' and set it as a working directory in your interpreter (VSCode).
All the neccessary files (except the VGG19 trained model - 60 epochs - for the Task 5 image recognition) for program to work are in this directory so please don't move any files or rename them. Images for demonstration can be loaded from other maps if it is required. 
# Please use your own trained model or train it with the VGG_Trainer file and then define the path to the model in line 877 -> nn_path = '' #

For Tasks 1 and 2 chessboard with pattern size (11,8) were used. Can be changed for other patterns accordingly.
Load the images for Task 1 and Task 2 from folders that only contain images.

Task 1 detects corners of chessboard image and shows the functions of camera calibration.
Task 2 presents the augmented reality capabilities of OpenCV with the function of writing letters onto the chessboard images.
Task 3 shows the stereo disparity map and the ability to find the selected point on the imageL on the corresponding imageR.
Task 4 presents the use of SIFT algorithm for finding keypoints on images and connecting two images into one based on the matched keypoints.
Task 5 presents trained CIFAR10 Classifier using VGG19 with BN. Training dataset: 50000 images in total. Validation dataset: 10000 images in total.
