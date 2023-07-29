# Augmented Reality for daVinci Surgical Robot

The repository contains code to project the tool tips of the Da Vinci robot to the stereo endoscopic camera. This is accomplished by training a deep neural network to map 3D positions of the tool tips to 2D image space for the left and right camera.

## Dependencies

### Python dependencies

* Python 3
* Tensorflow
* Numpy
* Pandas
* Scikit-learn
* OpenCV

### ROS packages required

* image_geometry
* message_filters
* tf2
* tf
* tf2_ros


## Data Collection

The data pertaining to tool and camera positions, along with camera images are collected as bag files using the GUI developed by careslab and the dvrk system developed by John's Hopkins University. The bag files contain the following topics:
* `/dvrk/ECM/position_cartesian_current`
* `/dvrk/PSM1/position_cartesian_current`
* `/dvrk/PSM2/position_cartesian_current`
* `/image_raw_left`
* `/image_raw_right`

Each bag file contains approximately three minutes of data and the images contain tool tips withing the field of view, painted red and green.

## Data Extraction

The bag data needs to be extracted and stored in a form that can be fed into a neural network. This is accomplished by using two scripts:
* First, launch the time_filter node with the command `roslaunch time_filter time_filter.launch` and in a separate terminal, play the bag file with the recorded data. The ros node extracts the positions of tool tips and the camera and writes it to an excel file. The location and name of the files can be modified in line 33 and 34 of 'time_filter.py'. The node also extracts the images and saves it the left and right folder under the path specified in line 33.

* Once the tool 3D positions are extracted, the script "extract_tool_points.py" gets the pixel locations of the tool for both the left and right camera. The script takes the folder name as argument which should be folder containing in the extracted data from the above point. The extracted pixel positions are further saved in two separate csv files along with an additional file to save any exceptions. The script used HSV based thresholds to detect green and red points but it is susceptible to poor lighting conditions. The exceptions file lists all frames where either of the tools were not identified. If the are a lot of exceptions, it is a good idea to tinker with the HSV values to get the most points. If the exceptions are few, then it might be faster to open the particular image and note down the position of red and green tip centers as accurately as possible.

## Data Cleaning

The data from the different files are then combined into a single csv file with 10 columns. 6 input columns corresponding to the xyz positions of the tool and camera and 4 output columns for the pixel locations of the left and right camera. This step is done manually and could be automated in the future.

## Neural Network Training

The script "train.py" under "NN/Tf" contains the code to create a dense neural network and train it with the given data. Line 15 takes in the csv file name containing the combined data. Lines 40 and 42 specify the training parameters used and line 63 saves the trained model. 
