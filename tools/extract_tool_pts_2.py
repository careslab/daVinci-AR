#!/usr/bin/env python3
import cv2
import csv
import os
import argparse
import numpy as np


class ToolExtractor:

    def __init__(self, folder_name, csv_name):

        self.csv_l = os.getcwd() + "/" + folder_name + "/" + csv_name + "_left.csv"
        self.csv_r = os.getcwd() + "/" + folder_name + "/" + csv_name + "_right.csv"
        self.exception_file = os.getcwd() + "/" + folder_name + "/exceptions.csv"
        self.directory = os.getcwd() + "/" + folder_name
        self.left_imgs = os.getcwd() + "/" + folder_name + "/Left"
        self.right_imgs = os.getcwd() + "/" + folder_name + "/Right"

        # self.lower_thresh_g = np.array([40, 55, 75], np.uint8)
        # self.upper_thresh_g = np.array([70, 255, 255], np.uint8)

        self.lower_thresh_g = np.array([20, 110, 110], np.uint8)  # [45,101,52]
        self.upper_thresh_g = np.array([70, 245, 255], np.uint8)  # [70,245,255]

        self.lower_thresh_r = np.array([130, 45, 70], np.uint8)  # [130, 30, 69],
        self.upper_thresh_r = np.array([180, 195, 255], np.uint8)  # [180, 255, 255]

        self.image_len = len(os.listdir(self.left_imgs))

        self.exception = []

    def run(self):
        print("Procesing Left camera images")
        self.process_imgs("Left")
        print("Procesing Right camera images")
        self.process_imgs("Right")
        print("Processing exceptions")
        self.processException()

    def process_imgs(self, camera):

        if camera == 'Left':
            # path = self.left_imgs
            directory = self.directory + "/Left"
            csv_file = self.csv_l
        elif camera == "Right":
            # path = self.right_imgs
            directory = self.directory + "/Right"
            csv_file = self.csv_r
        else:
            print("Invalid camera value")
            return

        with open(csv_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            # for i in range(self.image_len):
            for file in os.scandir(directory):
                # img_name = path + "/" + "img" + str(i) + ".png"
                img_name = file.path
                just_name = file.name
                frame_id = just_name[3:-4]
                # print(img_name)
                img = cv2.imread(img_name)
                blur = cv2.GaussianBlur(img, (3, 3), 0)
                hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

                tool_g = self.getTool(hsv, "green", img_name)
                tool_r = self.getTool(hsv, "red", img_name)

                # if (tool_g):
                csvwriter.writerow([frame_id, img_name, tool_g[0], tool_g[1], tool_r[0], tool_r[1]])
                # if (tool_r):
                #     csvwriter.writerow([frame_id, img_name, 'red', tool_r[0], tool_r[1]])

        csvfile.close()

    def getTool(self, img, tool, img_name):

        out = []

        if tool == "green":
            mask = cv2.inRange(img, self.lower_thresh_g, self.upper_thresh_g)
        elif tool == "red":
            mask = cv2.inRange(img, self.lower_thresh_r, self.upper_thresh_r)
        else:
            print("invalid tool")
            return out

        contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        sorted_areas = np.sort(areas)

        if (sorted_areas.any()) and sorted_areas[-1] >= 5.2:
            cnt = contours[areas.index(sorted_areas[-1])]  # the biggest contour

            m = cv2.moments(cnt)
            out.append(int(m["m10"] / m["m00"]))
            out.append(int(m["m01"] / m["m00"]))
        else:
            self.exception.append(f"no {tool} in {img_name}")
            out.append(-1)
            out.append(-1)

        return out

    def processException(self):
        if self.exception:
            with open(self.exception_file, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                for i in range(len(self.exception)):
                    csvwriter.writerow([self.exception[i]])
            csvfile.close()
        else:
            print("No Exceptions!! :D")


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--folder", type=str, help="name of folder containing datset; defaults to dataset2")

    args = argParser.parse_args()

    if args.folder:
        folder_name = args.folder
    else:
        folder_name = "dataset5"

    tools = ToolExtractor(folder_name=folder_name, csv_name="tool_img_pos")
    tools.run()
