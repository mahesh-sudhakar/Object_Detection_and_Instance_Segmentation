import os
import sys
from math import ceil

import cv2
import numpy as np
import kitti_dataHandler


def main():

    ################
    # Options
    ################
    # Input dir and output dir
    depth_dir = 'data/test/est_depth/'
    box_dir = 'data/test/yolo/'
    output_dir = 'data/test/est_segmentation/'
    sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in sample_list:
      # Read depth map
      z = cv2.imread(depth_dir+sample_name+'.png')
      print("[INFO] Loading image "+sample_name+" :")

      # generate a dummy mask
      dummy_mask = 255 * np.ones(z.shape)

      # Discard depths less than 10cm from the camera
      ### UPDATE: THIS PORTION IS INCLUDED IN 'part1_estimate_depth.py'

      # Read 2d bbox
      threshold_boxes = np.load(box_dir+sample_name+'.npy')

      # For each bbox
      for l in range(len(threshold_boxes)):
        box_depth = []
        zero_depth = []

        (x, y) = (threshold_boxes[l][0], threshold_boxes[l][1])
        (w, h) = (threshold_boxes[l][2], threshold_boxes[l][3])
        y_limit = y+h
        x_limit = x+w

        # calculate centroid of the bounding box
        center_x = x + (w/2)
        center_y = y + (w/2)

        centroid = z[int(center_y)][int(center_x)][0]
        print("Depth at centroid of bounding box "+str(l+1)+" is : ", centroid)

        if y_limit > z.shape[0]:
          y_limit = z.shape[0]
        if x_limit > z.shape[1]:
          x_limit = z.shape[1]

        for i in range(y , y_limit):
          for j in range(x , x_limit):
            if z[i][j][0] != 0:
              box_depth.append(z[i][j])
            else:
              zero_depth.append(z[i][j])

        # Go inside loop only if there exists atleast a single depth value within
        # the bounding box
        if box_depth :
          # Estimate the average depth of the objects
          avg_box_depth = np.mean(box_depth, axis=(0 , 1))
          print("Average depth of bounding box "+str(l+1)+" is : %.3f" %avg_box_depth)

          # Find the pixels within a certain distance from the centroid
          # Based on the average depth, a threshold is set
          depth_range = int(ceil(avg_box_depth))
          if depth_range <= 12:
            depth_threshold = 0.25
          elif depth_range > 12 and depth_range <= 30:
            depth_threshold = 0.43
          elif depth_range > 30 and depth_range <= 50:
            depth_threshold = 0.35
          else:
            depth_threshold = 0.1

          # The set threshold is used to calculate the minimum and maximum depth range
          # to be included inside the bounding box
          depth_threshold_min = (1-depth_threshold)*avg_box_depth
          depth_threshold_max = (1+depth_threshold)*avg_box_depth

          # Pixels within the min and max depth threshold range are considered to be car's
          # and set to '0' as adviced
          for i in range(y , y_limit):
            for j in range(x , x_limit):
              if z[i][j][0] <= depth_threshold_max and z[i][j][0] >= depth_threshold_min:
                  dummy_mask[i][j] = 0
              else:
                dummy_mask[i][j] = 255

        # if no depth value exists
        if not box_depth:
          print("[WARNING] Bypassing as no depth value detected")


      print("\n Predicted segment mask for image "+sample_name+" is : ")
      cv2.imshow('Result: ', dummy_mask)


      # Save the segmentation mask
      cv2.imwrite(output_dir+sample_name+'.png',dummy_mask)
      print("\n")
      cv2.waitKey(0)

if __name__ == '__main__':
    main()
