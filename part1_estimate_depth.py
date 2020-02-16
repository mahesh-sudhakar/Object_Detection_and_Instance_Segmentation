import os
import sys

import cv2
import numpy as np
import kitti_dataHandler


def mse(imageA, imageB):
    # taken from https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/

    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def main():
    ################
    # Options
    ################
    # Input dir and output dir
    disp_dir = 'data/test/disparity/'
    output_dir = 'data/test/est_depth/'
    # gt_depth_dir = 'data/train/gt_depth/'
    calib_dir = 'data/test/calib/'
    sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in (sample_list):

        # Read disparity map
        disp_map = cv2.imread(disp_dir + sample_name + '.png')

        # Read ground truth depth map
        # gt_depth = cv2.imread(gt_depth_dir+sample_name+'.png')
        # print("GT depth image : ")
        # cv2_imshow(gt_depth)

        # Read calibration info
        calib_obj = kitti_dataHandler.FrameCalib()
        calib_obj = kitti_dataHandler.read_frame_calib(calib_dir + sample_name + '.txt')

        stereo_obj = kitti_dataHandler.StereoCalib()
        stereo_obj = kitti_dataHandler.get_stereo_calibration(calib_obj.p2, calib_obj.p3)

        # Calculate depth (z = f*B/disp)
        fB = stereo_obj.f * stereo_obj.baseline
        print("fB for image " + sample_name + " is : ", fB)

        z = fB / disp_map

        # Discard pixels past 80m and less than 10cm
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if z[i][j][0] > 80:
                    z[i][j] = 0

                if z[i][j][0] < 0.1:
                    z[i][j] = 0

        print('\n After discarding...')
        print("Maximum pixel depth after discarding : ", np.max(z))

        # error = mse(gt_depth, z)
        # print("MSE for image "+sample_name+" is : %.3f" %error)

        print("\n Calculated depth map for image " + sample_name + " is : ")
        cv2.imshow('Result: ', z)
        cv2.waitKey(0)

        # Save depth map
        cv2.imwrite(output_dir + sample_name + '.png', z)
        print("\n")


if __name__ == '__main__':
    main()