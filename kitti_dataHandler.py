import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import csv
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import os


class FrameCalib:
    """Frame Calibration

    Fields:
        p0-p3: (3, 4) Camera P matrices. Contains extrinsic and intrinsic parameters.
        r0_rect: (3, 3) Rectification matrix
        velo_to_cam: (3, 4) Transformation matrix from velodyne to cam coordinate
            Point_Camera = P_cam * R0_rect * Tr_velo_to_cam * Point_Velodyne
        """

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.velo_to_cam = []


def read_frame_calib(calib_file_path):
    """Reads the calibration file for a sample

    Args:
        calib_file_path: calibration file path

    Returns:
        frame_calib: FrameCalib frame calibration
    """

    data_file = open(calib_file_path, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    frame_calib = FrameCalib()
    frame_calib.p0 = p_all[0]
    frame_calib.p1 = p_all[1]
    frame_calib.p2 = p_all[2]
    frame_calib.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calib.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calib.velo_to_cam = np.reshape(tr_v2c, (3, 4))

    return frame_calib


class StereoCalib:
    """Stereo Calibration

    Fields:
        baseline: distance between the two camera centers
        f: focal length
        k: (3, 3) intrinsic calibration matrix
        p: (3, 4) camera projection matrix
        center_u: camera origin u coordinate
        center_v: camera origin v coordinate
        """

    def __init__(self):
        self.baseline = 0.0
        self.f = 0.0
        self.k = []
        self.center_u = 0.0
        self.center_v = 0.0


def krt_from_p(p, fsign=1):
    """Factorize the projection matrix P as P=K*[R;t]
    and enforce the sign of the focal length to be fsign.


    Keyword Arguments:
    ------------------
    p : 3x4 list
        Camera Matrix.

    fsign : int
            Sign of the focal length.


    Returns:
    --------
    k : 3x3 list
        Intrinsic calibration matrix.

    r : 3x3 list
        Extrinsic rotation matrix.

    t : 1x3 list
        Extrinsic translation.
    """
    s = p[0:3, 3]
    q = np.linalg.inv(p[0:3, 0:3])
    u, b = np.linalg.qr(q)
    sgn = np.sign(b[2, 2])
    b = b * sgn
    s = s * sgn

    # If the focal length has wrong sign, change it
    # and change rotation matrix accordingly.
    if fsign * b[0, 0] < 0:
        e = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    if fsign * b[2, 2] < 0:
        e = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    # If u is not a rotation matrix, fix it by flipping the sign.
    if np.linalg.det(u) < 0:
        u = -u
        s = -s

    r = np.matrix.transpose(u)
    t = np.matmul(b, s)
    k = np.linalg.inv(b)
    k = k / k[2, 2]

    # Sanity checks to ensure factorization is correct
    if np.linalg.det(r) < 0:
        print('Warning: R is not a rotation matrix.')

    if k[2, 2] < 0:
        print('Warning: K has a wrong sign.')

    return k, r, t


def get_stereo_calibration(left_cam_mat, right_cam_mat):
    """Extract parameters required to transform disparity image to 3D point
    cloud.

    Keyword Arguments:
    ------------------
    left_cam_mat : 3x4 list
                   Left Camera Matrix.

    right_cam_mat : 3x4 list
                   Right Camera Matrix.


    Returns:
    --------
    stereo_calibration_info : Instance of StereoCalibrationData class
                              Placeholder for stereo calibration parameters.
    """

    stereo_calib = StereoCalib()
    k_left, r_left, t_left = krt_from_p(left_cam_mat)
    _, _, t_right = krt_from_p(right_cam_mat)

    stereo_calib.baseline = abs(t_left[0] - t_right[0])
    stereo_calib.f = k_left[0, 0]
    stereo_calib.k = k_left
    stereo_calib.center_u = k_left[0, 2]
    stereo_calib.center_v = k_left[1, 2]

    return stereo_calib


class ObjectLabel:
    """Object Label

    Fields:
        type (str): Object type, one of
            'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
            'Cyclist', 'Tram', 'Misc', or 'DontCare'
        truncation (float): Truncation level, float from 0 (non-truncated) to 1 (truncated),
            where truncated refers to the object leaving image boundaries
        occlusion (int): Occlusion level,  indicating occlusion state:
            0 = fully visible,
            1 = partly occluded,
            2 = largely occluded,
            3 = unknown
        alpha (float): Observation angle of object [-pi, pi]
        x1, y1, x2, y2 (float): 2D bounding box of object in the image. (top left, bottom right)
        h, w, l: 3D object dimensions: height, width, length (in meters)
        t: 3D object centroid x, y, z in camera coordinates (in meters)
        ry: Rotation around Y-axis in camera coordinates [-pi, pi]
        score: Only for results, indicating confidence in detection, needed for p/r curves.
    """

    def __init__(self):
        self.type = None  # Type of object
        self.truncation = 0.0
        self.occlusion = 0
        self.alpha = 0.0
        self.x1 = 0.0
        self.y1 = 0.0
        self.x2 = 0.0
        self.y2 = 0.0
        self.h = 0.0
        self.w = 0.0
        self.l = 0.0
        self.t = (0.0, 0.0, 0.0)
        self.ry = 0.0
        self.score = 0.0


def read_labels(label_dir, sample_name):
    """Reads in label data file from Kitti Dataset
    Args:
        label_dir: label directory
        sample_name: sample_name
    Returns:
        obj_list: list of ObjectLabels
    """

    # Check label file
    label_path = label_dir + '/{}.txt'.format(sample_name)

    if not os.path.exists(label_path):
        raise FileNotFoundError('Label file could not be found:', label_path)
    if os.stat(label_path).st_size == 0:
        return []

    labels = np.loadtxt(label_path, delimiter=' ', dtype=str, ndmin=2)
    num_rows, num_cols = labels.shape
    if num_cols not in [15, 16]:
        raise ValueError('Invalid label format')

    num_labels = num_rows
    is_results = num_cols == 16

    obj_list = []
    for obj_idx in np.arange(num_labels):
        obj = ObjectLabel()

        # Fill in the object list
        obj.type = labels[obj_idx, 0]
        obj.truncation = float(labels[obj_idx, 1])
        obj.occlusion = float(labels[obj_idx, 2])
        obj.alpha = float(labels[obj_idx, 3])

        obj.x1, obj.y1, obj.x2, obj.y2 = (labels[obj_idx, 4:8]).astype(np.float32)
        obj.h, obj.w, obj.l = (labels[obj_idx, 8:11]).astype(np.float32)
        obj.t = (labels[obj_idx, 11:14]).astype(np.float32)
        obj.ry = float(labels[obj_idx, 14])

        if is_results:
            obj.score = float(labels[obj_idx, 15])
        else:
            obj.score = 0.0

        obj_list.append(obj)

    return np.asarray(obj_list)