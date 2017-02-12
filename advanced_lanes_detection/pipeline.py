import calibrate
import warp
import util
import threshold
import matplotlib.pyplot as plt
import numpy as np
import detector
import glob
import cv2
from moviepy.editor import *
from moviepy import *

img_shape = (720, 1280)
chessboard_calibrator = calibrate.ChessboardCalibrator(file_name="calibration.p")
warper = warp.Warper(img_shape)
lanes_detector = detector.LanesDetector(img_shape, 50, 10, 50, 3, 5)


def process_image(img):
    img = util.img_to_float(img)

    calibrated = chessboard_calibrator.undistort(img)
    thresholded = threshold.find_lines(calibrated)
    warped = warper.warp(thresholded)

    lanes_detector.next_image(warped)

    if lanes_detector.left_line is None or lanes_detector.right_line is None:
        return img

    left_line = lanes_detector.left_line
    right_line = lanes_detector.right_line

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

    lanes_img = np.zeros((*img_shape, 3), np.float32)

    left_poly = left_line.to_cv_points(ploty)
    right_poly = right_line.to_cv_points(ploty)

    concatenated = np.concatenate([left_poly, right_poly[::-1]])
    cv2.fillPoly(lanes_img, [concatenated], (0, 1, 0))
    cv2.polylines(lanes_img, [left_poly], False, (1.0, 0.0, 0), thickness=30)
    cv2.polylines(lanes_img, [right_poly], False, (1.0, 0.0, 0), thickness=30)

    lanes_img = warper.unwarp(lanes_img)

    result =  cv2.addWeighted(img, 0.8, lanes_img, 0.2, 0)

    return (result * 255).astype(np.uint8)


def warped_colored(img):
    img = img.astype(np.float32) / 255.0

    calibrated = chessboard_calibrator.undistort(img)
    warped = warper.warp(calibrated)

    result = warped
    return (result * 255).astype(np.uint8)


def warped_thresholded(img):
    img = img.astype(np.float32) / 255.0

    calibrated = chessboard_calibrator.undistort(img)
    thresholded = threshold.find_lines(calibrated)
    warped = warper.warp(thresholded)

    result = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)
    return (result * 255).astype(np.uint8)


out_video = 'result.mp4'
in_video = "input/project_video.mp4"
main_video = VideoFileClip(in_video)
main_video = main_video.fl_image(process_image)

additional_video = VideoFileClip(in_video)
additional_video = additional_video.fl_image(warped_thresholded)

result_video = clips_array([[main_video], [additional_video]])
result_video.write_videofile(out_video, audio=False, fps=4)
