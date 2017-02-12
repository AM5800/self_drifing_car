import calibration
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
from calibration import ChessboardCalibrator
from util import *

img_shape = (720, 1280)
chessboard_calibrator = ChessboardCalibrator(file_name="calibration.p")
warper = warp.Warper(img_shape)
lanes_detector = detector.LanesDetector(img_shape, 100, 5, 50, 3)


def process_image(img):
    img = img_to_float(img)

    calibrated = chessboard_calibrator.undistort(img)
    thresholded = threshold.threshold_lines(calibrated)
    warped = warper.warp(thresholded)

    lanes_detector.next_image(warped)

    if lanes_detector.left_line is None or lanes_detector.right_line is None:
        return img

    left_line = lanes_detector.left_line
    right_line = lanes_detector.right_line

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

    overlay_img = np.zeros((*img_shape, 3), np.float32)

    left_poly = left_line.to_cv_points(ploty)
    right_poly = right_line.to_cv_points(ploty)

    concatenated = np.concatenate([left_poly, right_poly[::-1]])

    color_green = (0, 1, 0)
    color_red = (1.0, 0.0, 0)

    cv2.fillPoly(overlay_img, [concatenated], color_green)
    cv2.polylines(overlay_img, [left_poly], False, color_red, thickness=30)
    cv2.polylines(overlay_img, [right_poly], False, color_red, thickness=30)

    overlay_img = warper.unwarp(overlay_img)

    result = cv2.addWeighted(img, 0.8, overlay_img, 0.2, 0)
    result = util.img_to_int(result)

    print_overlay_info(left_line, result, right_line)

    return result


def draw_offset_marker(result, x, y, color):
    cv2.line(result, (x, y), (x, y + 20), color, thickness=6)


def print_overlay_info(left_line, result, right_line):
    h = img_shape[0] - 100
    curvature = left_line.r_curvature(h) + right_line.r_curvature(h) / 2
    left_x = left_line.apply(h)
    right_x = right_line.apply(h)
    offset = (right_x - left_x) / 2 + left_x - img_shape[1] / 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    curvature_msg = "Curvature: {0:.2f}".format(curvature)
    cv2.putText(result, curvature_msg, (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    offset_msg = "Offset: {0:.2f}".format(offset)
    cv2.putText(result, offset_msg, (50, 80), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    draw_offset_marker(result, int(img_shape[1] / 2), img_shape[0] - 50, (0, 0, 255))
    draw_offset_marker(result, int((left_x + right_x)/2), img_shape[0] - 30, (0, 255, 0))


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

# additional_video = VideoFileClip(in_video)
# additional_video = additional_video.fl_image(warped_thresholded)
#
# result_video = clips_array([[main_video], [additional_video]])

main_video.write_videofile(out_video, audio=False)
