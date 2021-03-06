import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import *
from util import *
import detector
import threshold
import warp
from calibration import ChessboardCalibrator

img_shape = (720, 1280)
chessboard_calibrator = ChessboardCalibrator(file_name="calibration.p")
warper = warp.Warper(img_shape)
lanes_detector = detector.LanesDetector(img_shape, 100, 5, 50, 2)

frame = 0


def process_image(img):
    global frame
    frame += 1

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

    left_points = left_line.to_cv_points(ploty)
    right_points = right_line.to_cv_points(ploty)

    concatenated = np.concatenate([left_points, right_points[::-1]])

    color_green = (0, 1, 0)
    color_red = (1, 0, 0)

    cv2.fillPoly(overlay_img, [concatenated], color_green)
    cv2.polylines(overlay_img, [left_points], False, color_red, thickness=30)
    cv2.polylines(overlay_img, [right_points], False, color_red, thickness=30)

    overlay_img = warper.unwarp(overlay_img)

    result = cv2.addWeighted(img, 0.8, overlay_img, 0.2, 0)

    result = img_to_int(result)

    print_overlay_info(left_line, result, right_line)

    if frame == 5:
        save_debug_images(left_line, right_line, result, ploty)

    return result


def set_color(img, xs, ys, color):
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        img[y][x] = color


def save_debug_images(left_line, right_line, result, ploty):
    left_points = left_line.to_cv_points(ploty)
    right_points = right_line.to_cv_points(ploty)

    debug_img = lanes_detector.merge_queue()
    debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
    plt.imsave("out/queue.png", debug_img)

    set_color(debug_img, left_line.xs, left_line.ys, (0, 0, 1))
    set_color(debug_img, right_line.xs, right_line.ys, (1, 0, 0))

    cv2.polylines(debug_img, [left_points], False, (1, 1, 0), thickness=5)
    cv2.polylines(debug_img, [right_points], False, (1, 1, 0), thickness=5)
    plt.imsave("out/detected.png", debug_img)
    plt.imsave("out/final.png", result)


def draw_offset_marker(result, x, y, color):
    cv2.line(result, (x, y), (x, y + 20), color, thickness=6)


def print_overlay_info(left_line, result, right_line):
    h = img_shape[0]  # making measurements at the bottom of the image - where the car is

    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700

    real_left_line = left_line.scale(ym_per_pix, xm_per_pix)
    real_right_line = right_line.scale(ym_per_pix, xm_per_pix)

    curvature = real_left_line.r_curvature(h) + real_right_line.r_curvature(h) / 2
    left_x = left_line.apply(h)
    right_x = right_line.apply(h)
    offset = ((right_x - left_x) / 2 + left_x - img_shape[1] / 2) * xm_per_pix
    font = cv2.FONT_HERSHEY_SIMPLEX
    curvature_msg = "Curvature: {0:.2f} m".format(curvature)
    cv2.putText(result, curvature_msg, (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    offset_msg = "Offset: {0:.2f} m".format(offset)
    cv2.putText(result, offset_msg, (50, 80), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    draw_offset_marker(result, int(img_shape[1] / 2), img_shape[0] - 50, (0, 0, 255))
    draw_offset_marker(result, int((left_x + right_x) / 2), img_shape[0] - 30, (0, 255, 0))


out_video = 'result.mp4'
in_video = "input/project_video.mp4"
main_video = VideoFileClip(in_video)
main_video = main_video.fl_image(process_image)

main_video.write_videofile(out_video, audio=False)
