import pickle
from queue import deque

import util
import cv2
import numpy as np
from moviepy.editor import *
from vehicle_classifier import VehicleClassifierInterface
from scipy.ndimage.measurements import label


def search_far(img, classifier: VehicleClassifierInterface):
    img_width = img.shape[1]

    region_top = 386
    region_bottom = 482

    window_size = 64
    x_stride = 16
    y_stride = 16

    return search_window(classifier, img, img_width, region_bottom, region_top, window_size, x_stride, y_stride)


def search_middle(img, classifier: VehicleClassifierInterface):
    img_width = img.shape[1]

    region_top = 362
    region_bottom = 586

    window_size = 128
    x_stride = 32
    y_stride = 32

    return search_window(classifier, img, img_width, region_bottom, region_top, window_size, x_stride, y_stride)


def search_near(img, classifier: VehicleClassifierInterface):
    img_width = img.shape[1]

    region_top = 330
    region_bottom = 650

    window_size = 256
    x_stride = 64
    y_stride = 64

    return search_window(classifier, img, img_width, region_bottom, region_top, window_size, x_stride, y_stride)


def search_window(classifier: VehicleClassifierInterface, img, img_width, region_bottom, region_top, window_size,
                  x_stride, y_stride):
    result = []
    for x in range(0, img_width, x_stride):
        for y in range(region_top, region_bottom - window_size + 1, y_stride):
            window = img[y:y + window_size, x:x + window_size, :]
            if window.shape[0:2] != (window_size, window_size):
                continue

            window = cv2.resize(window, classifier.get_image_size())

            is_vehicle = classifier.is_vehicle(window)

            if is_vehicle:
                left = (x, y)
                right = (x + window_size, y + window_size)
                result.append((left, right))
    return result


def draw_boxes(img, boxes):
    for left, right in boxes:
        img = cv2.rectangle(img, (left[0], left[1]), (right[0], right[1]), (0, 1, 0), 3)


class VehicleTracker:
    def __init__(self, queue_size: int, img_size, classifier: VehicleClassifierInterface, heat_threshold):
        self.__heat_threshold = heat_threshold
        self.__classifier = classifier
        self.__img_size = img_size
        self.__queue_size = queue_size
        self.__detections_queue = deque()

    def process(self, img):
        near = search_middle(img, self.__classifier)
        far = search_far(img, self.__classifier)
        detections = near + far

        self.__detections_queue.append(detections)

        while len(self.__detections_queue) > self.__queue_size:
            self.__detections_queue.popleft()

        if len(self.__detections_queue) == self.__queue_size:
            return self.__detect_vehicles(img)

        return img

    def __detect_vehicles(self, img):
        heatmap = np.zeros(self.__img_size)

        for detections in self.__detections_queue:
            local_heatmap = np.zeros(self.__img_size)
            for bbox in detections:
                local_heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] = 0.1
            heatmap += local_heatmap

        heatmap[heatmap <= self.__heat_threshold] = 0

        return self.draw_labeled_bboxes(img, label(heatmap))

    # Copied from lecture
    @staticmethod
    def draw_labeled_bboxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 1), 6)
        # Return the image
        return img


def process_image(img, tracker: VehicleTracker):
    img = util.img_to_float(img)
    return util.img_to_int(tracker.process(img))


if __name__ == "__main__":
    img_size = (720, 1280)
    classifier = pickle.load(open("classifier.p", "rb"))
    tracker = VehicleTracker(10, img_size, classifier, 0.6)

    in_video = "project_video.mp4"
    out_video = "result.mp4"

    main_video = VideoFileClip(in_video)
    main_video = main_video.fl_image(lambda img: process_image(img, tracker))
    main_video.write_videofile(out_video, audio=False)

    # img = util.load_image_float("test_images/test7.png")
    # draw_boxes(img, search_middle(img, classifier))
    # draw_boxes(img, search_far(img, classifier))
    # draw_boxes(img, search_near(img, classifier))
    # plt.imshow(img)
    # plt.show()
