import numpy as np
import queue
import cv2


class PolyLine:
    def __init__(self, ys, xs):
        self.__coeffs = np.polyfit(ys, xs, 2)

    def apply(self, ys):
        return self.__coeffs[0] * ys ** 2 + self.__coeffs[1] * ys + self.__coeffs[2]

    def to_cv_points(self, ys):
        return np.array(list(zip(self.apply(ys), ys)), np.int32)

    def r_curvature(self, y):
        nominator = (1 + (2 * self.__coeffs[0] * y + self.__coeffs[1]) ** 2) ** 1.5
        denominator = np.absolute(2 * self.__coeffs[0])
        return nominator / denominator


def all_not_none(*items):
    return all(v is not None for v in items)


class LanesDetector:
    def __init__(self, img_shape, sliding_window_width, vertical_windows_num, window_shift_tolerance, img_buffer_len):
        self.__img_buffer_len = img_buffer_len
        self.__window_shift_tolerance = window_shift_tolerance
        self.__vertical_windows_num = vertical_windows_num
        self.__sliding_window_width = sliding_window_width
        self.__img_shape = img_shape
        self.__image_queue = queue.deque()
        self.__tracking_attempts = 0

        self.left_line = None
        self.right_line = None

    def next_image(self, input_image):
        if input_image.shape != self.__img_shape:
            raise Exception("Wrong image shape! Expected: {0}, but got {1}".format(self.__img_shape, input_image.shape))

        self.__image_queue.append(input_image)
        while len(self.__image_queue) > self.__img_buffer_len:
            self.__image_queue.popleft()

        if len(self.__image_queue) != self.__img_buffer_len:
            self.__try_set_new_lines(None, None)
            return

        self.__detect()

    def __try_set_new_lines(self, left, right):
        if left is None or right is None:
            return False

        # Simple sanity check - lines should have enough space between them.
        h = self.__img_shape[0] * 0.9
        distance = left.apply(h) - right.apply(h)
        if abs(distance) < 500:
            return False

        # Another sanity check - radiuses of both lines should be close to equal
        lr = left.r_curvature(h)
        rr = right.r_curvature(h)
        radius_f = max(lr, rr) / min(lr, rr)
        if radius_f > 2:
            return False

        self.left_line = left
        self.right_line = right
        return True

    def __detect(self):
        input_image = self.merge_queue()

        all_left_xs = []
        all_left_ys = []
        all_right_xs = []
        all_right_ys = []

        histogram = np.sum(input_image[int(input_image.shape[0] / 2):, :], axis=0)

        left_x, right_x = self.__get_left_right_peaks(histogram)

        bands = self.__vertical_windows_num
        for i in range(bands):
            left_xs, left_ys = self.__get_window(i, bands, input_image, left_x)
            right_xs, right_ys = self.__get_window(i, bands, input_image, right_x)

            if len(left_xs) > 50:
                left_x = np.average(left_xs)
            if len(right_xs) > 50:
                right_x = np.average(right_xs)

            all_left_xs.extend(left_xs)
            all_left_ys.extend(left_ys)

            all_right_xs.extend(right_xs)
            all_right_ys.extend(right_ys)

        if len(all_left_xs) == 0 or len(all_right_xs) == 0:
            self.__try_set_new_lines(None, None)
            return

        left = PolyLine(all_left_ys, all_left_xs)
        right = PolyLine(all_right_ys, all_right_xs)

        self.__try_set_new_lines(left, right)

    def merge_queue(self):
        return np.clip(np.sum(self.__image_queue, axis=0), 0, 1)

    def __get_left_right_peaks(self, histogram):
        max = np.argmax(histogram)
        neighbourhood = self.__sliding_window_width
        left_bound = max - neighbourhood
        right_bound = max + neighbourhood

        left_histogram = histogram[:left_bound]
        right_histogram = histogram[right_bound:]

        peaks = [(histogram[max], max)]
        if len(left_histogram) != 0:
            max_left = np.argmax(left_histogram)
            peaks.append((histogram[max_left], max_left))

        if len(right_histogram) != 0:
            max_right = np.argmax(right_histogram) + right_bound
            peaks.append((histogram[max_right], max_right))

        best = sorted(sorted(peaks, key=lambda x: -x[0])[:2], key=lambda x: x[1])

        return best[0][1], best[1][1]

    def __get_window(self, band, bands, img, x):

        img_width = img.shape[1]
        img_height = img.shape[0]

        left = x - self.__sliding_window_width / 2

        band_height = int(img.shape[0] / bands)

        top = img.shape[0] - band_height * band - band_height

        xs = []
        ys = []

        for i in range(band_height):
            for j in range(self.__sliding_window_width):
                x = int(j + left)
                y = int(i + top)

                if x >= img_width or y >= img_height or x < 0 or y < 0:
                    continue

                if img[y][x] > 0:
                    xs.append(x)
                    ys.append(y)

        return np.array(xs), np.array(ys)
