import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
import os


class ChessboardCalibrator:
    def __init__(self, image_size=None, file_name=None):
        self.__image_points = []
        self.__object_points = []
        if image_size is not None:
            self.__image_size = image_size
            self.__mtx = None
            self.__dist = None
        elif file_name is not None:
            data = pickle.load(open(file_name, "rb"))
            self.__image_size = data["image_size"]
            self.__mtx = data["mtx"]
            self.__dist = data["dist"]
        else:
            raise Exception("Either image_size or file_name should be filled")

    def add_chessboard_image(self, img, pattern_size):
        if img.shape != self.__image_size:
            raise Exception("Image size mismatch!")

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ret:
            return False

        self.__image_points.append(corners)
        self.__object_points.append(self.__create_object_points(pattern_size))
        self.__mtx = None
        self.__dist = None
        return True

    @staticmethod
    def __create_object_points(pattern_size):
        w = pattern_size[0]
        h = pattern_size[1]
        object_points = np.zeros((w * h, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        return object_points

    def save(self, file_name):
        self.__ensure_calibrated()
        pickle.dump({"image_size": self.__image_size, "mtx": self.__mtx, "dist": self.__dist}, open(file_name, "wb"))

    def __ensure_calibrated(self):
        if self.__mtx is not None:
            return

        if len(self.__image_points) == 0:
            raise Exception("No image points")

        ret, self.__mtx, self.__dist, rvecs, tvecs = cv2.calibrateCamera(self.__object_points, self.__image_points,
                                                                         self.__image_size[0:2], None, None)

    def undistort(self, img):
        if img.shape != self.__image_size:
            raise Exception("Image size mismatch!")

        self.__ensure_calibrated()

        return cv2.undistort(img, self.__mtx, self.__dist, None, self.__mtx)


if __name__ == "__main__":
    def main():
        image_size = (720, 1280, 3)
        out_file = "calibration.p"
        if not os.path.exists(out_file):
            calibrator = ChessboardCalibrator(image_size)
            for img_file in glob.glob("camera_cal/calibration*.jpg"):
                img = mpimg.imread(img_file)

                if img.shape != image_size:
                    print("Unexpected image shape", img.shape)
                    continue

                if not calibrator.add_chessboard_image(img, (9, 6)):
                    if not calibrator.add_chessboard_image(img, (9, 5)):
                        print("Chessboard pattern not found in", img_file)

            calibrator.save(out_file)

        else:
            calibrator = ChessboardCalibrator(file_name=out_file)

        src = mpimg.imread("input/test_images/test1.jpg")
        dst = calibrator.undistort(src)

        result = np.concatenate([src, dst], axis=1)
        plt.imshow(result)
        plt.show()


    main()
