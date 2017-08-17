import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
	
# All tweakable parameters
grayscale_argument = [0.8, 0.1, 0.1] # This value highlights yellow lanes very well
horizon_relative_height = 0.55 # Relative height of the horizon from top of the image
trapeze_top = 0.61 # Height of the top part of region of interest
trapeze_display_top = 0.64 # Save as above, but it only trims lines and not image
lane_angle_limit = 0.5 # Approx 28 degree.
gaussian_blur_kernel = 3
canny_low_threshold = 100
canny_high_threshold = 200
rho = 1
theta = np.pi / 180
threshold = 50
min_line_length = 25
max_line_gap = 3


def better_grayscale(img):
    # In challenge video there is a piece where yellow line dissappears from screen
    # after converting image to grayscale.
    # Actually, grayscale conversion can be parametrized.
    # I've opened Photoshop and played with different parameters of conversion
    # grayscale_argument - it a result of my experiments
    
    args = np.array(grayscale_argument)
    return np.dot(img, args).astype(np.uint8)


def curtain(img, curtain_height):
    # Helper function which discards upper part of the 'img', ending at 'curtain_height'
    img_width = img.shape[1]
    img_height = img.shape[0]

    vertices = np.array([[(0, img_height),
                          (img_width, img_height),
                          (img_width, img_height * curtain_height),
                          (0, img_height * curtain_height)]], dtype=np.int32)

    return region_of_interest(img, vertices)


def trim_to_trapeze(img):
    height = img.shape[0]
    width = img.shape[1]
    horizon = height * horizon_relative_height
    vertices = np.array([[(width / 2, horizon), (0, height), (width, height)]], dtype=np.int32)
    return curtain(region_of_interest(img, vertices), trapeze_top)


def prolong_line(line, img_height, img_weight):
    # Just some linear algebra to compute arguments of the line
    # and then use them to stretch line on the full screen
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[0][2]
    y2 = line[0][3]

    if x1 == x2:
        return np.array([[x1, 0, x1, img_height]], np.int32)

    k = (y1 - y2) / (x1 - x2)
    b = (y1 + y2 - k * (x1 + x2)) / 2

    fx1 = 0
    fy1 = b
    fx2 = img_weight
    fy2 = k * fx2 + b

    result = np.array([[fx1, fy1, fx2, fy2]], np.int32)
    return result


def get_line_angle(line):
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[0][2]
    y2 = line[0][3]

    angle = np.pi / 2
    if x1 != x2:
        angle = math.atan((y1 - y2) / (x1 - x2))

    return angle


def is_lane_line(line):
    # Acutally, there are a lot of lines on the sample videos.
    # For example - car's hood in the challenge video
    # But lane lines should be close to vertical
    # so filter them
    return abs(get_line_angle(line)) > lane_angle_limit


def compute_average_line(lines):
    if len(lines) == 0:
        return None

    max_x = 0

    k_sum = 0
    b_sum = 0

    for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]

        max_x = max(max_x, x1, x2)

        k = (y1 - y2) / (x1 - x2)
        b = (y1 + y2 - k * (x1 + x2)) / 2

        k_sum += k
        b_sum += b

    final_k = k_sum / len(lines)
    final_b = b_sum / len(lines)

    fx1 = 0
    fy1 = final_k * fx1 + final_b
    fx2 = max_x
    fy2 = final_k * fx2 + final_b

    return np.array([[fx1, fy1, fx2, fy2]], np.int32)


def get_left_right_lanes(lines):
    # left and right lanes can be distinguished by their slope
    left_lines = []
    right_lines = []

    for line in lines:
        angle = get_line_angle(line)
        if angle < 0:
            left_lines.append(line)
        if angle > 0:
            right_lines.append(line)

    left = compute_average_line(left_lines)
    right = compute_average_line(right_lines)

    return left, right


def detect_lanes(img):
    img_width = img.shape[1]
    img_height = img.shape[0]

    result = better_grayscale(img)
    result = gaussian_blur(result, gaussian_blur_kernel)
    result = canny(result, canny_low_threshold, canny_high_threshold)
    result = trim_to_trapeze(result)

    lines = cv2.HoughLinesP(result, rho, theta, threshold, np.array([]), minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    lines = filter(is_lane_line, lines)
    lines = map(lambda line: prolong_line(line, img_height, img_width), lines)
    return get_left_right_lanes(lines)


def draw_lanes_on_image(lanes, image):
    lanes = filter(lambda lane: lane is not None, lanes)
    line_img = np.zeros(image.shape, dtype=np.uint8)

    draw_lines(line_img, lanes, thickness=10)

    line_img = curtain(line_img, trapeze_display_top)
    return weighted_img(line_img, image)


def test_images():
    for img_name in os.listdir("my_test_images"):
        img = mpimg.imread('my_test_images/' + img_name)
        left, right = detect_lanes(img)

        lanes = [left, right]

        result = draw_lanes_on_image(lanes, img)

        plt.figure(figsize=(20, 20))
        plt.imshow(result, cmap='gray')


test_images()