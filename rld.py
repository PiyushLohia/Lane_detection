# Project: Lane Lines Detection using OpenCV

# Importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob
from moviepy.editor import VideoFileClip

def list_images(images, cols=2, rows=5, cmap=None):
    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

test_images = [mpimg.imread(img) for img in glob.glob('test_images/*.jpg')]
list_images(test_images)

def RGB_color_selection(image):
    lower_threshold = np.uint8([200, 200, 200])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower_threshold, upper_threshold)
    lower_threshold = np.uint8([175, 175, 0])
    upper_threshold = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower_threshold, upper_threshold)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def HSV_color_selection(image):
    converted = convert_hsv(image)
    white_mask = cv2.inRange(converted, np.uint8([0, 0, 210]), np.uint8([255, 30, 255]))
    yellow_mask = cv2.inRange(converted, np.uint8([18, 80, 80]), np.uint8([30, 255, 255]))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

def convert_hsl(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def HSL_color_selection(image):
    converted = convert_hsl(image)
    white_mask = cv2.inRange(converted, np.uint8([0, 200, 0]), np.uint8([255, 255, 255]))
    yellow_mask = cv2.inRange(converted, np.uint8([10, 0, 100]), np.uint8([40, 255, 255]))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

list_images(list(map(HSL_color_selection, test_images)))
color_selected_images = list(map(HSL_color_selection, test_images))

def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

gray_images = list(map(gray_scale, color_selected_images))
list_images(gray_images)

def gaussian_smoothing(image, kernel_size=13):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

blur_images = list(map(gaussian_smoothing, gray_images))
list_images(blur_images)

def canny_detector(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

edge_detected_images = list(map(canny_detector, blur_images))
list_images(edge_detected_images)

def region_selection(image):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(image, mask)

masked_image = list(map(region_selection, edge_detected_images))
list_images(masked_image)

def hough_transform(image):
    rho = 1
    theta = np.pi/180
    threshold = 20
    minLineLength = 20
    maxLineGap = 300
    return cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

hough_lines = list(map(hough_transform, masked_image))

def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    image = np.copy(image)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

line_images = [draw_lines(img, lines) for img, lines in zip(test_images, hough_lines)]
list_images(line_images)

def average_slope_intercept(lines):
    left_lines, left_weights = [], []
    right_lines, right_weights = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)
    left_lane  = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return ((x1, int(y1)), (x2, int(y2)))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    return pixel_points(y1, y2, left_lane), pixel_points(y1, y2, right_lane)

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

lane_images = []
for image, lines in zip(test_images, hough_lines):
    lane_images.append(draw_lane_lines(image, lane_lines(image, lines)))

list_images(lane_images)

# Video Stream Processing
def frame_processor(image):
    color_select = HSL_color_selection(image)
    gray = gray_scale(color_select)
    smooth = gaussian_smoothing(gray)
    edges = canny_detector(smooth)
    region = region_selection(edges)
    hough = hough_transform(region)
    result = draw_lane_lines(image, lane_lines(image, hough))
    return result

def process_video(test_video, output_video):
    input_video = VideoFileClip(os.path.join('test_videos', test_video), audio=False)
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(os.path.join('output_videos', output_video), audio=False)

# Example usage:
process_video('solidWhiteRight.mp4', 'solidWhiteRight_output.mp4')
process_video('solidYellowLeft.mp4', 'solidYellowLeft_output.mp4')
process_video('challenge.mp4', 'challenge_output.mp4')
