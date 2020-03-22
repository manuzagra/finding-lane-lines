import numpy as np
import math


class CombinedLine:
    """
    This class will handle all the operations with lines after they have been classified as left or right.
    """
    def __init__(self):
        self.lines = []
        self.n_lines = 0
        self.length = 0
        self.slope = 0
        self.point_middle = [0, 0]
        self.point_bottom = [0, 0]
        self.point_top = [float('INF'), float('INF')]

    def x(self, y):
        """
        It returns the x coordinate of a point contained in the line with the y coordinated given
        :param y:
        :return: the x coordinate corresponded to the y coordinate given
        """
        m = math.tan(self.slope)
        return (y - self.point_middle[1] + m * self.point_middle[0]) / m

    def y(self, x):
        """
        It returns the y coordinate of a point contained in the line with the x coordinated given
        :param x:
        :return: the y coordinate corresponded to the x coordinate given
        """
        m = math.tan(self.slope)
        return m * x - m * self.point_middle[0] + self.point_middle[1]

    def add(self, line):
        """
        It add a line and recalculate all the parameters of the combined line
        :param line:
        :return: void
        """
        # List with all the lines
        self.lines.append(line)
        # number of lines combined
        self.n_lines += 1
        # The length of the actual line, it is used to weight the average
        length = math.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
        # Total length of all the lines combined
        self.length += length
        # Slope of the actual line
        slope = math.atan2(line[3] - line[1], line[2] - line[0])
        # Average slope of all the lines combined
        self.slope = (self.slope * (self.length - length) + slope * length) / self.length
        # Average of all the points in all the lines combined
        self.point_middle[0] = (self.point_middle[0] * 2 * (self.length - length) + (line[0] + line[2]) * length) / (2 * self.length)
        self.point_middle[1] = (self.point_middle[1] * 2 * (self.length - length) + (line[1] + line[3]) * length) / (2 * self.length)
        # Extreme points of all the lines
        if self.point_bottom[1] < line[1]:
            self.point_bottom[0] = line[0]
            self.point_bottom[1] = line[1]
        if self.point_bottom[1] < line[3]:
            self.point_bottom[0] = line[2]
            self.point_bottom[1] = line[3]
        if self.point_top[1] > line[1]:
            self.point_top[0] = line[0]
            self.point_top[1] = line[1]
        if self.point_top[1] > line[3]:
            self.point_top[0] = line[2]
            self.point_top[1] = line[3]

    def np_lines(self):
        """
        Gives np.array format to the lines stored
        :return: np.array containing all the lines
        """
        lines = np.empty(shape=(self.n_lines, 1, 4), dtype=np.int16)
        for index, line in enumerate(self.lines):
            lines[index] = line
        return lines

    @classmethod
    def intersection(cls, l1, l2):
        """
        Calculates the point of intersection between two CombinedLines
        :param l1: CombinedLines
        :param l2: CombinedLines
        :return: (x, y) of the intersection point
        """
        m1 = math.tan(l1.slope)
        m2 = math.tan(l2.slope)
        x = (m1 * l1.point_middle[0] - m2 * l2.point_middle[0] - l1.point_middle[1] + l2.point_middle[1]) / (m1 - m2)
        y = m1 * x - m1 * l1.point_middle[0] + l1.point_middle[1]
        return x, y


def create_img_lines(lines, shape):
    """

    :param lines: np.array containing the lines to plot. Should be the same format than the output of a HoughLinesP function.
    :param shape: (height, width) of the output image.
    :return: np.array containing the image with the lines plotted.
    """
    line_img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def filter_by_slope(slope):
    """
    Filter horizontal lines
    :param slope: the slope of the line
    :return: False if the line is almost horizontal, True otherwise
    """
    # The calculation is done creating a valid zone around the vertical
    deviation_from_vertical = 70*math.pi/180
    vertical = 90*math.pi/180
    # Transpose the slope to the 1 and 2 quadrants
    if slope < 0:
        slope += math.pi
        # Filter
    if vertical-deviation_from_vertical < slope < vertical+deviation_from_vertical:
        return True
    return False


def process_lines(lines):
    """
    It classifies the lines and combine them into a CombinedLine
    :param lines: np.array with all the lines detected in the image. It should be the output of a HoughLinesP function
    :return: np.array with 2 lines
    """
    lines_l = CombinedLine()
    lines_r = CombinedLine()
    for line in lines[:, 0]:
        # the slope of the line
        slope = math.atan2(line[3] - line[1], line[2] - line[0])
        # Filter almost horizontal lines
        if not filter_by_slope(slope):
            continue
        # Classifies lines in left and right lane lines and add them to the corespondent CombinedLine
        if slope > 0:
            lines_r.add(line)
        else:
            lines_l.add(line)

    # The max_y coordinate gives and approximation of the bottom of the image
    max_y = max(lines_l.point_bottom[1], lines_r.point_bottom[1])
    # Calculate the intersection, it gives an approximation of the horizon
    intersection = CombinedLine.intersection(lines_l, lines_r)
    # A parameter to cut the horizon below intersection
    p_horizon = 1.1

    # The output is created using the horizon and max_y as y coordinates and camculating the xs
    return np.array([[[lines_l.x(intersection[1]*p_horizon), intersection[1]*p_horizon, lines_l.x(max_y), max_y],
                    [lines_r.x(intersection[1]*p_horizon), intersection[1]*p_horizon, lines_r.x(max_y), max_y]]],
                    dtype=np.int16)






lines = np.array([[[159, 269, 230, 275]],
 [[649, 361, 791, 369]],
 [[700, 445, 802, 373]],
 [[711, 445, 845, 376]],
 [[712, 446, 874, 437]],
 [[733, 472, 874, 468]],
 [[177, 315, 270, 306]],
 [[200, 200, 400, 600]],
 [[179, 316, 311, 314]],
 [[182, 329, 326, 315]],
 [[251, 332, 404, 338]],
 [[269, 357, 467, 341]],
 [[408, 360, 484, 361]],
 [[511, 360, 587, 367]],
 [[743, 482, 911, 509]],
 [[844, 537, 955, 535]],
 [[879, 539, 955, 538]],
 [[880, 539, 958, 539]]])

lines2 = np.array([[[251, 482, 484, 306]],
 [[159, 539, 404, 361]],
 [[649, 316, 911, 338]],
 [[712, 445, 874, 538]],
 [[743, 472, 845, 535]],
 [[700, 446, 802, 509]],
 [[408, 360, 467, 314]],
 [[182, 537, 311, 437]],
 [[880, 360, 955, 369]],
 [[177, 269, 230, 275]],
 [[879, 357, 958, 367]],
 [[269, 315, 326, 315]],
 [[733, 361, 791, 373]],
 [[179, 539, 270, 468]],
 [[844, 332, 955, 341]],
 [[711, 445, 874, 539]],
 [[511, 329, 587, 376]]])


lines3 = np.array([[[195, 538, 488, 309]],
 [[184, 533, 380, 386]],
 [[743, 467, 858, 536]],
 [[661, 324, 959, 350]],
 [[818, 502, 883, 538]],
 [[378, 389, 477, 315]],
 [[777, 376, 854, 392]],
 [[750, 464, 884, 538]],
 [[777, 374, 856, 389]],
 [[619, 319, 668, 324]],
 [[909, 366, 959, 373]],
 [[478, 310, 607, 381]],
 [[194, 538, 234, 509]],
 [[743, 466, 863, 538]],
 [[710, 265, 768, 236]],
 [[904, 368, 959, 376]]])


detect_line_lanes(lines3)