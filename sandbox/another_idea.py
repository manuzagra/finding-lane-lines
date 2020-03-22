import numpy as np
import math


def compare_lines(l1, l2, mode='p'):
    """
    Compare 2 lines.

    Returns true if they are similar, false otherwise
    """
    if mode == 'p':
        dist_error = 50  # pixels, max distance between points
        if (math.sqrt((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2) < dist_error and \
             math.sqrt((l1[2] - l2[2]) ** 2 + (l1[3] - l2[3]) ** 2) < dist_error) or \
            (math.sqrt((l1[0] - l2[2]) ** 2 + (l1[1] - l2[3]) ** 2) < dist_error and \
             math.sqrt((l1[2] - l2[0]) ** 2 + (l1[3] - l2[1]) ** 2) < dist_error):
            return True
        return False
    elif mode == 'm':
        m_error = 5 * math.pi / 180  # max difference between slopes
        b_error = 2  # max difference between intersection with axes
        if abs(l1[4] - l2[4]) < m_error: # and abs(l1[5] - l2[5]) < b_error:
            return True
        return False


def combine_lines(lines, mode='p'):
    """
    Combine lines. Lines must be in the form y=m*x+b  ->  (m, b) or in the form (x0, y0, x1, y1)

    Returns a line that has the average parameters
    """
    if mode == 'p' or mode == 'm':
        n = lines.shape[0]
        return (lines['x1'].sum()/n, lines['y1'].sum()/n, lines['x2'].sum()/n, lines['y2'].sum()/n, lines['m'].sum()/n, lines['b'].sum()/n)
    elif mode == 'm':
        points = []
        for x1,y1,x2,y2,m,b in lines:
            points.append([x1,y1])
            points.append([x2,y2])
        p = np.asarray(points)
        p.sort()
        return (p[0,0], p[0,1], p[-1,0], p[-1,1], 0, 0) # TODO calculate the slope and the b



def combine_close_lines(lines):
    """
    `lines` should be the output of a HoughLinesP function.

    Returns only valid lines.
    """
    # TODO write a nice definition

    # A place to save final lines
    new_lines = np.copy(lines)

    # I have to use a while because I need to modify the index in the loop
    index = 0
    # index with valuable information
    filled = []
    while index < lines.shape[0]:
        # save the index before the loop
        index_pre = index
        # Compare lines till the moment they are different
        while index < lines.shape[0]-1 and compare_lines(lines[index], lines[index + 1]):
            index += 1
        # combining the lines
        if index_pre != index:
            # Combine the lines and save the combined line
            l = combine_lines(lines[index_pre:index+1])
            new_lines[index] = l
        filled.append(index)
        # update the index
        index += 1
    new_lines = new_lines[filled]

    return new_lines


def line_redundant_definition(line):
    m = abs(float(line[0, 3] - line[0, 1]) / (line[0, 2] - line[0, 0]))
    b = -m * line[0, 0] + line[0, 1]
    return line[0, 0], line[0, 1], line[0, 2], line[0, 3], m, b


def filter_by_slope(line):
    # TODO escribir definicion
    # TODO volver a poner en radianes
    deviation_from_vertical = 70#*math.pi/180
    vertical = 90#*math.pi/180
    slope = math.atan2(line[3]-line[1], line[2]-line[0]) *180/math.pi
    s = slope*180/math.pi
    if vertical-deviation_from_vertical < abs(slope) < vertical+deviation_from_vertical:
        return True
    return False


def detect_line_lanes(lines_p):
    """
    `lines` should be the output of a HoughLinesP function.

    Returns only valid lines.
    """
    # TODO create a nice definition

    # Create an array with both configurations of the lines
    lines = np.zeros(shape=(lines_p.shape[0]),
                     dtype=[('x1', np.int16), ('y1', np.int16), ('x2', np.int16), ('y2', np.int16),
                            ('m', np.float16), ('b', np.float16)])

    # Transform each line and filter by slope
    filled = []
    for index in range(0, lines_p.shape[0]):
        l = line_redundant_definition(lines_p[index])
        if filter_by_slope(l):
            lines[index] = l
            filled.append(index)
    lines = lines[filled]

    # combine similar lines by their definition in points
    #lines2[lines2[:, 0, 1].argsort()]
    lines.sort(order='x1')
    lines = combine_close_lines(lines)

    # combine similar lines by their definition in slope
    #lines2[lines2[:, 0, 1].argsort()]
    lines.sort(order='m')
    lines = combine_close_lines(lines)


    l = np.empty(shape=(lines.shape[0], lines_p.shape[1], lines_p.shape[2]), dtype=np.int16)
    for index in range(lines.shape[0]):
        for i in range(4):
            l[index,0,i] = lines[index][i]
    return l
