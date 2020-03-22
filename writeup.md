# **Finding Lane Lines on the Road**


The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---
### Pipeline

My pipeline consisted of 8 steps:
1. Convert the image to gray scale
2. Smooth the image convoluting it with a gaussian kernel
3. Calculate the edges using canny function
4. Calculate the Hough transform, look for lines
5. Filter and process the lines
6. Create an image with the lines
7. Create the vertices of the mask for the ROI and mask the image with the lines
8. Combine the initial image and the image with the lines

##### 1. Convert the image to gray scale
Using the function cvtColor from OpenCV and specifing *COLOR_RGB2GRAY* as the color space conversion code.
```python
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```


##### 2. Smooth the image convoluting it with a gaussian kernel
The function GaussianBlur from OpenCV does the convolution between the input image and a gaussian kernel of the specified size. In this case, the kernel is a square one.
```python
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```

##### 3. Calculate the edges using canny function
The function Canny from OpenCV does the calculation the edges. It first does a noise reduction, then it calculates the intensity gradient of the image and eliminate some points using Non-maximum Suppression. At the end it pass an hysteresis thresholding
using the given thresholds throw the image.
```python
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
```


##### 4. Calculate the Hough transform, look for lines
The function HoughLinesP from OpenCV look for lines in an image (usually the output of an edge detection algorithm). It transform the image to the Hough space, make a grid in that space and look for intersections of lines in that space. The cells that have enough crosses of lines are converted back and returned as lines. The paremeters of this function must be tunned carefuly.
```python
lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
```


##### 5. Filter and process the lines
The output of *HoughLinesP* functions are all kind of lines. First almost horizontal lines are filtered, then lines are classified as left and right lane lines. The lines of each side are combined using an average weighted by the length of the segment. Finally it returns 2 segments from the lines using the y coordinate of the intersection and the minimum y coordinate in all the lines to cut them.
```python
class CombinedLine:
    """
    This class will handle all the operations with lines after they have been classified as left or right.
    """
    ...

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
```


##### 6. Create an image with the lines
Create a black image and plot the line segments in it.
```python
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

def create_img_lines(lines, shape):
    """
    Creates a black image with the lines plotted in it.
    :param lines: np.array containing the lines to plot. Should be the same format than the output of a HoughLinesP function.
    :param shape: (height, width) of the output image.
    :return: np.array containing the image with the lines plotted.
    """
    line_img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
```


##### 7. Create the vertices of the mask for the ROI and mask the image with the lines
Vertices is a numpy array containing the vertices of our ROI. In this case it is a trapecium. Create a mask using the defined ROI and mask the input image.
```python
vertices = np.array([ [ (img.shape[1], img.shape[0]), (0, img.shape[0]), (img.shape[1]*1/4, img.shape[0]/2), (img.shape[1]*3/4, img.shape[0]/2)] ], np.int32)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
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
```


##### 8. Combine the initial image and the image with the lines
The function addWeighted from OpenCV calculates the weighted sum of two arrays. In this case the two arrays are two images.
```python
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
```





---
### Reflection

The defined pipeline is simple and straightforward so it has some weak points:

* It is very dependent of the parameters given to the Hough transform.
* It only detects straight lines.
* It combines all the lines that are not horizontal. If the Hough transform outputs lines that are not lane lines they will influence the final solution.
* It combines lines that may not be very close to each other.
* Bad detected lines have significant influence in the output.
* The color of the lines may affect the result.

Future improvement that could be done:

* Check the distance (not necessarely euclidean distance) between lines before combine them.
* Use the length of the segments to weight the average while combining lines makes it a bit more robust to bad detected lines, but it may not be enough. The union of the output of the Hough transform with some kind of model based detection for lane lines could improve the average false positive.
* Replace the *HoughLinesP* function for another algorithm that detects parabolics paterns to be able to detect curved lane lines.
