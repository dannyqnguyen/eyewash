from os import listdir, rename, mkdir
from os.path import isfile, join, isdir, split
from collections import OrderedDict
from shutil import rmtree
import cv2
import numpy as np
from imutils import face_utils
import dlib
import imutils



# Load the Haar cascades
face_cascade = cv2.CascadeClassifier('./eyewash/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('./eyewash/haarcascade_eye.xml')

# Load dlib face detectyors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./eyewash/shape_predictor_68_face_landmarks.dat')



# Global Variables
FACE_KERNEL_SIZE = 1.3
FACE_NUM_NEIGHBORS = 5
EYE_KERNEL_SIZE = 1.1
EYE_NUM_NEIGHBORS = 3
# KERNEL_SIZE is the size of the image reduced when applying the detection
# NUM_NEIGHBORS is the num of neighbors after which we accept that is a face

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

# Define function that will do detection
def read_and_return_mod_image(img, create_mask=False, landmark_list =[]):
    """ 
    Main function to read an image and return a modified img. If create_mask is False, 
    modification will be to fix red pixels with black. If create_mask is True, then return
    segmented mask of blemishes.

    :param img: cv2 img
    :param create_mask: optional boolean flag to switch to produce an image mask of segmented blemishes 
    :return: cv2 img
    """

    # Now get the tuples that detect the faces using the above cascade
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if create_mask:
        img_out = np.zeros(img.shape)
    else:
        img_out = img.copy()
    faces = face_cascade.detectMultiScale(gray_image, FACE_KERNEL_SIZE, FACE_NUM_NEIGHBORS)

    # faces are the tuples of the 4 numbers
    # x,y => upperleft corner coordinates of face
    # width(w) of rectangle in the face
    # height(h) of rectangle in the face

    # Now iterate over the faces and detect eyes
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            #cv2.rectangle(img_out, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Arguements => image, top-left coordinates, bottomright coordinates, color, rectangle border thickness

            # we now need two region of interests(ROI) grey and color for eyes one to detect and another to draw rectangle
            roi_gray_image = gray_image[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            roi_color_out = img_out[y:y + h, x:x + w]
            # Detect eyes now
            process_eyes(roi_gray_image, roi_color, roi_color_out, create_mask)
            #process_blue(roi_gray_image, roi_color, roi_color_out, create_mask)

    else:
        process_eyes(gray_image, img, img_out, create_mask)
        #process_blue(gray_image, img, img_out, create_mask)
    if create_mask and landmark_list:
        process_face_landmarks(gray_image, img_out, landmark_list)
    if create_mask:
        img_out[img_out==1]=2
        img_out[img_out==0]=1
        img_out[img_out==2]=0
    return img_out

def show_img(img):
    """
    Helper function to print images to screen. Useful for debugging

    :param img: cv2 img
    :return: None
    """
    cv2.imshow("img", img)
    cv2.waitKey(0)

def process_blue(roi_gray_image, roi_color, roi_color_out):
    """
    Image detection function to detect and return segmented mask of blue pixels in roi_color_out

    :param roi_gray_image: grayscale version cv2 image used for detection.
    :param roi_color: color version of cv2 image used for detection
    :param roi_color_out: color version of cv2 image for output
    :param create_mask: flag to create segmented mask
    :return: None (roi_color_out will be modified)

    """


    # Split image into 3 channels
    b = roi_color[:, :, 0]
    g = roi_color[:, :, 1]
    r = roi_color[:, :, 2]


    # Add the green and blue channels.
    rg = cv2.min(r, g)
    rg_threshold = 2.1*rg

    # Simple red eye detector
    mask = (b > rg_threshold)
    mask = mask.astype(np.uint8) * 255
    mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=1, borderType=1, borderValue=1)
    mask = mask / 255
    mask = mask[:, :, np.newaxis]


    if create_mask:
        # Copy the mask to the roi
        np.copyto(roi_color_out, mask.astype(np.uint8))




def process_eyes(roi_gray_image, roi_color, roi_color_out, create_mask):
    """
    Image detection function to detect and return segmented mask of blue pixels in roi_color_out

    :param roi_gray_image: grayscale version cv2 image used for detection.
    :param roi_color: color version of cv2 image used for detection
    :param roi_color_out: color version of cv2 image for output
    :param create_mask: flag to create segmented mask
    :return: None (roi_color_out will be modified)

    """

    eyes = eyes_cascade.detectMultiScale(roi_gray_image, EYE_KERNEL_SIZE,
                                            EYE_NUM_NEIGHBORS)
    for (ex, ey, ew, eh) in eyes:
        #cv2.rectangle(roi_color_out, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        eye_out = roi_color_out[ey:ey + eh, ex:ex + ew]
        eye = roi_color[ey:ey + eh, ex:ex + ew]

        # Split eye image into 3 channels
        b = eye[:, :, 0]
        g = eye[:, :, 1]
        r = eye[:, :, 2]

        # Add the green and blue channels.
        bg = cv2.max(b, g)
        bg_threshold = 1.65*bg

        # Simple red eye detector
        mask = (r > bg_threshold)


        mean = bg / 2
        mask = mask[:, :, np.newaxis]
        mean = mean[:, :, np.newaxis]
        mean = mean.astype(np.uint8)

        if create_mask:
            # Copy the mask to the roi
            np.copyto(eye_out, mask.astype(np.uint8))
        else:
            # Copy the mean image to the output image.
            np.copyto(eye_out, mean, where=mask)

def process_face_landmarks(gray, color_out, landmark_list):
    """
    Image detection function to detect and return segmented mask using landmark_list

    :param gray: grayscale version cv2 image used for detection.
    :param roi_color_out: color version of cv2 image for output
    :param landmark_list: list of landmarks to segment
    :return: None (color_out will be modified)

    """

    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, dlib.rectangle(0,0,gray.shape[0],gray.shape[1]))
    shape = face_utils.shape_to_np(shape)

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]

        # check if are supposed to draw the jawline
        if name == "jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            pass


        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        elif name in landmark_list:
            hull = cv2.convexHull(pts)
            dilation_scale = 1.3
            M = cv2.moments(hull)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            hull_scaled = hull.copy()
            for point in hull_scaled:
                point[0][0] = point[0][0] - cX
                point[0][1] = point[0][1] - cY
            hull_scaled = (hull_scaled * dilation_scale).astype('int32')
            for point in hull_scaled:
                point[0][0] = point[0][0] + cX
                point[0][1] = point[0][1] + cY




            cv2.fillPoly(color_out, pts=[hull_scaled], color=(1,1,1))


"""
void scaleContour(const std::vector<cv::Point>& src, std::vector<cv::Point>& dst, float scale)
{
    cv::Rect rct = cv::boundingRect(src);

    std::vector<cv::Point> dc_contour;
    cv::Point rct_offset(-rct.tl().x, -rct.tl().y);
    contourOffset(src, dc_contour, rct_offset);

    std::vector<cv::Point> dc_contour_scale(dc_contour.size());

    for (int i = 0; i < dc_contour.size(); i++)
        dc_contour_scale[i] = dc_contour[i] * scale;

    cv::Rect rct_scale = cv::boundingRect(dc_contour_scale);

    cv::Point offset((rct.width - rct_scale.width) / 2, (rct.height - rct_scale.height) / 2);
    offset -= rct_offset;
    dst.clear();
    dst.resize(dc_contour_scale.size());
    for (int i = 0; i < dc_contour_scale.size(); i++)
        dst[i] = dc_contour_scale[i] + offset;
    }

"""




def resize_to_square_and_pad_dir(desired_size, raw_img_dir, propcessed_img_dir):
    """
    This function will look for img files in raw_img_dir. It will then create a new output propcessed_img_dir. For each
    file, this function will resize and pad to desired_size.

    :param desired_size: desired pixel size of image width and height
    :param raw_img_dir: string abs path to image path
    :param propcessed_img_dir: string abs path to create directory to put processed file. If folder exists, it will be
    deleted and a new one will be created
    :return: None
    """


    raw_files = [join(raw_img_dir, f) for f in listdir(raw_img_dir) if isfile(join(raw_img_dir, f))]

    if isdir(propcessed_img_dir):
        rmtree(propcessed_img_dir)

    mkdir(propcessed_img_dir)
    for im_pth in raw_files:
        im = cv2.imread(im_pth)
        new_im = resize_to_square_and_pad(im, desired_size)

        head, tail = split(im_pth)
        new_pth = join(propcessed_img_dir, tail)
        cv2.imwrite(new_pth, new_im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def resize_to_square_and_pad(im, desired_size):
    """
    This function will take an cv2 img resize and pad to desired_size (same widht and height).

    :param im: cv2 img
    :param desired_size: desired pixel size of image width and height
    :return: new_im (cv2 img)
    """

    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im

def pad_to_match_im_dim(im, matching_im):
    """
    This function will take an cv2 img and pad to match dimensions of another cv2 img.

    :param im: cv2 img
    :param matching_im: cv2 img to find dimensions and pad to desired dimensions
    :return: new_im (cv2 img)
    """


    old_size =im.shape  # old_size is in (height, width, channel) format
    new_size = matching_im.shape

    delta_w = new_size[1] - old_size[1]
    delta_h = new_size[0] - old_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords