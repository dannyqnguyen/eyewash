from os import listdir, rename, mkdir
from os.path import isfile, join, isdir, split
from shutil import rmtree
import cv2
import numpy as np

# Load the Haar cascades
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

# Global Variables
FACE_KERNEL_SIZE = 1.3
FACE_NUM_NEIGHBORS = 5
EYE_KERNEL_SIZE = 1.1
EYE_NUM_NEIGHBORS = 3
# KERNEL_SIZE is the size of the image reduced when applying the detection
# NUM_NEIGHBORS is the num of neighbors after which we accept that is a face



# Define function that will do detection
def read_and_return_mod_image(img, create_mask=False):
    """ Input = image
        Output = modified image
    """
    def _process_eyes(roi_gray_image, roi_color, roi_color_out):

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

    def _process_blue(roi_gray_image, roi_color, roi_color_out):


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
            _process_eyes(roi_gray_image, roi_color, roi_color_out)
            #_process_blue(roi_gray_image, roi_color, roi_color_out)

    else:
        _process_eyes(gray_image, img, img_out)
        #_process_blue(gray_image, img, img_out)
    if create_mask:
        img_out[img_out==1]=2
        img_out[img_out==0]=1
        img_out[img_out==2]=0
    return img_out

def show_img(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)


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

