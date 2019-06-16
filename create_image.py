import argparse
import cv2
from image_utils import read_and_return_mod_image

parser = argparse.ArgumentParser()
parser.add_argument("jpg_path_in", help="path to input jpg file", type=str)
parser.add_argument("jpg_path_out", help="path to save output jpg file", type=str)
args = parser.parse_args()

img = cv2.imread(args.jpg_path_in)
img_out = read_and_return_mod_image(img)
cv2.imwrite(args.jpg_path_out, img_out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

