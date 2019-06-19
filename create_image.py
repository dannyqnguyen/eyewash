import argparse
import cv2
from image_utils import read_and_return_mod_image, show_img, resize_to_square_and_pad
from os.path import isdir, splitext, join
from os import mkdir
from shutil import rmtree

parser = argparse.ArgumentParser()
parser.add_argument("jpg_path_in", help="path to input jpg file", type=str)
parser.add_argument("output_dir", help="dir to save outputs", type=str)
parser.add_argument("-use_gan", help="output a detection mask and use a GAN to fill it masked portion", type=bool)
args = parser.parse_args()

if isdir(args.output_dir):
    rmtree(args.output_dir)

mkdir(args.output_dir)
filename_split = splitext(args.jpg_path_in)

img = cv2.imread(args.jpg_path_in)
if args.use_gan:
    img = resize_to_square_and_pad(img, 128)
    append_name = '_resized'
    img_resized_path = join(args.output_dir, filename_split[0] + append_name + filename_split[1])
    cv2.imwrite(img_resized_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

img_out = read_and_return_mod_image(img, args.use_gan)

if args.use_gan:
    mask = img_out.copy()
    append_name = '_mask'
    img_out[img_out==1]=255
else:
    append_name = '_out'

img_out_path = join(args.output_dir, filename_split[0] + append_name + filename_split[1])
cv2.imwrite(img_out_path, img_out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

