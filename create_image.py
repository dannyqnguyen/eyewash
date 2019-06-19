import argparse
import cv2
from image_utils import read_and_return_mod_image, show_img, resize_to_square_and_pad
from os.path import isdir, splitext, join
from os import mkdir
from shutil import rmtree
import argparse
import os
import tensorflow as tf

from dcgan.model import DCGAN

parser = argparse.ArgumentParser()
parser.add_argument("jpg_path_in", help="path to input jpg file", type=str)
parser.add_argument("output_dir", help="dir to save outputs", type=str)
parser.add_argument("--use_gan", help="output a detection mask and use a GAN to fill it masked portion", type=bool)
args = parser.parse_args()

IMGSIZE = 64

if isdir(args.output_dir):
    rmtree(args.output_dir)

mkdir(args.output_dir)
filename_split = splitext(args.jpg_path_in)

img = cv2.imread(args.jpg_path_in)
if args.use_gan:
    img = resize_to_square_and_pad(img, IMGSIZE)
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

if args.use_gan:
    args = argparse.Namespace(
        approach='adam',
        lr = .01,
        beta1 = 0.9,
        beta2 = 0.999,
        eps = 1e-8,
        hmcBeta = 0.2,
        hmcEps = 0.001,
        hmcL = 100,
        hmcAnneal = 1,
        nIter = 20000,
        imgSize = IMGSIZE,
        lam = 0.1,
        checkpointDir = 'model_archive/checkpoint_dcgan64',
        outDir = args.output_dir,
        outInterval = 200,
        maskType='custom',
        centerScale = 0.25,
        imgs = [img_resized_path]
    )

    assert(os.path.exists(args.checkpointDir))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        dcgan = DCGAN(sess, image_size=args.imgSize,
                      batch_size=min(64, len(args.imgs)),
                      checkpoint_dir=args.checkpointDir, lam=args.lam)
        dcgan.complete(args, img_out)

