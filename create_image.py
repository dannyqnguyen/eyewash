import argparse
import cv2
from eyewash.image_utils import read_and_return_mod_image, pad_to_match_im_dim
from os.path import isdir, splitext, join, basename
from os import mkdir
from shutil import rmtree, copyfile, move
import argparse
import os
import tensorflow as tf
from dcgan.model import DCGAN
import subprocess
import glob


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("jpg_path_in", help="path to input jpg file", type=str)
parser.add_argument("output_dir", help="dir to save outputs", type=str)
parser.add_argument("--checkpoint_dir", help="dir to load gan checkpoint ", type=str, default='checkpoint')
parser.add_argument("--use_gan", help="output a detection mask and use a GAN to fill it masked portion", type=str2bool,
                    const=True, default=False, nargs='?')
args = parser.parse_args()

#GAN is trained on 128x128 img size
IMGSIZE = 128
ALIGN_IN_DIR = './data/alignment/'
ALIGN_OUT_DIR = './data/alignment_out/'

if isdir(args.output_dir):
    rmtree(args.output_dir)

mkdir(args.output_dir)
filename_split = splitext(basename(args.jpg_path_in))

if args.use_gan:
    img_copy_path = join(ALIGN_IN_DIR, basename(args.jpg_path_in))
    copyfile(args.jpg_path_in, img_copy_path)
    # call face alignment for gan processing
    result = subprocess.call([
        'python',
        './openface/util/align-dlib.py',
        ALIGN_IN_DIR,
        'align',
        'innerEyesAndBottomLip',
        ALIGN_OUT_DIR,
        '--size',
        str(IMGSIZE)])
    align_img_copy_path = join(ALIGN_OUT_DIR, filename_split[0] +'.png')
    append_name = '_aligned'
    img_aligned_path = join(args.output_dir, filename_split[0] + append_name + '.png')
    #move aligned input to output folder
    move(align_img_copy_path, img_aligned_path)
    img = cv2.imread(img_aligned_path)
    #cleanup intermediate directory and file
    os.remove(img_copy_path)
    rmtree(ALIGN_OUT_DIR)
else:
    img = cv2.imread(args.jpg_path_in)

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
    gan_args = argparse.Namespace(
        approach='adam',
        lr = .01,
        beta1 = 0.9,
        beta2 = 0.999,
        eps = 1e-8,
        hmcBeta = 0.2,
        hmcEps = 0.001,
        hmcL = 100,
        hmcAnneal = 1,
        nIter = 3000,
        imgSize = IMGSIZE,
        lam = 0.1,
        checkpointDir = args.checkpoint_dir,
        outDir = args.output_dir,
        outInterval = 200,
        maskType='custom',
        centerScale = 0.25,
        imgs = [img_aligned_path]
    )

    assert(os.path.exists(gan_args.checkpointDir))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        dcgan = DCGAN(sess, image_size=gan_args.imgSize,
                      batch_size=min(64, len(gan_args.imgs)),
                      checkpoint_dir=gan_args.checkpointDir, lam=gan_args.lam)
        dcgan.complete(gan_args, img_out)

    gan_out_dir = join(args.output_dir, 'completed')
    list_of_gan_files = glob.glob(gan_out_dir + '/*')
    latest_gan_file = max(list_of_gan_files, key = os.path.getctime)
    gan_filepath = join(args.output_dir, filename_split[0] + '_gan' + '.png')
    copyfile(latest_gan_file, gan_filepath)
    output_filepath = join(args.output_dir, filename_split[0] + '_out' + '.jpg')
    gan_pad_filepath = join(args.output_dir, filename_split[0] + '_gan_pad' + '.jpg')

    im = cv2.imread(gan_filepath)
    matching_im = cv2.imread(args.jpg_path_in)
    new_im = pad_to_match_im_dim(im, matching_im)
    cv2.imwrite(gan_pad_filepath, new_im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    result = subprocess.call([
        'python',
        './FaceSwap/main.py',
        '--src',
        gan_pad_filepath,
        '--dst',
        args.jpg_path_in,
        '--out',
        output_filepath])

