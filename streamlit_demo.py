import os
import subprocess
import argparse
from os.path import isdir, splitext, join, basename
from os import mkdir, stat
from os import stat as os_stat
import stat
import errno
import glob
from shutil import rmtree, copyfile, move
import argparse

import tensorflow as tf
import cv2
import streamlit as st

from dcgan.model import DCGAN
from eyewash.image_utils import read_and_return_mod_image, pad_to_match_im_dim


def complete_callback(img, dims, name):
    #:st.write(type(img))
    #st.write(dims)
    st.write(name)
    st.image(name, clamp=False)

def show_header(name, avatar_image_url, **links):
    links = ' | '.join('[%s](%s)' % (key, url) for key, url in links.items())
    st.write("""
        <img src="%s" style="border-radius:50%%;height:100px;vertical-align:text-bottom;padding-bottom:10px"/>
        <span style="display:inline-block;padding-left:10px;padding-bottom:20px;font-size:3rem;vertical-align:bottom">%s</span>

        %s
    """ % (avatar_image_url, name, links))

show_header('Danny Nguyen', 'https://media.licdn.com/dms/image/C5603AQEv9IViG0KoyQ/profile-displayphoto-shrink_800_800/0?e=1569456000&v=beta&t=46yBuKaKQtrSdLer7jJpgcIOMV2iLjCjgNhynUJY54M',
    github='https://github.com/dannyqnguyen/eyewash',
    linkedin='https://www.linkedin.com/in/dannyqnguyen/',
    resume='https://www.dropbox.com/s/ods9gvnj19nc6jx/Danny_Nguyen_FinalResume.pdf?dl=1'
)

st.write(
    """
    ## Eyewash
    With the advent of powerful cameras within smartphones, photography is becoming more commonplace. With this increased usage, more people will find themselves with an amazing picture and be disappointed due to it being ruined by unwanted blemishes. However, current ways to remove blemishes such as redeye require manual work to apply the appropriate effect.
    
    Eyewash is a package to automatically remove blemishes from portrait photos. Users no long have to manually select pixels as well as create more realistic fixes to the blemishes rather than filling in with a specific color. The implementation uses OpenCV HAAR cascades to detect redeye and remove the affected pixels through image infilling with Deep Convolutional Generative Adversarial Networks (DCGANs).  This functionality is extended by selecting various facial features to  detect and replace.

    """
         )
status = st.empty()
status.error("Please select an image to run and which facial features to replace. After setting the desired number of iterations, press the 'Run GAN' button to generate images.")

ex_img1 = cv2.cvtColor(cv2.imread('./data/redeye_samples/14.jpg'), cv2.COLOR_BGR2RGB)
ex_img2 = cv2.cvtColor(cv2.imread('./data/redeye_samples/2.jpg'), cv2.COLOR_BGR2RGB)
ex_img3 = cv2.cvtColor(cv2.imread('./data/redeye_samples/24.jpg'), cv2.COLOR_BGR2RGB)

st.image([ex_img1, ex_img2, ex_img3], caption= ['1','2','3'], width = 200)

run_images = [
    './data/redeye_samples/14.jpg',
    './data/redeye_samples/2.jpg',
    './data/redeye_samples/24.jpg'
    ]
image_index = st.selectbox('Selected Image:', ['1', '2', '3'])
run_image = run_images[image_index]

st.write('<small>Facial Features to Replace:</small>')
facial_features = [
    "mouth",
    "left_eye",
    "right_eye",
    "nose",
]

selected_features = []
for feature in facial_features:
    if st.checkbox(feature):
        selected_features.append(feature)

def str2bool(v):
    """
    Helper function to deal with booleans using argparse
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


args = argparse.Namespace(
    jpg_path_in = run_image,
    output_dir = './output_dir/',
    checkpoint_dir = 'checkpoint',
    use_gan = True,
    landmark_list = selected_features
)



"""

parser = argparse.ArgumentParser()
parser.add_argument("jpg_path_in", help="path to input jpg file", type=str)
parser.add_argument("output_dir", help="dir to save outputs", type=str)
parser.add_argument("--checkpoint_dir", help="dir to load gan checkpoint ", type=str, default='checkpoint')
parser.add_argument("--use_gan", help="output a detection mask and use a GAN to fill it masked portion", type=str2bool,
                    const=True, default=False, nargs='?')
parser.add_argument('-l','--landmark_list', nargs='*', help='list of face landmarks to segment out', required=False)
args = parser.parse_args()
"""
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
    if os.path.exists(ALIGN_OUT_DIR):
        os.system('rf -rfv %s' % ALIGN_OUT_DIR)
    move(align_img_copy_path, img_aligned_path)
    img = cv2.imread(img_aligned_path)
    #cleanup intermediate directory and file
    os.remove(img_copy_path)
    rmtree(ALIGN_OUT_DIR)
else:
    img = cv2.imread(args.jpg_path_in)

img_out = read_and_return_mod_image(img, args.use_gan, args.landmark_list)

if args.use_gan:
    mask = img_out.copy()
    append_name = '_mask'
    img_out[img_out==1]=255
else:
    append_name = '_out'

img_out_path = join(args.output_dir, filename_split[0] + append_name + filename_split[1])
cv2.imwrite(img_out_path, img_out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

if args.use_gan:
    nIter = st.slider('number of iterations', 1000, 200, 5000, 100)
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
        nIter = nIter,
        imgSize = IMGSIZE,
        lam = 0.1,
        checkpointDir = args.checkpoint_dir,
        outDir = args.output_dir,
        outInterval = 200,
        maskType='custom',
        centerScale = 0.25,
        imgs = [img_aligned_path]
    )

    try:
        dir_exists = stat.S_ISDIR(os_stat(gan_args.checkpointDir).st_mode)
    except OSError, e:
        if e.errno == errno.ENOENT:
            print("checkpointDir: %s does not exists".format(gan_args.checkpointDir))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    st.text('About to create session')
    if st.button('Run GAN'):
        waiting_message = st.warning('Computing image, please wait 30 seconds')

        @st.cache(ignore_hash=True)
        def create_dc_gan(config):
            tf_session = tf.Session(config=config)
            dcgan = DCGAN(tf_session, image_size=gan_args.imgSize,
                        batch_size=min(64, len(gan_args.imgs)),
                        checkpoint_dir=gan_args.checkpointDir, lam=gan_args.lam)
            return tf_session, dcgan

        tf_session, dcgan = create_dc_gan(config)
        with tf_session.as_default():
            dcgan.complete(gan_args, img_out, callback=complete_callback)

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

        waiting_message.success('done')

        result = subprocess.call([
            'python',
            './FaceSwap/main.py',
            '--src',
            gan_pad_filepath,
            '--dst',
            args.jpg_path_in,
            '--out',
            output_filepath,
            '--no_debug_window'
            #'True'
            ]
        )

        st.image(output_filepath)
