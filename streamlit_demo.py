import os
import sys
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
import random
import threading

import tensorflow as tf
import cv2
import streamlit as st
import numpy as np
import pandas as pd

from dcgan.model import DCGAN
from eyewash.image_utils import read_and_return_mod_image, pad_to_match_im_dim

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

def main():
    st.write(
        """
        ## Eyewash
        With the advent of powerful cameras within smartphones, photography is becoming more commonplace. With this increased usage, more people will find themselves with an amazing picture and be disappointed due to it being ruined by unwanted blemishes. However, current ways to remove blemishes such as redeye require manual work to apply the appropriate effect.
        
        Eyewash is a package to automatically remove blemishes from portrait photos. Users no long have to manually select pixels as well as create more realistic fixes to the blemishes rather than filling in with a specific color. The implementation uses OpenCV HAAR cascades to detect redeye and remove the affected pixels through image infilling with Deep Convolutional Generative Adversarial Networks (DCGANs).  This functionality is extended by selecting various facial features to  detect and replace.

        """
            )
    status = st.error("""
        To use, select an image to run and which facial features to replace.
        After setting the desired number of iterations, press the
        `Run GAN` button to generate images.""")

    ex_img1 = cv2.cvtColor(cv2.imread('./data/redeye_samples/14.jpg'), cv2.COLOR_BGR2RGB)
    ex_img2 = cv2.cvtColor(cv2.imread('./data/redeye_samples/2.jpg'), cv2.COLOR_BGR2RGB)
    ex_img3 = cv2.cvtColor(cv2.imread('./data/redeye_samples/24.jpg'), cv2.COLOR_BGR2RGB)

    # st.image([ex_img1, ex_img2, ex_img3], caption= ['1','2','3'], width = 150)

    run_images = [
        './data/redeye_samples/14.jpg',
        './data/redeye_samples/2.jpg', 
        './data/redeye_samples/24.jpg',
        ]
    # image_index = st.selectbox('Select an image using arrow in dropdown box below:', ['1', '2', '3'])
    image_index = st.selectbox('Select an image using arrow in dropdown box below:', [
        ('Unsmiling Female', 0),
        ('Smiling Child', 1),
        ('Smiling Female', 2)])[1]
    st.write(image_index)
    nIter = st.slider('Number of iterations:', 10 , 100, 50, 300)
    run_image = run_images[image_index]

    st.write('<small>Facial features to replace:</small>')
    facial_features = [
        "mouth",
        "left_eye",
        "right_eye",
        "nose",
    ]

    selected_features = []
    selected_a_feature = False
    for feature in facial_features:
        if st.checkbox(feature, feature == 'mouth'):
            selected_a_feature = True
            selected_features.append(feature)
    process_red_eyes = st.checkbox('red eyes')
    selected_a_feature = selected_a_feature or process_red_eyes
    if not selected_a_feature:
        st.error('Must select at least one feature.')
        return

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


    session_id = ''.join(random.choice('0123456789abcdef') for n in xrange(5))
    args = argparse.Namespace(
        jpg_path_in = run_image,
        output_dir = './output_dir_%s/' % session_id,
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

    img_out = read_and_return_mod_image(img, args.use_gan, args.landmark_list, process_red_eyes=process_red_eyes)

    st.image([[ex_img1, ex_img2, ex_img3][image_index], img[:,:,[2,1,0]], img_out * 0.75],
        caption=["Full Image", "Input Image", "Mask"], width=150)

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
            nIter = nIter,
            imgSize = IMGSIZE,
            lam = 0.1,
            checkpointDir = args.checkpoint_dir,
            outDir = args.output_dir,
            outInterval = 10,
            maskType='custom',
            centerScale = 0.25,
            imgs = [img_aligned_path]
        )

        try:
            dir_exists = stat.S_ISDIR(os_stat(gan_args.checkpointDir).st_mode)
        except OSError as e:
            if e.errno == errno.ENOENT:
                print("checkpointDir: %s does not exists" % gan_args.checkpointDir)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        if st.button('Run GAN'):
            waiting_message = st.empty()

            thread_id = threading.currentThread().ident
            @st.cache(ignore_hash=True)
            def create_dc_gan(config, thread_id):
                try:
                    tf_session = tf.Session(config=config)
                    dcgan = DCGAN(tf_session, image_size=gan_args.imgSize,
                                batch_size=min(64, len(gan_args.imgs)),
                                checkpoint_dir=gan_args.checkpointDir, lam=gan_args.lam)
                    return tf_session, dcgan
                except ValueError:
                    st.error("Value Error: Please select `Clear Cache` from the menu on the upper right and try again.")
                    raise

            animated_image = st.empty()
            progress_bar = st.progress(0)
            animated_loss = st.empty()
            created_loss_chart = [False]
            def complete_callback(img, iteration, loss):
                # st.write(name)
                # global created_loss_chart
                animated_image.image(img, caption='iteration %s' % iteration, clamp=False)
                loss_df = pd.DataFrame({'loss': loss})
                loss_df.index = [iteration]
                progress_bar.progress(iteration / float(nIter))
                if not created_loss_chart[0]:
                    animated_loss.area_chart(loss_df)
                    created_loss_chart[0] = True
                else:
                    animated_loss.add_rows(loss_df)

            tf_session, dcgan = create_dc_gan(config, thread_id)
            try:
                with tf_session.as_default():
                    dcgan.complete(gan_args, img_out, callback=complete_callback)
            except ValueError:
                st.error("Value Error: Please select `Clear Cache` from the menu on the upper right and try again.")
                return
            progress_bar.progress(100)

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

            waiting_message.empty()

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

            display_paths = [run_image, output_filepath]
            st.image(display_paths, ["Before","After"], width = 200)
    os.system('ls -lh %s' % args.output_dir)
    rmtree(args.output_dir)
    os.system('ls -lh %s' % args.output_dir)
    st.success('done')

if __name__ == '__main__':
    main()