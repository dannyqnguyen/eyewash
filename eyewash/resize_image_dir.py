import argparse
from image_utils import resize_to_square_and_pad_dir

parser = argparse.ArgumentParser()
parser.add_argument("desired_size", help="desired pixel length of square image", type=int)
parser.add_argument("raw_img_dir", help="path to image dir to process", type=str)
parser.add_argument("processed_img_dir", help="path to image dir put processed images", type=str)
args = parser.parse_args()

resize_to_square_and_pad_dir(args.desired_size, args.raw_img_dir, args.processed_img_dir)

