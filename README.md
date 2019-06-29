# Eyewash
Clean Photos through AI

## Project Description
Eyewash is a package to automatically remove blemishes from portrait photos. Users no long have to manually select pixels as well as create more realistic fixes to the blemishes rather than filling in with a specific color.  The implementation uses OpenCv haar cascades to detect redeye and remove the affected pixels through image infilling with DCGANs.   

A google slide presentation can be found here: [Eyewash](http://tinyurl.com/redeyewash)

## Setup

Clone this repo:

```
git clone https://github.com/dannyqnguyen/eyewash.git eyewash
cd eyewash
```

Install requirements
```
pip install -r requirements.txt
```

Add the following libraries to your PYTHONPATH. To do this in a conda enviornment, run the following commands:

```
conda-develop ./eyewash
conda-develop ./dcgan
conda-develop ./FaceSwap
conda-develop ./openface/util
conda-develop ./openface
```


## Usage


you can run the command line as follows:

```
python create_image.py data/redeye_samples/2.jpg output_dir --checkpoint_dir checkpoint --use_gan True
```

`first_argument` Path to input image.

`second_argument` Path to create output directory where output files are stored.

`--checkpoint_dir` Path to directory containing saved gan model checkpoint

`--use_gan` Optional argument to feed boolean value to use the GAN to fill in blemishes. If this is not supplied or set to False, the blemishes will be filled with black pixels.  

## Pipeline
This GAN pipeline starts with face alignment, followed by redeye blemish detection. After that, the GAN will fill in the detected blemish pixels and finally we use Wu Huikai's [library](https://github.com/wuhuikai/FaceSwap) for faceswap back onto the original image.  

## Images
I have included some sample images to test out redeye blemish removal in `data\preprocessed\redeye`. Please note for GAN workflow that we need to do face alignment on a single subject and redeye images where there are multiple faces or cropped faces will fail this pipeline. To process these images, set the `--use_gan` flag to `False`.

## DCGAN Training
This project modifies Brandon Amos's [DCGAN model](https://github.com/bamos/dcgan-completion.tensorflow). It uses the same training procedure as well. 

For best results, we process the training dataset of photos through face alignment. For this we use OpenFace’s alignment [library](https://cmusatyalab.github.io/openface/) to pre-process the images to be 64x64.

```
./openface/util/align-dlib.py <path_to_training_images> align innerEyesAndBottomLip <path_to_save_aligned_training_images> --size 64
```

And finally we’ll flatten the aligned images directory so that it just contains images and no sub-directories.

```
pushd <path_to_training_images>
find . -name '*.png' -exec mv {} . \;
find . -type d -empty -delete
popd
```

We’re ready to train the DCGAN. 

```
dcgan/train-dcgan.py --dataset <path_to_saved_aligned_training_images> --epoch 20
```

You can check what randomly sampled images from the generator look like in the samples directory.

You can also view the TensorFlow graphs and loss functions with TensorBoard.

```
tensorboard --logdir ./logs
```


## Requirements
Eyewash was tested and developed with the following packages.
```


```
PYTHON PATH
Lets start with a blank slate: remove `.git` and re initialize the repo
```
cd $repo_name
rm -rf .git   
git init   
git status
```  

## Requisites

- List all packages and software needed to build the environment
- This could include cloud command line tools (i.e. gsutil), package managers (i.e. conda), etc.

#### Dependencies

- [Streamlit](streamlit.io)

#### Installation
To install the package above, pleae run:
```shell
pip install -r requiremnts
```

## Build Environment
- Include instructions of how to launch scripts in the build subfolder
- Build scripts can include shell scripts or python setup.py files
- The purpose of these scripts is to build a standalone environment, for running the code in this repository
- The environment can be for local use, or for use in a cloud environment
- If using for a cloud environment, commands could include CLI tools from a cloud provider (i.e. gsutil from Google Cloud Platform)
```
# Example

# Step 1
# Step 2
```

## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- **DO NOT STORE CREDENTIALS IN THE CONFIG DIRECTORY!!**
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)


## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
`
