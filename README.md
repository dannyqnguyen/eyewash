# Eyewash
Clean Photos through AI

## Project Description
Eyewash is a package to automatically remove blemishes from portrait photos. Users no long have to manually select pixels as well as create more realistic fixes to the blemishes rather than filling in with a specific color.  The implementation uses OpenCv haar cascades to detect redeye and remove the affected pixels through image infilling with DCGANs.   

A google slide presenntation can be found here: [Eyewash](http://tinyurl.com/redeyewash)

## Usage

After cloning this repo and installing the requirements, you can run the command line as follows:

```
python create_image.py data/preprocessed/redeye/051.jpg output_dir --checkpoint_dir checkpoint --use_gan True
```
`first_argument` Path to input image.
`second_argument` Path to create output directory where output files are stored.
`--checkpoint_dir` Path to directory containing saved gan model checkpoint
`--use_gan` Optional argument to feed boolean value to use the GAN to fill in blemishes. If this is not supplied or set to False, the blemishes will be filled with black pixels.  


## DCGAN Training
This project modifies Brandon Amos's DCGAN model. It uses the same training procedure as well. 

For best results, we process the training dataset of photos through face alignment. For this we use OpenFace’s alignment tool to pre-process the images to be 64x64.

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
PYTHON PATH
Lets start with a blank slate: remove `.git` and re initialize the repo
```
cd $repo_name
rm -rf .git   
git init   
git status
```  
You'll see a list of file, these are files that git doesn't recognize. At this point, feel free to change the directory names to match your project. i.e. change the parent directory Insight_Project_Framework and the project directory Insight_Project_Framework:
Now commit these:
```
git add .
git commit -m "Initial commit"
git push origin $branch_name
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
