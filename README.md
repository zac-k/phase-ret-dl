Phase retrieval using deep learning (phase-ret-dl)
==================================================






# Table of contents

* [Introduction](#introduction)
* [System requirements](#system-requirements)
* [Parameters](#parameters)
    * [Hyperparameters](#hyperparameters)
    * [Imaging parameters](#imaging_parameters)
    * [Specimen parameters](#specimen_parameters)
    * [Paths](#paths)
    * [Example Settings](#example-settings)
* [Files](#files)
    * [main.py](#mainpy)
    * [phase.py](#phasepy)
    * [plot.py](#plotpy)
    * [utils.py](#utilspy)
    * [README.md](#READMEmd)
    * [notes.txt](#notestxt)
* [Acknowledgements](#acknowledgements)



# Introduction

This project uses supervised learning to train a neural network on simulated electron micrographs or retrieved phases. This requires a large volume of training data (A minimum of ~5000 examples), which is only feasible if the specimens are procedurally generated. See my repository [specimen-generator](https://github.com/zac-k/specimen-generator), which contains a python script that uses the Blender API to procedurally generate specimen files that are compatible with this program.

A phase imaging system class, which simulates out-of-focus images using the projection approximation, and retrieves the phase using a FFT solution to the transport-of-intensity equation is used to train, and test, an artificial neural network (ANN), which is implemented using tensorflow 1.2.1.

The process begins by simulating two out-of-focus images (and one in-focus, if set) using the projected potential of a specimen that is read from file. Sources of error, such as shot noise and image misalignment, can also be added. These images can then be used to obtain the phase. This project uses a phase retrieval algorithm based on the transport-of-intensity equation (TIE), but additional methods (such as the Gerchberg-Saxton algorithm) will probably be added in the future.

The ANN has input layer of size n_images*m<sup>2</sup>---where m is the width of each image in pixels, and n_images is one for the phase input method, and two or three for the image input method---and the output layer is size m<sup>2</sup>. The images (or phase) are flattened and joined end-to-end to form the input vector. The projected potential is scaled to form an 'exact phase', which is also flattened, and is used as the output vector for training the ANN.

One input and one output vector constitutes one training pair. A large number of these pairs are generated, and the ANN is trained on these pairs in batches. After training, the ANN can be used to process a test set of input images or phases (depending on which was used in the training). If simulated images are used for the test set, root-mean-square (rms) errors are computed for each output by comparing it with the exact phase. Average rms errors are calculated for the entire set. 

# System requirements

The hardware required depends on the resolution you are using, the number of training and test sets, and the complexity of the model. I am currently using

CPU:    Core i5-4670K @4.0GHz

RAM:    16GB DDR3

GPU:    GTX970 - 4GB VRAM

When running on the cpu, this is fine for most of the settings I use. Simulating the training data (from stored specimen files) takes about 40 minutes, and training the network with 5000 examples takes about 5 hours. Processing a test set of 100 examples, including calculating errors and saving images, takes approximately half an hour.

It also works fine on my other system (Core i5-2500K @4.2GHz, 8GB DDR3) at about the same speed, but the RAM maxes out and it starts using virtual memory, which makes the PC largely unusable for other purposes while it's running.

GPU compute was unusable due to the relatively small amount of VRAM on my card, and I suggest making sure you have enough VRAM for your purposes before going to the trouble of installing tensorflow-gpu, CUDA, and CuDNN, if you don't already have them. 

# Parameters

[main.py](https://github.com/zac-k/phase-ret-dl/blob/master/main.py) contains four dicts with values that can be adjusted to alter properties of the image simulation, phase retrieval method, and training and testing of the neural network. The parameters are roughly grouped into meaningful categories (the dicts). Many of the parameters are subsequently referenced by shorter named variables for brevity. An explanation of these is given below.

## <a name="hyperparameters"></a>hyperparameters

### 'Hidden Layer Size'
This is a list containing the number of neurons per hidden layer. The total number of hidden layers is given by the length of the list. An ANN with one hidden layer is created by setting this parameter to a list with one entry.

### 'Input Type'

Set to `'images'` or `'phases'`. Determines whether the neural network is trained and tested on out-of-focus images directly, or on retrieved phases.

### 'Number of Images'

Set to `2` or `3`. This determines how many measured images are simulated using a transfer function formalism (or provided, when using experimental data). If set to `2`, only under- and over-focus images are used. If set to `3`, an in-focus image is also used.

### 'Train with In-focus Image'

Boolean. Has no effect when three images are used. When two images are used, setting this parameter to `True` averages the under- and over-focus micrographs to obtain an in-focus image to train and test the ANN with three images per set.

### 'Train/Valid/Test Split'

Dict of length three containing the size of the training, validation, and test sets, respectively. Validation is currently not implemented in this code and has no effect.

### 'Batch Size'

Number of training examples to use in each training batch.

### 'Optimiser Type'

Method to use in optimising the ANN. Valid values are `'gradient descent'`, `'adam'`, `'adadelta'`, `'adagrad'`, `'adagrad da'`, `'momentum'`, `'ftrl'`, `'proximal gradient descent'`, `'proximal adagrad'`, and `'rms prop'`.

### 'Learning Rate'

Learning rate of the optimiser. A good starting value is `0.5` for gradient descent, or `1e-4` for adam.

### 'Use Convolutional Layers'

Boolean. Determines whether to use convolutional layers in the ANN. `True` has not been tested due to RAM limitations on my machine.

### 'Number of Epochs'

Number of epochs (iterations over the full training set)the training will run for.

### 'Initialisation Type'

How to initialise the weights and biases. `'identity'` is sufficient for most purposes in the present context. Other options are `'random'`, `'randomised identity'`, `'ones'`, and `'zeros'`.

## <a name="simulation-parameters"></a>simulation_parameters

### 'Pre-remove Offset'

Boolean. If `True`, removes the mean error from training and test retrieved phases. Used for removing this source of error so that other sources can be studied independently. A more rigorous way of doing this would be to compute the mean error only over an annulus, so that the offset is computed as the average difference between the unaltered (free-space) phase. This may be implemented in future versions.

### 'Phase Retrieval Method'

Either 'TIE' (transport-of-intensity equation) or 'GS' (Gerchberg-Saxton-Misell). The latter has been implemented, but I'm not sure if it's working yet. It may just be a matter of choosing appropriate parameters to get it to work properly. 

### 'Misalignment'

Three element list of booleans. Tells the ANN whether to use random variations in rotation, scaling, and translation, respectively. Without loss of generality, one image in each through-focal series can be left unaltered to act as a reference against which to compare the exact phase. In three image mode, the reference image is the in-focus image. In two image mode, it is the under-focus image. 

### 'Rotation/Scale/Shift'

Three element list containing the standard deviation of variations in rotation (in degrees), scaling, and translation, respectively. If rotation mode is `'uniform'`, the first element here is the range. Scale and shift are selected from a Gaussian distribution.

### 'Rotation Mode'

Set to `'uniform'` or `'gaussian'`. The former would typically be used for completely arbitrary rotations (range will need to be set to `360` for this separately), while the latter is a reasonable choice for small misalignments.

### 'Load Model'

Boolean. If `True`, loads the model from files saved by previous simulations, negating the need to spend time re-training the model. The other hyperparameter will need to be set to the same values as were used in the previous simulation. This information is available in `errors.txt`, which is output when running the program. Future versions should automate this. Other parameters can be altered, to test the same trained model on different types of test sets, but this hasn't been tested rigorously for all parameters.

### 'Experimental Test Data'

Boolean. If `True`, loads the test images from experimental data. The filenames of specific images are currently hard-coded into the program, and will need to be changed for specific use cases.

### 'Retrieve Phase Component'

Set to 'total', 'electrostatic', or 'magnetic'. Component of the phase that is used for training, phase retrieval, and error determination.

## imaging_parameters

### 'Window Function Radius'

Radius of the apodisation function as a fraction of the image width. This is used to apodise the images, and also defines the circular domain over which errors are computed. Set this somewhere in the range 0.3--0.5, depending on particle size and defocus.

### 'Accelerating Voltage'

Electron accelerating voltage in keV; e.g., `200`, `300`, etc.

### 'Use Multislice'

Boolean. Whether to use the multislice method for image simulation. The projection approximation is still used to generate projected potentials used as the 'exact phase'.

### 'Multislice Method'

String. If set to 'files', program will load wavefield files and compute the phase maps, for the image simulations, using a phase unwrapping algorithm. This can be used for loading any externally created wavefield files, which don't necessarily have to be multislice simulated. Setting to anything else will cause this program to perform the multislice simulations itself. Currently, this is infeasibly slow, and the functions need to be vectorised to speed this up. The Numba package and `@jit` decorator are used to speed up the calculations somewhat, but are still too slow for them to be useful for generating enough training sets. The method used for simulation is based on [E. J. Kirkland's C code)[http://people.ccmr.cornell.edu/~kirkland/]. It is currently recommended to use the C code to generate wavefield files and utilise them in this project using the 'files' setting described here.

### 'Multislice Wavefield Path'

Path containing the multislice wavefield files.

### 'Image Size in Pixels'

Integer. Height and width of the images in pixels.

### 'Multislice Resolution in Pixels'

Integer. Height and width of the phase maps, in pixels, used for image simulation. Once the intensity images are produced, they are downsampled to the image size. Must be 2^n times the image size, where n is some integer.

### 'Domain Size'

Width of images in metres.

### 'Noise Level'

Fractional noise level. For example, set this to `0.05` for 5\% noise.

### 'Defocus'

Defocus in metres.

### 'Error Limits'

Two element dict containing the maximum and minimum phase values for the error output images. Values outside these limits are clipped at the maximum and minimum value.

### 'Phase Limits'

Two element dict containing the maximum and minimum phase values for the phase output images. Values outside these limits are clipped at the maximum and minimum value.

### 'Image Limits'
Two element dict containing the maximum and minimum intensity values for the intensity output images. Values outside these limits are clipped at the maximum and minimum value.



## specimen_parameters

### 'Use Electrostatic/Magnetic Potential'

Two element boolean list. Determines which components of the phase to utilise in the simulation. If only one of these is True, the exact phase is set to that phase. If both are True, the exact phase is the combined phase, and specimen flipping is implemented to separate the phases if simulation_parameters['Retrieve Phase Component'] is set to 'electrostatic' or 'magnetic'.

### 'Mean Inner Potential'

Complex float. The real part is the mean inner potential in volts (~-17 for magnetite), and the imaginary part is related to absorption coefficient. A reasonable value for this is around `1j`.

### 'Mass Magnetization'

Uniform mass magnetization of the specimen in emu/g (80 for magnetite).

### 'Density'

Uniform density of the specimen in g/cm^3 (5.18 for magnetite).

## Paths

These paths determine directory locations where input files are found and output files will be saved. Path names end in `/` and the directory must exist.

### 'Experimental Data Path'

Directory containing experimental micrographs for use as test data.

### 'Image Output Path'

Directory where png image files (i.e., micrographs)will be saved

### 'Phase Output Path'

Directory where the exact, retrieved, and ANN computed phases will be output as png files.

### 'Error Output Path'

Images of the total error for retrieved and ANN computed phases are output here, as is the `errors.txt` file. This file contains a printout of the parameter dicts, the average errors over training and test sets, as well as the individual errors for each example in these sets.

### 'Load Model Path'

Path where model files, which allow the ANN to be reused without retrianing, must be located. 

### 'Save Model Path'

Location where model files are saved for reuse.

### 'Specimen Input Path'

Location that must contain the specimen files.

## Example settings

Because I am often testing out fringe cases, while at the same time working on improving the code and expanding its functionality, the parameters in the latest version may not be a good starting place for newcomers wanting to understand the process and the code. Here, I am including a set of parameters that 

    hyperparameters = {'Hidden Layer Size': [50000],
                       'Input Type': 'images',
                       'Number of Images': 2,
                       'Train with In-focus Image': False,  # False has no effect if n_images == 3
                       'Train/Valid/Test Split': [5000, 0, 100],
                       'Batch Size': 50,
                       'Optimiser Type': 'gradient descent',
                       'Learning Rate': 0.5,
                       'Use Convolutional Layers': False,
                       'Number of Epochs': 50,
                       'Initialisation Type': 'identity'
                       }
                        
    simulation_parameters = {'Pre-remove Offset': False,
                             'Misalignment': [True, True, True],  # rotation, scale, translation
                             'Rotation/Scale/Shift': [3, 0.02, 0.01],  # Rotation is in degrees
                             'Rotation Mode': 'gaussian',  # 'uniform' or 'gaussian'
                             'Load Model': False,
                             'Experimental Test Data': False,
                             'Retrieve Phase Component': 'total',  # 'total', 'electrostatic', or 'magnetic'
                             }
                             
    imaging_parameters = {'Window Function Radius': 0.5,
                          'Accelerating Voltage': 300,  # electron accelerating voltage in keV
                          'Use Multislice': False,
                          'Multislice Method': 'files',
                          'Multislice Wavefield Path': './data/images/multislice/',
                          'Image Size in Pixels': 64,
                          'Multislice Resolution in Pixels': 1024,
                          'Domain Size': 150e-9,  # Width of images in metres
                          'Noise Level': 0.00,
                          'Defocus': 10e-6,
                          'Error Limits': [-3, 3],
                          'Phase Limits': [-3, 3],
                          'Image Limits': [0, 2]
                          }
                          
    specimen_parameters = {'Use Electrostatic/Magnetic Potential': [True, False],
                           'Mean Inner Potential': -17 + 1j,
                           'Mass Magnetization': 80,  # emu/g
                           'Density': 5.18  # g/cm^3
                           }
    
    paths = {'Experimental Data Path': './data/images/experimental/',
             'Image Output Path': './data/images/',
             'Phase Output Path': './data/images/',
             'Error Output Path': './data/images/',
             'Load Model Path': './data/images/',
             'Save Model Path': './data/images/',
             'Specimen Input Path': './data/specimens/'}

# Files

## [main.py](https://github.com/zac-k/phase-ret-dl/blob/master/main.py)

Contains the dicts that set up the ANN and phase retrieval parameters. Runs the phase retrieval system, trains the ANN, and computes the errors on the test set.

## [phase.py](https://github.com/zac-k/phase-ret-dl/blob/master/phase.py)

PhaseImagingSystem class. Simulates out-of-focus electron micrographs using the projection approximation or multislice method<sup>1</sup>. Wavefield files, generated using the C code separately, can be imported into this project, and that is the method I recommend if you want to use the more realistic image simulations created using the multislice approach, unless you have the time and expertise to vectorise the python code.

This file also contains methods to retrieve the phase, using the TIE, from the out-of-focus micrographs, and includes relevant methods such as those for downsampling.


<sup>1. The multislice method is a python implementation of [E. J. Kirkland's C code](http://people.ccmr.cornell.edu/~kirkland/), but runs extremely slowly. The code needs to be vectorised to make it run fast enough to use; this is not straightforward.</sup>

## [plot.py](https://github.com/zac-k/phase-ret-dl/blob/master/plot.py)

Contains functions for displaying images and plots. Can display images immediately using matplotlib, but [main.py](https://github.com/zac-k/phase-ret-dl/blob/master/main.py) is currently set up to save the images directly in png format.

## [utils.py](https://github.com/zac-k/phase-ret-dl/blob/master/utils.py)

Small library of functions that do not belong anywhere else. Includes error calculation functions.

## [README.md](https://github.com/zac-k/phase-ret-dl/blob/master/README.md)

This README.

## [notes.txt](https://github.com/zac-k/phase-ret-dl/blob/master/notes.txt)

These are just notes I keep on my own personal usage of this package. Some of the information in there may be useful to others, such as what learning rates to try for certain optimisers, but most of it will not make sense out of context. The notes are often speculative and not based on well tested ideas, so read with caution.

# Acknowledgements

Thanks to [Hvass-Labs' TensorFlow tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials), which were immensely helpful in understanding how to implement neural networks using TensorFlow. Some of the code from the tutorial is used in this project without much alteration.