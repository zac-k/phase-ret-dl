Phase retrieval using deep learning (phase-ret-dl)
==================================================

This README is currently under construction.

Thanks to [Hvass-Labs' TensorFlow tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials), which were immensely helpful in understanding how to implement neural networks using TensorFlow. Some of the code from the tutorial is used in this project without much alteration.

This project uses supervised learning to train a neural network on simulated electron micrographs or retrieved phases. This requires a large volume of training data (A minimum of ~5000 examples), which is only feasible if the specimens are procedurally generated. See my repository [random-specimen-generator](https://github.com/zac-k/random-specimen-generator), which contains a python script that uses the Blender API to procedurally generate specimen files that are compatible with this program.

# Table of contents

* [Introduction](#introduction)
* [Parameters](#parameters)
    * [Hyperparameters](#hyperparameters)
    * [Imaging parameters](#imaging_parameters)
    * [Specimen parameters](#specimen_parameters)
    * [Paths](#paths)



# Introduction
This package contains an artificial neural network (ANN) implemented using tensorflow 1.2.1, and a phase imaging system class, which simulates out-of-focus images using the projection approximation, and retrieves the phase using a FFT solution to the transport-of-intensity equation.

The process begins by simulating two out-of-focus images (and one in-focus, if set)
The ANN has input and output layers of size m^2, where m is the width of each image in pixels. These imag

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

Method to use in optimising the ANN. `'gradient descent'` and `'adam'` are currently supported. Tensorflow supports many more optimisers, and these are trivial to implement in this code if desired.

### 'Learning Rate'

Learning rate of the optimiser. A good starting value is `0.5` for gradient descent, or `1e-4` for adam.

### 'Use Convolutional Layers'

Boolean. Determines whether to use convolutional layers in the ANN. `True` has not been tested due to RAM limitations on my machine.

### 'Number of Epochs'

Number of epochs the training will run for.

### 'Initialisation Type'

How to initialise the weights and biases. `'identity'` is sufficient for most purposes in the present context. Other options are 'random'`, `'randomised identity'`, `'ones'`, and `'zeros'`.

## <a name="simulation-parameters"></a>simulation_parameters

### 'Pre-remove Offset'

Boolean. If `True`, removes the mean error from training and test retrieved phases. Used for removing this source of error so that other sources can be studied independently.

### 'Misalignment'

Three element list of booleans. Tells the ANN whether to use random variations in rotation, scaling, and translation, respectively.

### 'Rotation/Scale/Shift'

Three element list containing the standard deviation of variations in rotation (in degrees), scaling, and translation, respectively. If rotation mode is `'uniform'`, the first element here is the range. Scale and shift are selected from a Gaussian distribution.

### 'Rotation Mode'

Set to `'uniform'` or `'gaussian'`. The former would typically be used for completely arbitrary rotations (range will need to be set to `360` for this separately), while the latter is a reasonable choice for small misalignments.

### 'Load Model'

Boolean. If `True`, loads the model from files saved by previous simulations, negating the need to spend time re-training the model. The other hyperparameter will need to be set to the same values as were used in the previous simulation. This information is available in `errors.txt`, which is output when running the program. Future versions should automate this. Other parameters can be altered, to test the same trained model on different types of test sets, but this hasn't been tested rigorously for all parameters.

### 'Experimental Test Data'

Boolean. If `True`, loads the test images from experimental data. The filenames of specific images are currently hard-coded into the program, and will need to be changed for specific use cases.

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

### 'Mean Inner Potential'

Complex float. The real part is the mean inner potential in volts (~-17 for magnetite), and the imaginary part is related to absorption coefficient. A reasonable value for this is around `1j`.

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