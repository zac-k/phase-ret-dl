import pyprind
import sys
import numpy as np
import tensorflow as tf
import utils
from plot import show
import plot
import phase
import pandas as pd

# Thanks to Hvass-Labs' TensorFlow tutorials (https://github.com/Hvass-Labs/TensorFlow-Tutorials),
# upon which much of this work is based


def new_weights(shape, init_type):
    assert init_type == 'random' or \
           init_type == 'identity' or \
           init_type == 'randomised identity' or \
           init_type == 'ones' or \
           init_type == 'zeros'
    if init_type == 'random':
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    elif init_type == 'identity':
        return tf.Variable(tf.eye(shape[0], num_columns=shape[1]))
    elif init_type == 'randomised identity':
        return tf.Variable(tf.eye(shape[0], num_columns=shape[1]) + tf.truncated_normal(shape, stddev=0.05))
    elif init_type == 'ones':
        return tf.Variable(tf.ones(shape))
    elif init_type == 'zeros':
        return tf.Variable(tf.zeros(shape))


def new_biases(length, init_type):
    assert init_type == 'random' or \
           init_type == 'identity' or \
           init_type == 'randomised identity' or \
           init_type == 'ones' or \
           init_type == 'zeros'
    # if init_type == 'random' or init_type == 'randomised identity':
    #     return tf.Variable(tf.constant(0.05, shape=[length]))
    # else:
    return tf.Variable(tf.zeros([length]))


def new_fc_layer(in_vector,         # The previous Layer
                 num_inputs,    # Number of inputs from prev layer
                 num_outputs,   # Number of outputs
                 activation_function,      # Tensorflow activation function to use
                 init_type):       # Randomise weights and biases?
    # Create weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs], init_type=init_type)
    biases = new_biases(length=num_outputs, init_type=init_type)

    # Calculate output
    layer = tf.matmul(in_vector, weights) + biases

    # Rectify
    if activation_function is not None:
        layer = activation_function(layer)

    return layer


def new_conv_layer(input,
                   num_input_channels,
                   kernel_size,
                   num_kernels,
                   use_pooling=True):

    # Shape of the kernels
    shape = [kernel_size, kernel_size, num_input_channels, num_kernels]

    # Create weights and biases
    weights = new_weights(shape=shape, init_type='random')
    biases = new_biases(length=num_kernels, init_type='random')

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

        layer = tf.nn.relu(layer)

        return layer, weights


def flatten_layer(layer):

    # layer_shape should be [num_images, img_height, img_width, num_channels]
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    # layer_flat is not [num_images, img_height * img_width * num_channels]
    return layer_flat, num_features



np.set_printoptions(threshold=np.inf)


# Create dict of hyperparameter values, each of which will be assigned to the appropriate
# variable closer to where they are used. Number of images is the actual number of simulated
# micrographs. If number of images == 2, 'Train with In-focus Image' uses the approximated
# image for training, and processing the test images, otherwise they are only used for
# the TIE.
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
                   'Initialisation Type': 'identity'}
# 'Pre-remove Offest' removes the mean difference between the exact and retrieved phases for both
# the training and test sets. Will not work with experimental images.
simulation_parameters = {'Pre-remove Offset': False,
                         'Misalignment': [True, False, False],  # rotation, scale, translation
                         'Rotation/Scale/Shift': [60, 0.02, 0.01],  # Rotation is in degrees
                         'Rotation Mode': 'gaussian',  # 'uniform' or 'gaussian'
                         'Load Model': False,
                         'Experimental Test Data': False}
imaging_parameters = {'Window Function Radius': 0.5,
                      'Accelerating Voltage': 300,  # electron accelerating voltage in keV
                      'Use Multislice': False,
                      'Multislice Method': 'files',
                      'Multislice Wavefield Path': 'D:/code/images/multislice/',
                      'Image Size in Pixels': 64,
                      'Multislice Resolution in Pixels': 1024,
                      'Domain Size': 150e-9,  # Width of images in metres
                      'Noise Level': 0.00,
                      'Defocus': 10e-6,
                      'Error Limits': [-3, 3],
                      'Phase Limits': [-3, 3],
                      'Image Limits': [0, 2]}
specimen_parameters = {'Mean Inner Potential': -17 + 1j}
paths = {'Experimental Data Path': './data/images/experimental/',
         'Image Output Path': './data/figures/',
         'Phase Output Path': './data/figures/',
         'Error Output Path': './data/figures/',
         'Load Model Path': './data/figures/',
         'Save Model Path': './data/figures/',
         'Specimen Input Path': './data/specimens/training4/'}


assert simulation_parameters['Rotation Mode'] == 'gaussian' or \
       simulation_parameters['Rotation Mode'] == 'uniform'


exp_path = paths['Experimental Data Path']
image_output_path = paths['Image Output Path']
phase_output_path = paths['Phase Output Path']
error_output_path = paths['Error Output Path']
load_model_path = paths['Load Model Path']
save_model_path = paths['Save Model Path']
f = open(error_output_path + 'errors.txt', 'w')

n_savefile_sets = hyperparameters['Train/Valid/Test Split']
utils.write_dict(f, hyperparameters)
utils.write_dict(f, simulation_parameters)
utils.write_dict(f, imaging_parameters)
utils.write_dict(f, specimen_parameters)

# Set image size and shape
img_size = imaging_parameters['Image Size in Pixels']
M = imaging_parameters['Multislice Resolution in Pixels']
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

use_multislice = imaging_parameters['Use Multislice']

# Set size of hidden layers
hidden_layer_size = hyperparameters['Hidden Layer Size']

# Set parameters for the convolutional layers
kernel_size = [5, 16]
num_kernels = [16, 36]

# Set mean inner potential and noise level
mip = specimen_parameters['Mean Inner Potential']
noise_level = imaging_parameters['Noise Level']

# Set whether to use images or retrieved phases as input data
input_type = hyperparameters['Input Type']
assert input_type == 'images' or input_type == 'phases'
# Set number of images used in through focal series
n_images = hyperparameters['Number of Images']
assert n_images == 2 or n_images == 3

# Create arrays to hold flattened training data
phase_exact_flat_train = []
phase_retrieved_flat_train = []
image_flat_train = []

# Set quantities for training/validation/test split
num_train, num_valid, num_test = hyperparameters['Train/Valid/Test Split']

# Import specimen files
specimen_files = []
specimen_path = paths['Specimen Input Path']
specimen_ext = ''
specimen_name = 'particle'
for i in range(num_train + num_test):
    specimen_files.append(specimen_path + specimen_name + '(' + str(i) + ')' + specimen_ext)

if not simulation_parameters['Load Model']:
    # Compute retrieved training phases and flatten training data
    print('Generating training data...')
    train_generate_bar = pyprind.ProgBar(num_train, stream=sys.stdout)
    for item in range(num_train):
        train_generate_bar.update()
        specimen_file = specimen_files[np.random.randint(len(specimen_files))]
        system_train = phase.PhaseImagingSystem(
               image_size=img_size,
               defocus=imaging_parameters['Defocus'],
               image_width=imaging_parameters['Domain Size'],
               energy=imaging_parameters['Accelerating Voltage']*1e3,
               specimen_file=specimen_files[item],
               mip=mip,
               is_attenuating=True,
               noise_level=noise_level,
               use_multislice=use_multislice,
               multislice_method=imaging_parameters['Multislice Method'],
               M=M,
               item=item,
               path=imaging_parameters['Multislice Wavefield Path'])
        system_train.generate_images(n_images)
        if simulation_parameters['Misalignment'][0]:
            system_train.rotate_images(std=simulation_parameters['Rotation/Scale/Shift'][0],
                                       mode=simulation_parameters['Rotation Mode'],
                                       n_images=n_images)
        if simulation_parameters['Misalignment'][1]:
            system_train.scale_images(simulation_parameters['Rotation/Scale/Shift'][1],
                                      n_images=n_images)
        if simulation_parameters['Misalignment'][2]:
            system_train.shift_images(std=img_size*simulation_parameters['Rotation/Scale/Shift'][2],
                                      n_images=n_images)
        system_train.approximate_in_focus()
        system_train.add_noise_to_micrographs()
        system_train.apodise_images(imaging_parameters['Window Function Radius'])
        system_train.retrieve_phase()
        if simulation_parameters['Pre-remove Offset']:
            system_train.remove_offset()


        system_train.apodise_phases(imaging_parameters['Window Function Radius'])
        phase_exact_flat_train.append(system_train.phase_exact.real.reshape(img_size_flat))
        phase_retrieved_flat_train.append(system_train.phase_retrieved.real.reshape(img_size_flat))

        if item < n_savefile_sets[0]:
            plot.save_image(system_train.image_under,
                            image_output_path + 'image_train_under_' + str(item) + '.png',
                        imaging_parameters['Image Limits'])
            plot.save_image(system_train.image_in,
                            image_output_path + 'image_train_in_' + str(item) + '.png',
                        imaging_parameters['Image Limits'])
            plot.save_image(system_train.image_over,
                            image_output_path + 'image_train_over_' + str(item) + '.png',
                        imaging_parameters['Image Limits'])

        if input_type == 'images':
            if n_images == 3 or hyperparameters['Train with In-focus Image']:
                image_flat_train.append(np.concatenate((system_train.image_under.real.reshape(img_size_flat),
                                        system_train.image_in.real.reshape(img_size_flat),
                                        system_train.image_over.real.reshape(img_size_flat))))
            elif n_images == 2:
                image_flat_train.append(np.concatenate((system_train.image_under.real.reshape(img_size_flat),
                                        system_train.image_over.real.reshape(img_size_flat))))




    # Define average error in training set, calculate it, and print output.
    if input_type == 'phases':
        error_train_ave = utils.average_normalised_rms_error_flat(phase_exact_flat_train, phase_retrieved_flat_train)
        print("Average accuracy on ", "training", "-set: {0: .1%}".format(error_train_ave[0]) + "+/- {0: .1%}".format(error_train_ave[1]), sep='')
        f.write("Average accuracy on training-set: {0: .1%}".format(error_train_ave[0]) + "+/- {0: .1%}".format(error_train_ave[1]) + '\n')


# Create arrays to hold flattened test data
phase_exact_flat_test = []
phase_retrieved_flat_test = []
image_flat_test = []

# Compute retrieved test phases and flatten test data
print("Generating test data...")
test_generate_bar = pyprind.ProgBar(num_test, stream=sys.stdout)
for item in range(num_train, num_test + num_train):
    test_generate_bar.update()
    system_test = phase.PhaseImagingSystem(image_size=img_size,
                                           defocus=imaging_parameters['Defocus'],
                                           image_width=imaging_parameters['Domain Size'],
                                           energy=imaging_parameters['Accelerating Voltage']*1e3,
                                           specimen_file=specimen_files[item],
                                           mip=mip,
                                           is_attenuating=True,
                                           noise_level=noise_level,
                                           use_multislice=use_multislice,
                                           multislice_method=imaging_parameters['Multislice Method'],
                                           M=M,
                                           item=item,
                                           path=imaging_parameters['Multislice Wavefield Path'])
    if simulation_parameters['Experimental Test Data']:
        system_test.image_under = utils.import_micrograph(
            exp_path + 'tims_data_-10um(' + str(item - num_train) + ')', img_size)
        system_test.image_over = utils.import_micrograph(
            exp_path + 'tims_data_10um(' + str(item - num_train) + ')', img_size)
        system_test.image_in = (system_test.image_under + system_test.image_over) / 2
    else:
        system_test.generate_images(n_images)
        if simulation_parameters['Misalignment'][0]:
            system_test.rotate_images(std=simulation_parameters['Rotation/Scale/Shift'][0],
                                      mode=simulation_parameters['Rotation Mode'],
                                      n_images=n_images)
        if simulation_parameters['Misalignment'][1]:
            system_test.scale_images(std=simulation_parameters['Rotation/Scale/Shift'][1],
                                     n_images=n_images)
        if simulation_parameters['Misalignment'][2]:
            system_test.shift_images(std=img_size*simulation_parameters['Rotation/Scale/Shift'][2],
                                     n_images=n_images)
        system_test.approximate_in_focus()
        system_test.add_noise_to_micrographs()

    system_test.apodise_images(imaging_parameters['Window Function Radius'])
    system_test.retrieve_phase()
    if simulation_parameters['Pre-remove Offset']:
        system_test.remove_offset()
    system_test.apodise_phases(imaging_parameters['Window Function Radius'])
    phase_exact_flat_test.append(system_test.phase_exact.real.reshape(img_size_flat))
    phase_retrieved_flat_test.append(system_test.phase_retrieved.real.reshape(img_size_flat))
    if item < n_savefile_sets[2] + num_train:
        plot.save_image(system_test.image_under,
                        image_output_path + 'image_test_under_' + str(item - num_train) + '.png',
                    imaging_parameters['Image Limits'])
        plot.save_image(system_test.image_in,
                        image_output_path + 'image_test_in_' + str(item - num_train) + '.png',
                    imaging_parameters['Image Limits'])
        plot.save_image(system_test.image_over,
                        image_output_path + 'image_test_over_' + str(item - num_train) + '.png',
                    imaging_parameters['Image Limits'])
    if input_type == 'images':
        if n_images == 3 or hyperparameters['Train with In-focus Image']:
            image_flat_test.append(np.concatenate((system_test.image_under.real.reshape(img_size_flat),
                                   system_test.image_in.real.reshape(img_size_flat),
                                   system_test.image_over.real.reshape(img_size_flat))))
        else:
            image_flat_test.append(np.concatenate((system_test.image_under.real.reshape(img_size_flat),
                                   system_test.image_over.real.reshape(img_size_flat))))



# Calculate and print average normalised rms error in test set prior to processing
# through neural network
if len(phase_exact_flat_test) > 0 and not simulation_parameters['Experimental Test Data']:
    error_pre_adj = utils.average_normalised_rms_error_flat(phase_exact_flat_test, phase_retrieved_flat_test)
    print("Accuracy on test set (pre adjustment): {0: .1%}".format(error_pre_adj[0]) + "+/- {0: .1%}".format(error_pre_adj[1]))
    f.write("Accuracy on test set (pre adjustment): {0: .1%}".format(error_pre_adj[0]) + "+/- {0: .1%}".format(error_pre_adj[1]) + '\n')

# Determine number of nodes in input layer
if input_type == 'images':
    if hyperparameters['Train with In-focus Image']:
        n_training_images = 3
    else:
        n_training_images = n_images

    input_size = n_training_images * img_size_flat
    num_channels = n_training_images
elif input_type == 'phases':
    input_size = img_size_flat
    num_channels = 1

# Define placeholder variables
x = tf.placeholder(tf.float32, [None, input_size])
y_true = tf.placeholder(tf.float32, [None, img_size_flat])

# Define convolutional layers
if hyperparameters['Use Convolutional Layers']:
    x_for_conv = tf.reshape(x, [-1, input_size, input_size, num_channels])
    layer_conv1, weights_conv1 = new_conv_layer(input=x_for_conv,
                                                num_input_channels=num_channels,
                                                kernel_size=kernel_size[0],
                                                num_kernels=num_kernels[0],
                                                use_pooling=True)
    layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                                num_input_channels=num_kernels[0],
                                                kernel_size=kernel_size[1],
                                                num_kernels=num_kernels[1],
                                                use_pooling=True)
    input_for_first_fc_layer, num_inputs_for_first_fc_layer = flatten_layer(layer_conv2)
else:
    input_for_first_fc_layer = x
    num_inputs_for_first_fc_layer = input_size


# Define fully connected layers
num_hidden_layers = len(hyperparameters['Hidden Layer Size'])
hidden_layers = []
for i in range(num_hidden_layers):
    hidden_layers.append(tf.Variable(tf.zeros([hidden_layer_size[i]])))
for i in range(num_hidden_layers):
    if i == 0:
        hidden_input = input_for_first_fc_layer
        hidden_input_size = num_inputs_for_first_fc_layer
    else:
        hidden_input = hidden_layers[i - 1]
        hidden_input_size = hidden_layer_size[i - 1]
    hidden_layers[i] = new_fc_layer(hidden_input,
                                    hidden_input_size,
                                    hidden_layer_size[i],
                                    activation_function=tf.nn.tanh,
                                    init_type=hyperparameters['Initialisation Type'])

if num_hidden_layers == 0:
    penultimate_layer = input_for_first_fc_layer
    penultimate_layer_size = num_inputs_for_first_fc_layer
else:
    penultimate_layer = hidden_layers[num_hidden_layers - 1]
    penultimate_layer_size = hidden_layer_size[-1]

output = new_fc_layer(penultimate_layer,
                      penultimate_layer_size,
                      img_size_flat,
                      activation_function=None,
                      init_type=hyperparameters['Initialisation Type']
                      )

# Define cost function
cost = tf.reduce_mean(tf.squared_difference(y_true, output))

# Define optimizer
learning_rate = hyperparameters['Learning Rate']
optimiser_type = hyperparameters['Optimiser Type']
if optimiser_type == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
elif optimiser_type == 'gradient descent':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
elif optimiser_type == 'adagrad':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
elif optimiser_type == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
elif optimiser_type == 'adagrad da':
    optimizer = tf.train.AdagradDAOptimizer(learning_rate=learning_rate).minimize(cost)
elif optimiser_type == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate).minimize(cost)
elif optimiser_type == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate).minimize(cost)
elif optimiser_type == 'proximal gradient descent':
    optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
elif optimiser_type == 'proximal adagrad':
    optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate).minimize(cost)
elif optimiser_type == 'rms prop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
else:
    raise ValueError('Unknown optimizer type')

session = tf.Session()
saver = tf.train.Saver()

# Define feed dict for test set
if input_type == 'images':
    feed_dict_test = {x: image_flat_test,
                      y_true: phase_exact_flat_test
                      }
elif input_type == 'phases':
    feed_dict_test = {x: phase_retrieved_flat_test,
                      y_true: phase_exact_flat_test
                      }
if simulation_parameters['Load Model']:
    saver.restore(session, load_model_path + 'model')

else:
    # Initialise variables
    session.run(tf.global_variables_initializer())



    # Set batch variables
    batch_size = hyperparameters['Batch Size']  # Number of training examples in each batch
    num_batches = int(np.floor(num_train / batch_size))  # Calculate number of batches

    # Calculate the mean of the training data for later use
    mean_exact_train = np.mean(phase_exact_flat_train, axis=0)


    # Train the model
    print('Training...')

    num_epochs = hyperparameters['Number of Epochs']
    train_bar = pyprind.ProgBar(num_epochs * num_batches, stream=sys.stdout)
    for q in range(num_epochs):
        for i in range(num_batches):
            train_bar.update()
            if (i + 1) * batch_size < num_train:
                if input_type == 'images':
                    x_batch = image_flat_train[i * batch_size: (i + 1) * batch_size]
                elif input_type == 'phases':
                    x_batch = phase_retrieved_flat_train[i * batch_size: (i + 1) * batch_size]
                y_true_batch = phase_exact_flat_train[i * batch_size: (i + 1) * batch_size]
                feed_dict_train = {x: x_batch,
                                   y_true: y_true_batch}

                session.run(optimizer, feed_dict=feed_dict_train)




# Calculate and print average normalised rms error in test set after processing through
# trained neural network
error = tf.sqrt(tf.reduce_sum(tf.squared_difference(y_true, output), 1) / tf.reduce_sum(tf.square(y_true), 1))
accuracy = tf.reduce_mean(error)
acc, x_val = session.run([accuracy, x], feed_dict=feed_dict_test)
print("Accuracy on ", "test", "-set (post-adjustment): {0: .1%}".format(acc), sep='')
f.write("Accuracy on test-set (post-adjustment): {0: .1%}".format(acc) + '\n')

# Obtain output of neural net on test set
output_images = session.run(output, feed_dict=feed_dict_test)

# Calculate average rms error between test outputs and the mean of the
# training target images (exact phases) and print it
if not simulation_parameters['Load Model']:
    error_test_vs_train = 0
    for output_image in output_images:
        error_test_vs_train += np.sqrt(
                            np.sum(np.square(mean_exact_train -
                                             output_image)) / np.sum(np.square(mean_exact_train))
                )
    error_test_vs_train /= num_test
if not simulation_parameters['Experimental Test Data']:
    #print("Accuracy on ", "test input", " compared to training output: {0: .1%}".format(error_test_vs_train), sep='')
    #f.write("Accuracy on test input compared to training output: {0: .1%}".format(error_test_vs_train) + '\n')

    error_ret = (np.array(phase_retrieved_flat_test) - np.array(phase_exact_flat_test)).tolist()
    error_adj = (np.array(output_images) - np.array(phase_exact_flat_test)).tolist()

if not simulation_parameters['Load Model']:
    print("Saving phases from training set...")
    train_phase_save_bar = pyprind.ProgBar(n_savefile_sets[0], stream=sys.stdout)
    for i in range(n_savefile_sets[0]):
        train_phase_save_bar.update()
        plot.save_image(np.reshape(phase_exact_flat_train[i], img_shape),
                        phase_output_path + 'phase_exact_train_' + str(i) + '.png',
                        imaging_parameters['Phase Limits'])
        plot.save_image(np.reshape(phase_retrieved_flat_train[i], img_shape),
                        phase_output_path + 'phase_retrieved_train_' + str(i) + '.png',
                        imaging_parameters['Phase Limits'])

print("Saving phases and errors from test set...")
test_phase_save_bar = pyprind.ProgBar(n_savefile_sets[2], stream=sys.stdout)
for i in range(n_savefile_sets[2]):
    test_phase_save_bar.update()
    if not simulation_parameters['Experimental Test Data']:
        plot.save_image(np.reshape(phase_exact_flat_test[i], img_shape),
                        phase_output_path + 'phase_exact_test_' + str(i) + '.png',
                        imaging_parameters['Phase Limits'])
        plot.save_image(np.reshape(error_ret[i], img_shape),
                        error_output_path + 'error_retrieved_test_' + str(i) + '.png',
                        imaging_parameters['Error Limits'])
        plot.save_image(np.reshape(error_adj[i], img_shape),
                        error_output_path + 'error_adjusted_test_' + str(i) + '.png',
                        imaging_parameters['Error Limits'])
    plot.save_image(np.reshape(phase_retrieved_flat_test[i], img_shape),
                    phase_output_path + 'phase_retrieved_test_' + str(i) + '.png',
                    imaging_parameters['Phase Limits'])
    plot.save_image(np.reshape(output_images[i], img_shape),
                    phase_output_path + 'phase_adjusted_test_' + str(i) + '.png',
                    imaging_parameters['Phase Limits'])



errors = pd.DataFrame({})

if not simulation_parameters['Experimental Test Data']:
    for i in range(num_test):
        error_test = utils.normalised_rms_error(phase_exact_flat_test[i], phase_retrieved_flat_test[i])
        f.write("Accuracy on test input " + str(i) + ": {0: .1%}".format(error_test) + '\n')

    accuracy = tf.sqrt(tf.reduce_sum(tf.squared_difference(y_true, output), 1) / tf.reduce_sum(tf.square(y_true), 1))
    acc, x_val = session.run([accuracy, x], feed_dict=feed_dict_test)
    for i, output_image in enumerate(output_images):
        f.write("Accuracy on test input " + str(i) + "(adjusted): {0: .1%}".format(acc[i]) + '\n')

if not simulation_parameters['Load Model']:
    for i in range(num_train):
        error_train = utils.normalised_rms_error(phase_exact_flat_train[i], phase_retrieved_flat_train[i])
        f.write("Accuracy on training input " + str(i) + ": {0: .1%}".format(error_train) + '\n')

f.close()

# Save trained model
if not simulation_parameters['Load Model']:
    saver.save(session, save_model_path + 'model')

#utils.beep()  # Alert user that script has finished
show()  # Prevent plt.show(block=False) from closing plot window
