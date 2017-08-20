import pyprind, sys
import numpy as np
import tensorflow as tf
import utils
from plot import show
import plot
import phase

# Thanks to Hvass-Labs' TensorFlow tutorials (https://github.com/Hvass-Labs/TensorFlow-Tutorials),
# upon which much of this work is based


def new_weights(shape, init_type):
    if init_type == 'random':
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    elif init_type == 'identity':
        return tf.Variable(tf.eye(shape[0], num_columns=shape[1]))
    elif init_type == 'randomised identity':
        return tf.Variable(tf.eye(shape[0], num_columns=shape[1]) + tf.truncated_normal(shape, stddev=0.05))
    elif init_type == 'ones':
        return tf.Variable(tf.ones(shape))


def new_biases(length, init_type):
    if init_type == 'random' or init_type == 'randomised identity':
        return tf.Variable(tf.constant(0.05, shape=[length]))
    else:
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
f = open('./data/figures/errors.txt', 'w')

# Create dict of hyperparameter values, each of which will be assigned to the appropriate
# variable closer to where they are used.
hyperparameters = {'Hidden Layer Size': 50000,
                   'Number of Hidden Layers': 1,
                   'Input Type': 'phases',
                   'Train/Valid/Test Split': [50, 0, 100],
                   'Batch Size': 50,
                   'Optimiser Type': 'gradient descent',
                   'Learning Rate': 0.5,
                   'Use Convolutional Layers': False,
                   'Number of Epochs': 50}
imaging_parameters = {'Window Function Radius': 0.5,
                      'Use Multislice': False,
                      'Image Size in Pixels': 64,
                      'Multislice Resolution in Pixels': 1024,
                      'Noise Level': 0.00}

utils.write_dict(f, hyperparameters)
utils.write_dict(f, imaging_parameters)

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
mip = -17 - 1j
noise_level = imaging_parameters['Noise Level']

# Set whether to use images or retrieved phases as input data
input_type = hyperparameters['Input Type']
assert input_type == 'images' or input_type == 'phases'

# Create arrays to hold flattened training data
phase_exact_flat_train = []
phase_retrieved_flat_train = []
image_flat_train = []

# Set quantities for training/validation/test split
num_train, num_valid, num_test = hyperparameters['Train/Valid/Test Split']

# Import specimen files
specimen_files = []
specimen_path = './data/specimens/training4/'
specimen_ext = ''
specimen_name = 'particle'
for i in range(num_train + num_test):
    specimen_files.append(specimen_path + specimen_name + '(' + str(i) + ')' + specimen_ext)

# Compute retrieved training phases and flatten training data
print('Generating training data...')
train_generate_bar = pyprind.ProgBar(num_train, stream=sys.stdout)
for item in range(num_train):
    train_generate_bar.update()
    specimen_file = specimen_files[np.random.randint(len(specimen_files))]
    system_train = phase.PhaseImagingSystem(
           image_size=img_size,
           defocus=8e-6,
           image_width=150e-9,
           energy=300e3,
           specimen_file=specimen_files[item],
           mip=mip,
           is_attenuating=True,
           noise_level=noise_level,
           use_multislice=use_multislice,
           M=M)
    system_train.generate_images()
    system_train.apodise_images(imaging_parameters['Window Function Radius'])
    system_train.retrieve_phase()
    system_train.apodise_phases(imaging_parameters['Window Function Radius'])
    phase_exact_flat_train.append(system_train.phase_exact.real.reshape(img_size_flat))
    phase_retrieved_flat_train.append(system_train.phase_retrieved.real.reshape(img_size_flat))
    if item < 5:
        plot.save_image(system_train.image_under,
                        './data/figures/image_under_' + str(item) + '.png', 'image')
        plot.save_image(system_train.image_in,
                        './data/figures/image_in_' + str(item) + '.png', 'image')
        plot.save_image(system_train.image_over,
                        './data/figures/image_over_' + str(item) + '.png', 'image')
    if input_type == 'images':
        image_flat_train.append(np.concatenate((system_train.image_under.real.reshape(img_size_flat),
                                system_train.image_in.real.reshape(img_size_flat),
                                system_train.image_over.real.reshape(img_size_flat))))




# Define average error in training set, calculate it, and print output.
if input_type == 'phases':
    error_train_ave = utils.average_normalised_rms_error_flat(phase_exact_flat_train, phase_retrieved_flat_train)
    print("Average accuracy on ", "training", "-set: {0: .1%}".format(error_train_ave), sep='')
    f.write("Average accuracy on training-set: {0: .1%}".format(error_train_ave) + '\n')


# Create arrays to hold flattened test data
phase_exact_flat_test = []
phase_retrieved_flat_test = []
image_flat_test = []

# Compute retrieved test phases and flatten test data
for item in range(num_test):
    system_test = phase.PhaseImagingSystem(image_size=img_size,
                                           defocus=8e-6,
                                           image_width=150e-9,
                                           energy=300e3,
                                           specimen_file=specimen_files[num_train + item],
                                           mip=mip,
                                           is_attenuating=True,
                                           noise_level=noise_level,
                                           use_multislice=use_multislice,
                                           M=M)
    system_test.generate_images()
    system_test.apodise_images(imaging_parameters['Window Function Radius'])
    system_test.retrieve_phase()
    system_test.apodise_phases(imaging_parameters['Window Function Radius'])
    phase_exact_flat_test.append(system_test.phase_exact.real.reshape(img_size_flat))
    phase_retrieved_flat_test.append(system_test.phase_retrieved.real.reshape(img_size_flat))
    if input_type == 'images':
        image_flat_test.append(np.concatenate((system_test.image_under.real.reshape(img_size_flat),
                               system_test.image_in.real.reshape(img_size_flat),
                               system_test.image_over.real.reshape(img_size_flat))))



# Calculate and print average normalised rms error in test set prior to processing
# through neural network
error_pre_adj = utils.average_normalised_rms_error_flat(phase_exact_flat_test, phase_retrieved_flat_test)
print("Accuracy on test set (pre adjustment): {0: .1%}".format(error_pre_adj))
f.write("Accuracy on test set (pre adjustment): {0: .1%}".format(error_pre_adj) + '\n')

# Determine number of nodes in input layer
if input_type == 'images':
    input_size = 3 * img_size_flat
    num_channels = 3
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
num_hidden_layers = hyperparameters['Number of Hidden Layers']
hidden_layers = []
for i in range(num_hidden_layers):
    hidden_layers.append(tf.Variable(tf.zeros([hidden_layer_size])))
for i in range(num_hidden_layers):
    if i == 0:
        hidden_input = input_for_first_fc_layer
        hidden_input_size = num_inputs_for_first_fc_layer
    else:
        hidden_input = hidden_layers[i - 1]
        hidden_input_size = hidden_layer_size
    hidden_layers[i] = new_fc_layer(hidden_input,
                                    hidden_input_size,
                                    hidden_layer_size,
                                    activation_function=tf.nn.tanh,
                                    init_type='identity')

if num_hidden_layers == 0:
    penultimate_layer = input_for_first_fc_layer
    penultimate_layer_size = num_inputs_for_first_fc_layer
else:
    penultimate_layer = hidden_layers[num_hidden_layers - 1]
    penultimate_layer_size = hidden_layer_size

output = new_fc_layer(penultimate_layer,
                      penultimate_layer_size,
                      img_size_flat,
                      activation_function=None,
                      init_type='identity'
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
else:
    raise ValueError('Unknown optimizer type')

# Initialise variables
session = tf.Session()
session.run(tf.global_variables_initializer())

# Define feed dict for test set
if input_type == 'images':
    feed_dict_test = {x: image_flat_test,
                      y_true: phase_exact_flat_test
                      }
elif input_type == 'phases':
    feed_dict_test = {x: phase_retrieved_flat_test,
                      y_true: phase_exact_flat_test
                      }

# Set batch variables
batch_size = hyperparameters['Batch Size']  # Number of training examples in each batch
num_batches = int(np.floor(num_train / batch_size))  # Calculate number of batches

# Calculate the mean of the training data for later use
mean_exact_train = np.mean(phase_exact_flat_train, axis=0)

# Store exact and reconstruced examples of training data for later use
phase_exact_flat_train_0 = phase_exact_flat_train[1]
phase_retrieved_flat_train_0 = phase_retrieved_flat_train[1]

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
        if input_type == 'images':
            del image_flat_train[0:batch_size]
        #del phase_exact_flat_train[0:batch_size]
        #del phase_retrieved_flat_train[0:batch_size]

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
error_test_vs_train = 0
for output_image in output_images:
    error_test_vs_train += np.sqrt(
                        np.sum(np.square(mean_exact_train -
                                         output_image)) / np.sum(np.square(mean_exact_train))
                )
error_test_vs_train /= num_test
print("Accuracy on ", "test input", " compared to training output: {0: .1%}".format(error_test_vs_train), sep='')
f.write("Accuracy on test input compared to training output: {0: .1%}".format(error_test_vs_train) + '\n')
# result_for_printing = []
# result_for_printing.append(plot.PrintableData(np.reshape(phase_exact_flat_train[0], img_shape),
#                                               'training example',
#                                               'phase'))
# result_for_printing.append(plot.PrintableData(np.reshape(phase_retrieved_flat_train[0], img_shape),
#                                               'training example (retrieved)',
#                                               'phase',
#                                               comment="{0: .1%} (ave'd)".format(error_train),))
# result_for_printing.append(plot.PrintableData(np.reshape(phase_exact_flat_train[1], img_shape),
#                                               'training example',
#                                               'phase'))
# result_for_printing.append(plot.PrintableData(np.reshape(phase_retrieved_flat_train[1], img_shape),
#                                               'training example (retrieved)',
#                                               'phase',
#                                               comment="{0: .1%} (ave'd)".format(error_train),))
# result_for_printing.append(plot.PrintableData(np.reshape(phase_exact_flat_train[2], img_shape),
#                                               'training example',
#                                               'phase'))
# result_for_printing.append(plot.PrintableData(np.reshape(phase_retrieved_flat_train[2], img_shape),
#                                               'training example (retrieved)',
#                                               'phase',
#                                               comment="{0: .1%} (ave'd)".format(error_train),))
# result_for_printing.append(plot.PrintableData(np.reshape(phase_exact_flat_train[3], img_shape),
#                                               'training example',
#                                               'phase'))
# result_for_printing.append(plot.PrintableData(np.reshape(phase_retrieved_flat_train[3], img_shape),
#                                               'training example (retrieved)',
#                                               'phase',
#                                               comment="{0: .1%} (ave'd)".format(error_train),))
# result_for_printing.append(plot.PrintableData(np.reshape(phase_exact_flat_train[4], img_shape),
#                                               'training example',
#                                               'phase'))
# result_for_printing.append(plot.PrintableData(np.reshape(phase_retrieved_flat_train[4], img_shape),
#                                               'training example (retrieved)',
#                                               'phase',
#                                               comment="{0: .1%} (ave'd)".format(error_train),))
# result_for_printing.append(plot.PrintableData(np.reshape(phase_exact_flat_train[5], img_shape),
#                                               'training example',
#                                               'phase'))
# result_for_printing.append(plot.PrintableData(np.reshape(phase_retrieved_flat_train[5], img_shape),
#                                               'training example (retrieved)',
#                                               'phase',
#                                               comment="{0: .1%} (ave'd)".format(error_train),))
# # result_for_printing.append(plot.PrintableData(np.reshape(mean_exact_train, img_shape),
# #                                               'mean training example',
# #                                               'phase'))
# result_for_printing.append(plot.PrintableData(np.reshape(phase_exact_flat_test[0], img_shape),
#                                               'test example',
#                                               'phase'))
# result_for_printing.append(plot.PrintableData(np.reshape(phase_retrieved_flat_test[0], img_shape),
#                                               'test example (retrieved)',
#                                               'phase',
#                                               comment="{0: .1%}".format(error_pre_adj)))
# result_for_printing.append(plot.PrintableData(output_images[0].reshape(img_shape),
#                                               'test example (ret_adj)',
#                                               'phase',
#                                               comment="{0: .1%}".format(acc)))
#
#
#
error_ret = (np.array(phase_retrieved_flat_test) - np.array(phase_exact_flat_test)).tolist()
error_adj = (np.array(output_images) - np.array(phase_exact_flat_test)).tolist()

for i in range(5):
    plot.save_image(np.reshape(phase_exact_flat_train[i], img_shape),
                    './data/figures/phase_exact_train_' + str(i) + '.png', 'phase')
    plot.save_image(np.reshape(phase_retrieved_flat_train[i], img_shape),
                    './data/figures/phase_retrieved_train_' + str(i) + '.png', 'phase')
    plot.save_image(np.reshape(phase_exact_flat_test[i], img_shape),
                    './data/figures/phase_exact_test_' + str(i) + '.png', 'phase')
    plot.save_image(np.reshape(phase_retrieved_flat_test[i], img_shape),
                    './data/figures/phase_retrieved_test_' + str(i) + '.png', 'phase')
    plot.save_image(np.reshape(output_images[i], img_shape),
                    './data/figures/phase_adjusted_test_' + str(i) + '.png', 'phase')
    plot.save_image(np.reshape(error_ret[i], img_shape),
                    './data/figures/error_retrieved_test_' + str(i) + '.png', 'error')
    plot.save_image(np.reshape(error_adj[i], img_shape),
                    './data/figures/error_adjusted_test_' + str(i) + '.png', 'error')



for i in range(num_test):
    error_test = utils.normalised_rms_error(phase_exact_flat_test[i], phase_retrieved_flat_test[i])
    f.write("Accuracy on test input " + str(i) + ": {0: .1%}".format(error_test) + '\n')

accuracy = tf.sqrt(tf.reduce_sum(tf.squared_difference(y_true, output), 1) / tf.reduce_sum(tf.square(y_true), 1))
acc, x_val = session.run([accuracy, x], feed_dict=feed_dict_test)
for i, output_image in enumerate(output_images):
    f.write("Accuracy on test input " + str(i) + "(adjusted): {0: .1%}".format(acc[i]) + '\n')

for i in range(num_train):
    error_train = utils.normalised_rms_error(phase_exact_flat_train[i], phase_retrieved_flat_train[i])
    f.write("Accuracy on training input " + str(i) + ": {0: .1%}".format(error_train) + '\n')





f.close()
# Plot images
#plot.plot_images(result_for_printing)
#utils.beep()  # Alert user that script has finished
show()  # Prevent plt.show(block=False) from closing plot window
