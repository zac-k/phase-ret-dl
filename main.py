
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

np.set_printoptions(threshold=np.inf)

# Set image size and shape
img_size = 64
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

# Set size of hidden layer and whether it is used
hidden_layer_size = 10000
is_single_layer = False

# Set mean inner potential and noise level
mip = -17 - 0.8j
noise_level = 0.05

# Create arrays to hold flattened training data
phase_exact_flat_train = []
phase_retrieved_flat_train = []

# Set quantities for training/validation/test split
num_train = 1224
num_valid = 0
num_test = 1

# Import specimen files
specimen_files = []
specimen_path = './data/specimens/training2/'
specimen_ext = '.txt'
specimen_name = 'particle'
for i in range(num_train + num_test):
    specimen_files.append(specimen_path + specimen_name + '(' + str(i) + ')' + specimen_ext)

# Compute retrieved training phases and flatten training data
for item in range(num_train):
    specimen_file = specimen_files[np.random.randint(len(specimen_files))]
    system_train = phase.PhaseImagingSystem(
           image_size=img_size,
           defocus=8e-6,
           image_width=150e-9,
           energy=300e3,
           specimen_file=specimen_files[item],
           mip=mip,
           is_attenuating=True,
           noise_level=noise_level)
    system_train.generate_images()
    system_train.apodise_images()
    system_train.retrieve_phase()
    phase_exact_flat_train.append(system_train.phase_exact.real.reshape(img_size_flat))
    phase_retrieved_flat_train.append(system_train.phase_retrieved.real.reshape(img_size_flat))


# Create arrays to hold flattened test data
phase_exact_flat_test = []
phase_retrieved_flat_test = []

# Compute retrieved test phases and flatten test data
for item in range(num_test):
    system_test = phase.PhaseImagingSystem(image_size=img_size,
                                           defocus=8e-6,
                                           image_width=150e-9,
                                           energy=300e3,
                                           specimen_file=specimen_files[num_train + item],
                                           mip=mip,
                                           is_attenuating=True,
                                           noise_level=noise_level
                                           )
    system_test.generate_images()
    system_test.apodise_images()
    system_test.retrieve_phase()
    phase_exact_flat_test.append(system_test.phase_exact.real.reshape(img_size_flat))
    phase_retrieved_flat_test.append(system_test.phase_retrieved.real.reshape(img_size_flat))

# Delete imaging systems now that they are no longer needed
del system_train
del system_test

# Calculate and print average normalised rms error in test set prior to processing
# through neural network
error = utils.average_normalised_rms_error_flat(phase_exact_flat_test, phase_retrieved_flat_test)
print("Accuracy on test set (pre adjustment): {0: .1%}".format(error))

# Define placeholder variables
x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, img_size_flat])

# Define layers
hidden_layer = new_fc_layer(x,
                            img_size_flat,
                            hidden_layer_size,
                            activation_function=tf.nn.tanh,
                            init_type='identity')
if is_single_layer:
    penultimate_layer = x
    penultimate_layer_size = img_size_flat
else:
    penultimate_layer = hidden_layer
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
optimizer_type = 'gradient descent'
if optimizer_type == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
elif optimizer_type == 'gradient descent':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
else:
    raise ValueError('Unknown optimizer type')

# Initialise variables
session = tf.Session()
session.run(tf.global_variables_initializer())

# Define feed dict for test set
feed_dict_test = {x: phase_retrieved_flat_test,
                  y_true: phase_exact_flat_test
                  }

# Set batch variables
batch_size = 30  # Number of training examples in each batch
num_batches = int(np.floor(num_train / batch_size))  # Calculate number of batches

# Calculate the mean of the training data for later use
mean_exact_train = np.mean(phase_exact_flat_train, axis=0)

# Store exact and reconstruced examples of training data for later use
phase_exact_flat_train_0 = phase_exact_flat_train[0]
phase_retrieved_flat_train_0 = phase_retrieved_flat_train[0]

# Train the model
print('Training...')
for i in range(num_batches):
    print('{0: .1%}'.format(i / num_batches)),  # Avoid new line
    if (i + 1) * batch_size < num_train:
        x_batch = phase_retrieved_flat_train[0:batch_size - 1]
        y_true_batch = phase_exact_flat_train[0:batch_size - 1]
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)
    del phase_exact_flat_train[i * batch_size:(i + 1) * batch_size - 1]
    del phase_retrieved_flat_train[i * batch_size:(i + 1) * batch_size - 1]


# Calculate and print average normalised rms error in test set after processing through
# trained neural network
error = tf.sqrt(tf.reduce_sum(tf.squared_difference(y_true, output), 1) / tf.reduce_sum(tf.square(y_true), 1))
accuracy = tf.reduce_mean(error)
acc, x_val = session.run([accuracy, x], feed_dict=feed_dict_test)
print("Accuracy on ", "test", "-set (post-adjustment): {0: .1%}".format(acc), sep='')

# Define average error in training set, calculate it, and print output.
error = tf.sqrt(tf.reduce_sum(tf.squared_difference(y_true, x), 1) / tf.reduce_sum(tf.square(y_true), 1))
accuracy = tf.reduce_mean(error)
acc, output_image_train = session.run([accuracy, output], feed_dict=feed_dict_train)
print("Accuracy on ", "training", "-set: {0: .1%}".format(acc), sep='')

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

# Plot images
plot.plot_images_([np.reshape(phase_exact_flat_train_0, img_shape),
                   np.reshape(phase_retrieved_flat_train_0, img_shape),
                   np.reshape(mean_exact_train, img_shape),
                   np.reshape(phase_exact_flat_test[0], img_shape),
                   np.reshape(phase_retrieved_flat_test[0], img_shape),
                   output_images[0].reshape(img_shape)],
                   ['training example',
                    'training example (retrieved)',
                    'mean training example',
                    'test example',
                    'test example (retrieved)',
                    'test example (ret_adj)'],
                   ['phase',
                    'phase',
                    'phase',
                    'phase',
                    'phase',
                    'phase'])
utils.beep()  # Alert user that script has finished
show()  # Prevent plt.show(block=False) from closing plot window
