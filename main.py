
import numpy as np
import tensorflow as tf
import utils
from plot import show
import plot
import phase

np.set_printoptions(threshold=np.inf)












img_size = 64
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

mip = -17 - 0.8j
# specimen_files = ['./data/specimens/head/512/particle.txt',
#              './data/specimens/multicube/512/particle.txt',
#              './data/specimens/array7/512/particle.txt',
#              './data/specimens/004/512/particle.txt',
#              './data/specimens/003/512/particle.txt']



phase_exact_flat_train = []
phase_retrieved_flat_train = []
num_train = 159
num_test = 1

specimen_files = []
for i in range(num_train + num_test):
    specimen_files.append('./data/specimens/training/particle(' + str(i) + ').txt')


noise_level = 0.1

for item in range(num_train):
    #specimen_file = './data/specimens/005/512/particle.txt'
    specimen_file = specimen_files[np.random.randint(len(specimen_files))]
    system_train = phase.PhaseImagingSystem(image_size=img_size,
                                           defocus=8e-6,
                                           image_width=150e-9,
                                           energy=300e3,
                                           specimen_file=specimen_files[item],
                                           mip=mip,
                                           is_attenuating=True,
                                           noise_level=noise_level
                                           )
    system_train.generate_images()
    system_train.apodise_images()
    system_train.retrieve_phase()
    phase_exact_flat_train.append(system_train.phase_exact.real.reshape(img_size_flat))
    phase_retrieved_flat_train.append(system_train.phase_retrieved.real.reshape(img_size_flat))



phase_exact_flat_test = []
phase_retrieved_flat_test = []

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
    print(utils.normalised_rms_error(system_test.phase_exact.real, system_test.phase_retrieved.real))

acc = 0
for i in range(num_test):
    acc += utils.normalised_rms_error(np.reshape(phase_exact_flat_test[i], img_shape),
                                      np.reshape(phase_retrieved_flat_test[i], img_shape))
acc /= num_test

print("Accuracy on test set (pre adjustment): {0: .1%}".format(acc))

#     plot.plot_images_([system.phase_exact.real,
#                        system.image_under.real,
#                        system.image_in.real,
#                        system.image_over.real,
#                        system.derivative.real,
#                        system.phase_retrieved.real],
#                      ['phase',
#                       'image',
#                       'image',
#                       'image',
#                       '',
#                       'phase'])
#
#     print(utils.normalised_rms_error(system.phase_exact.real, system.phase_retrieved.real))
#     show()













#data = input_data.read_data_sets("data/MNIST/", one_hot=True)
#data.test.cls = np.array([label.argmax() for label in data.test.labels])



x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, img_size_flat])
#y_true_cls = tf.placeholder(tf.int64, [None])

weights = tf.Variable(tf.ones([img_size_flat, img_size_flat]))
biases = tf.Variable(tf.zeros([img_size_flat]))

output = tf.matmul(x, weights) + biases


#y_pred = tf.nn.softmax(logits)
#y_pred_cls = tf.argmax(y_pred, dimension=1)

#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
#                                                        labels=y_true)

cost = tf.reduce_mean(tf.squared_difference(y_true, output))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

#correct_prediction = tf.equal(y_pred_cls, y_true_cls)

error = tf.sqrt(tf.reduce_sum(tf.squared_difference(y_true, output), 1) / tf.reduce_sum(tf.square(y_true), 1))
accuracy = tf.reduce_sum(error)

session = tf.Session()
session.run(tf.global_variables_initializer())



#batch_size = 100

feed_dict_test = {x: phase_retrieved_flat_test,
                  y_true: phase_exact_flat_test
                  }



num_iterations = 10
for i in range(num_iterations):
    # x_batch, y_true_batch = data.train.next_batch(batch_size)
    x_batch = phase_retrieved_flat_train
    y_true_batch = phase_exact_flat_train
    feed_dict_train = {x: x_batch,
                       y_true: y_true_batch}

    session.run(optimizer, feed_dict=feed_dict_train)

acc, x_val = session.run([accuracy, x], feed_dict=feed_dict_test)
print("Accuracy on ", "test", "-set: {0: .1%}".format(acc), sep='')









error = tf.sqrt(tf.reduce_sum(tf.squared_difference(y_true, x), 1) / tf.reduce_sum(tf.square(y_true), 1))
accuracy = tf.reduce_mean(error)

acc, output_image = session.run([accuracy, output], feed_dict=feed_dict_train)
print("Accuracy on ", "training", "-set: {0: .1%}".format(acc), sep='')





output_image = session.run(output, feed_dict=feed_dict_test)

mean_exact_train = np.mean(phase_exact_flat_train, axis=0)
error_test_vs_train = np.sqrt(
                    np.sum(np.square(mean_exact_train -
                                     output_image)) / np.sum(np.square(mean_exact_train))
                )
print(np.shape(np.sum(np.square(phase_exact_flat_train), 1)))
print("Accuracy on ", "test input", " compared to training output: {0: .1%}".format(error_test_vs_train), sep='')




#plot.plot_image(tf.reshape(y_true[0], img_shape))
#plot.plot_image(tf.reshape(x, img_shape))

#print(x_val.reshape(img_shape))

plot.plot_images_([np.reshape(phase_exact_flat_train[0], img_shape),
                   np.reshape(phase_retrieved_flat_train[0], img_shape),
                   np.reshape(phase_exact_flat_test[0], img_shape),
                   np.reshape(phase_retrieved_flat_test[0], img_shape),
                   output_image.reshape(img_shape),
                   np.reshape(mean_exact_train, img_shape)],
                   ['training example',
                    'training example (retrieved)',
                    'test example',
                    'test example (retrieved)',
                    'test example (ret_adj)',
                    'mean training example'],
                   ['phase',
                    'phase',
                    'phase',
                    'phase',
                    'phase',
                    'phase'])


#utils.beep()

# Prevent plt.show(block=False) from closing plot window
show()