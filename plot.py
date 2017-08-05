import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.colors

def plot_images(images, cls_true, img_shape, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create a figure with 3x3 subplots.
    fig, axes = plt.subplots(3, 3)
    #fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove tick marks from plot
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show(block=False)


def plot_images_(images, labels, types):

    size = int(np.ceil(np.sqrt(len(images))))
    fig, axes = plt.subplots(size, size)
    #fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image
        if i < len(images):
            if types[i] == 'image':
                vmin = 0
                vmax = 2
                ax.imshow(images[i], cmap='gray', vmin=vmin, vmax=vmax)
            elif types[i] == 'phase':
                vmin = -4
                vmax = 4
                ax.imshow(images[i], cmap='gray', vmin=vmin, vmax=vmax)
            else:
                ax.imshow(images[i], cmap='gray')
            ax.set_xlabel(labels[i])


        # Remove tick marks from plot
        ax.set_xticks([])
        ax.set_yticks([])


    plt.show(block=False)


def plot_image(image, type=None):
    #assert type == 'image' or type == 'phase'
    if type == 'image':
        vmin = 0
        vmax = 2
        plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    elif type == 'phase':
        vmin = -3
        vmax = 3
        plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    else:
        plt.imshow(image, cmap='gray')
    plt.show(block=True)


def print_confusion_matrix(plot):
    cls_true = data.test.cls

    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
    if plot:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        plt.tight_layout()
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show(block=False)
    else:
        return


def multiplot(plot1,  plot2, plot3, plot4):
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(x, y)
    #axarr[0, 0].set_title('Axis [0,0]')
    axarr[0, 1].scatter(x, y)
    #axarr[0, 1].set_title('Axis [0,1]')
    axarr[1, 0].plot(x, y ** 2)
    #axarr[1, 0].set_title('Axis [1,0]')
    axarr[1, 1].scatter(x, y ** 2)
    #axarr[1, 1].set_title('Axis [1,1]')
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    plt.show(block=False)

def show():
    plt.show()
