import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_images(data):

    size = int(np.ceil(np.sqrt(len(data))))
    fig, axes = plt.subplots(size, size)

    for i, ax in enumerate(axes.flat):
        # Plot image
        if i < len(data):
            if data[i].type == 'image':
                vmin = 0
                vmax = 2
                ax.imshow(data[i].image, cmap='gray', vmin=vmin, vmax=vmax)
            elif data[i].type == 'phase':
                vmin = -4
                vmax = 4
                ax.imshow(data[i].image, cmap='gray', vmin=vmin, vmax=vmax)
            elif data[i].type == 'error':
                vmin = -1
                vmax = 1
                ax.imshow(data[i].image, cmap='gray', vmin=vmin, vmax=vmax)
            else:
                ax.imshow(data[i].image, cmap='gray')
            ax.set_xlabel(data[i].title)
            ax.set_ylabel(data[i].comment)

        # Remove tick marks from plot
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show(block=False)


def plot_images_(images, labels, types):

    size = int(np.ceil(np.sqrt(len(images))))
    fig, axes = plt.subplots(size, size)

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


def save_image(image, output_path, type=None):

    fig = plt.figure()
    fig.set_size_inches(1, 1)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    if type == 'image':
        vmin = 0
        vmax = 2
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
    elif type == 'phase':
        vmin = -3
        vmax = 3
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
    elif type == 'error':
        vmin = -1
        vmax = 1
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
    else:
        ax.imshow(image, cmap='gray', aspect='auto', interpolation='none')
    # Remove tick marks from plot

    plt.savefig(output_path, dpi=800)
    plt.close()


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

class PrintableData(object):

    def __init__(self, image, title, type, comment=''):

        self.image = image
        self.title = title
        self.type = type
        self.comment = comment
