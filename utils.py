import numpy as np
import scipy

def normalised_rms_error(exact, reconstructed):

    """
    Calculate normalised rms error between exact and reconstructed
    arrays of arbitrary (but equal) dimensions
    :param exact:
    :param reconstructed:
    :return:
    """

    assert np.shape(exact) == np.shape(reconstructed)
    total = 0
    norm = 0

    len_flat = np.prod(np.shape(exact), dtype=int)
    exact = exact.reshape(len_flat)
    reconstructed = reconstructed.reshape(len_flat)

    for i in range(len_flat):
        total += (exact[i] - reconstructed[i]) * (exact[i] - reconstructed[i])
        norm += exact[i] * exact[i]
    return np.sqrt(total / norm), np.sqrt(norm)


def average_normalised_rms_error_flat(exact_flat, reconstructed_flat):

    """
    Returns the average normalised rms error for collection of n examples. Exact
    input images must be pre-flattened, with each row being a different example
    image. Reconstructed image must be a vector representing a single result

    :param exact_flat: 2D array of n exact images
    :param reconstructed_flat: 2D array of n reconstructed images
    :return: averaged error
    """

    assert np.shape(exact_flat) == np.shape(reconstructed_flat)

    n_examples = len(exact_flat)
    error = np.zeros(n_examples)
    rms = np.zeros(n_examples)
    for i in range(n_examples):
        error[i], rms[i] = normalised_rms_error(exact_flat[i],
                                      reconstructed_flat[i])
    output = [np.mean(error), np.std(error, ddof=0), np.mean(rms), np.std(rms, ddof=0)]
    return output


def beep():
    import winsound
    freq = 440  # Set Frequency To 2500 Hertz
    dur = 200  # Set Duration To 1000 ms == 1 second
    winsound.Beep(freq, dur)


def write_dict(file, dict):
    for key, value in dict.items():
        file.write('%s:%s\n' % (key, value) + '\n')
    return


def import_micrograph(micrograph_file, pix):

    with open(micrograph_file, 'r') as f:
        f.seek(0)
        image_temp = np.zeros((pix, pix), dtype=complex)

        for i in range(pix):
            line = f.readline().split()
            for j in range(pix):
                image_temp[i, j] = float(line[j])
    return image_temp

def truncated_gaussian(a, b, mu, sigma):
    return scipy.stats.truncnorm((a - mu)/sigma, (b - mu)/sigma, loc=mu, scale=sigma)
