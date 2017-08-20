import numpy as np


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
    return np.sqrt(total / norm)


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
    error = 0
    n_examples = len(exact_flat)
    for i in range(n_examples):
        error += normalised_rms_error(exact_flat[i],
                                      reconstructed_flat[i])
    error /= n_examples
    return error


def beep():
    import winsound
    freq = 440  # Set Frequency To 2500 Hertz
    dur = 200  # Set Duration To 1000 ms == 1 second
    winsound.Beep(freq, dur)


def write_dict(file, dict):
    for key, value in dict.items():
        file.write('%s:%s\n' % (key, value) + '\n')
    return
