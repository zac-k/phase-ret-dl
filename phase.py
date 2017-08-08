import numpy as np
from numpy import fft
from numpy import pi as PI
import copy
import plot
from scipy.ndimage.filters import gaussian_filter


h = 6.63e-34  # Planck's constant
m0 = 9.11e-31  # Electron rest mass
e = -1.6e-19  # Electron charge
c = 3e8  # Speed of light
hbar = h / (2 * PI)


class PhaseImagingSystem(object):

    def __init__(self, image_size, defocus, image_width, energy,
                 specimen_file, mip, is_attenuating, noise_level):

        # Initialise image/micrograph arrays
        self.image_under = np.zeros(list((image_size, image_size)), dtype=complex)
        self.image_over = np.zeros(list((image_size, image_size)), dtype=complex)
        self.image_in = np.zeros(list((image_size, image_size)), dtype=complex)

        # Initialise phase arrays
        self.phase_exact = np.zeros(list((image_size, image_size)), dtype=complex)
        self.phase_retrieved = np.zeros(list((image_size, image_size)), dtype=complex)

        # Initialise intensity derivative (used for troubleshooting purposes)
        self.derivative = np.zeros(list((image_size, image_size)), dtype=complex)

        # Set initialisations
        self.image_size = image_size
        self.defocus = defocus
        self.image_width = image_width
        self.energy = energy
        self.is_attenuating = is_attenuating
        self.image_intensity = 1
        self.noise_level = noise_level

        if is_attenuating:
            self.mip = mip
        else:
            self.mip = mip.real


        # Calculate wavelength from electron energy
        self.wavelength = (h / np.sqrt(2 * m0 * np.abs(e) * energy)) * (1 / np.sqrt(1 + np.abs(e) * energy / (2 * m0 * c * c)))

        #Initialise transfert function and specimen attributes
        self.transfer_function = None
        self.specimen = None
        self.specimen_size = None

        # Construct specimen from file, project phases, and
        # downsample phase to the system's image size
        self._construct_specimen(specimen_file)
        self._project_phase(orientation=0)

        while len(self.phase_exact) > image_size:

            self.phase_exact = self._downsample(self.phase_exact)

        # Set regularisation parameters and construct kernels
        self.reg_tie = 0.1 / (self.image_width * self.image_size)
        self.reg_image = 0.1
        if is_attenuating:
            self.reg_tie /= 2
        self.k_squared_kernel = self._construct_k_squared_kernel()
        self.inverse_k_squared_kernel = self._construct_inverse_k_squared_kernel()
        self.k_kernel = self._construct_k_kernel()

    def _add_noise(self, image):
        for i in range(len(image)):
            for j in range(len(image[0])):
                image[i, j] = np.random.poisson(image[i, j] /
                                  (self.noise_level *
                                   self.noise_level *
                                   self.image_intensity)) * (self.noise_level *
                                                             self.noise_level *
                                                             self.image_intensity)




    def _construct_inverse_k_squared_kernel(self):

        kernel = np.zeros(list((self.image_size, self.image_size)), dtype=complex)
        da = 1 / self.image_width
        for i in range(self.image_size):
            for j in range(self.image_size):
                i0 = i - self.image_size / 2
                j0 = j - self.image_size / 2

                kernel[i, j] = (i0 * i0 * da * da) + (j0 * j0 * da * da)
        kernel = kernel / (kernel * kernel + self.reg_tie ** 4)
        return kernel

    def _construct_k_squared_kernel(self):

        kernel = np.zeros(list((self.image_size, self.image_size)), dtype=complex)
        for i in range(self.image_size):
            for j in range(self.image_size):
                i0 = i - self.image_size / 2
                j0 = j - self.image_size / 2
                da = 1 / self.image_width
                kernel[i, j] = (i0 * i0) * da * da + j0 * j0 * da * da

        return kernel

    def _construct_k_kernel(self):
        kernel = np.zeros(list((self.image_size, self.image_size, 2)), dtype=complex)
        for i in range(self.image_size):
            for j in range(self.image_size):
                i0 = i - self.image_size / 2
                j0 = j - self.image_size / 2
                da = 1 / self.image_width
                kernel[i, j, 0] = i0 * da
                kernel[i, j, 1] = j0 * da
        return kernel

    def _construct_specimen(self, specimen_file):

        with open(specimen_file, 'r') as f:
            self.specimen_size = int(len(f.readline().split()))
            f.seek(0)
            self.specimen = np.zeros((self.specimen_size, self.specimen_size, self.specimen_size))
            for k in range(self.specimen_size):
                for i in range(self.specimen_size):
                    self.specimen[i, :, k] = f.readline().split()

                f.readline()

    def _downsample(self, input):
        input_len = len(input)
        ds_len = int(input_len / 2)
        ds = np.zeros(list((ds_len, ds_len,)), dtype=complex)
        for i in range(ds_len):
            for j in range(ds_len):
                sum_ = 0
                for p in range(2):
                    for q in range(2):
                        sum_ += input[2*i+p, 2*j+q]
                ds[i, j] = sum_ / 4

        return ds

    def _project_phase(self, orientation):
        """
        Computes the image plane phase shift of the specimen.

        :param self:
        :param orientation: either 0, 1, or 2, describes the axis of projection
        :return:
        """
        energy = self.energy
        wavelength = self.wavelength
        dz = self.image_width / self.specimen_size
        self.phase_exact = np.sum(self.specimen,
                                  axis=orientation) * PI/(energy * wavelength) * self.mip * dz

    def _set_transfer_function(self, defocus):
        """
        Sets the imaging system's transfer function for a given
        defocus.

        :param self:
        :param defocus:
        :return:
        """
        wavelength = self.wavelength
        self.transfer_function = np.exp(1j * (PI * defocus * wavelength * self.k_squared_kernel))
        return

    def _transfer_image(self, defocus):
        """
        Uses the defocus exact_phase (at the image plane) to produce an out of focus
        image.

        :param self:
        :param defocus:
        :return:
        """
        self._set_transfer_function(defocus=defocus)
        wavefunction = np.exp(-1j * self.phase_exact)
        wavefunction = fft.fft2(wavefunction)
        wavefunction = fft.fftshift(wavefunction)
        wavefunction = self.transfer_function * wavefunction
        wavefunction = fft.ifftshift(wavefunction)
        wavefunction = fft.ifft2(wavefunction)
        return np.absolute(wavefunction) * np.absolute(wavefunction)

    def generate_images(self):
        """
        Compute images at under-, in-, and over-focus
        :return:
        """
        self.image_over = self._transfer_image(defocus=-self.defocus)
        self.image_under = self._transfer_image(defocus=self.defocus)
        self.image_in = self._transfer_image(defocus=0)

        if self.noise_level != 0:
            for image in [self.image_under,
                          self.image_in,
                          self.image_over]:
                image = self._add_noise(image)

        return

    @staticmethod
    def apodise(image, rad_sup=0.3):
        output = copy.copy(image)
        for i in range(len(image)):
            for j in range(len(image[0])):
                i0 = i - len(image) / 2
                j0 = j - len(image[0]) / 2
                r_sqr = i0 * i0 + j0 * j0
                if r_sqr > (rad_sup * len(image)) * (rad_sup * len(image[0])):
                    output[i, j] = 0
        return output

    @staticmethod
    def convolve(image, kernel):
        # Kernel is in Fourier space
        assert len(image) == len(kernel) and len(image[0]) == len(kernel[0])
        image = fft.fft2(image)
        image = fft.fftshift(image)
        image *= kernel
        image = fft.ifftshift(image)
        image = fft.ifft2(image)
        return image

    def intensity_derivative(self):
        return (self.image_under - self.image_over) / (2 * self.defocus)

    @staticmethod
    def regularise_and_invert(kernel, reg):
        return kernel / (kernel * kernel + reg * reg)

    @staticmethod
    def dot_fields(field1, field2):
        return field1[:, :, 0] * field2[:, :, 0] + field1[:, :, 1] * field2[:, :, 1]

    def apodise_images(self, rad_sup):
        self.image_under = self.apodise(self.image_under, rad_sup)
        self.image_in = self.apodise(self.image_in, rad_sup)
        self.image_over = self.apodise(self.image_over, rad_sup)

    def apodise_phases(self, rad_sup):
        self.phase_exact = self.apodise(self.phase_exact, rad_sup)
        self.phase_retrieved = self.apodise(self.phase_retrieved, rad_sup)

    def retrieve_phase(self):
        """
        Utilise the transport-of-intensity equation to compute the phase from
        the micrographs.
        :return:
        """

        if self.is_attenuating:
            regularised_inverse_intensity = self.image_in / (self.image_in * self.image_in
                                                             + self.reg_image * self.reg_image)
            prefactor = (1. / self.wavelength) / (2. * PI)
            derivative = self.intensity_derivative()
            self.derivative = derivative
            derivative_vec = np.zeros(list((self.image_size, self.image_size, 2)), dtype=complex)
            derivative_vec[:, :, 0] = self.convolve(derivative, self.k_kernel[:, :, 0] *
                                                    self.inverse_k_squared_kernel
                                                    ) * regularised_inverse_intensity
            derivative_vec[:, :, 1] = self.convolve(derivative, self.k_kernel[:, :, 1] *
                                                    self.inverse_k_squared_kernel
                                                    ) * regularised_inverse_intensity

            derivative_vec[:, :, 0] = fft.fftshift(fft.fft2(derivative_vec[:, :, 0]))
            derivative_vec[:, :, 1] = fft.fftshift(fft.fft2(derivative_vec[:, :, 1]))
            derivative = self.dot_fields(self.k_kernel, derivative_vec) * self.inverse_k_squared_kernel
            self.phase_retrieved = prefactor * fft.ifft2(fft.ifftshift(derivative))
            return

        else:
            prefactor = (1. / self.wavelength) / (2 * PI * self.image_intensity)

            filtered = self.convolve(self.intensity_derivative(),
                                     self.inverse_k_squared_kernel)
            self.phase_retrieved = prefactor * filtered
        return








