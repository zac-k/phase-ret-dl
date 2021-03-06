import numpy as np
from numpy import fft
from numpy import pi as PI
import copy
import pyprind, sys
import random as rnd
from numba import jit
from unwrap import unwrap
from scipy.ndimage import rotate, zoom, shift
np.set_printoptions(threshold=10)
import warnings


h = 6.63e-34  # Planck's constant
m0 = 9.11e-31  # Electron rest mass
mu0 = 4 * PI * 10**-7  # Free space permeability
e = -1.6e-19  # Electron charge
c = 3e8  # Speed of light
hbar = h / (2 * PI)  # Reduced Planck constant
phi0 = h / (2 * e)  # Flux quantum

npmax = 12
nzmin = 1
nzmax = 103
ncmax = 132
nl = 3
ng = 3
SMALL = 1e-25

class PhaseImagingSystem(object):

    fe_table_read = 0
    fparams = [[0 for col in range(npmax)] for row in range(nzmax + 1)]

    def __init__(self, image_size, defocus, image_width, energy,
                 specimen_file, mip, is_attenuating, noise_level, use_multislice,
                 multislice_method, M, item, path, simulation_parameters, specimen_parameters):

        self.simulation_parameters = simulation_parameters
        self.specimen_parameters = specimen_parameters

        self.flipping = False
        if specimen_parameters['Use Electrostatic/Magnetic Potential'][0] and \
                specimen_parameters['Use Electrostatic/Magnetic Potential'][1]:
            if simulation_parameters['Retrieve Phase Component'] != 'total':
                self.flipping = True
        # Initialise image/micrograph arrays
        self.image_under = np.zeros(list((image_size, image_size)), dtype=complex)
        self.image_over = np.zeros(list((image_size, image_size)), dtype=complex)
        self.image_in = np.zeros(list((image_size, image_size)), dtype=complex)
        if self.flipping:
            self.image_under_reverse = np.zeros(list((image_size, image_size)), dtype=complex)
            self.image_over_reverse = np.zeros(list((image_size, image_size)), dtype=complex)
            self.image_in_reverse = np.zeros(list((image_size, image_size)), dtype=complex)


        # Initialise phase arrays
        self.phase_exact = np.zeros(list((image_size, image_size)), dtype=complex)
        self.phase_retrieved = np.zeros(list((image_size, image_size)), dtype=complex)
        if self.flipping:
            self.phase_reverse = np.zeros(list((image_size, image_size)), dtype=complex)

        if specimen_parameters['Use Electrostatic/Magnetic Potential'][0]:
            self.phase_electrostatic = np.zeros((image_size, image_size), dtype=complex)
        if specimen_parameters['Use Electrostatic/Magnetic Potential'][1]:
            self.phase_magnetic = np.zeros((image_size, image_size), dtype=complex)

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
        self.M = M
        self.lowest_integrated_intensity = 1
        self.wave_multislice_hr = []
        self.wave_multislice = []
        self.atom_locations = []
        self.use_multislice = use_multislice
        self.nbeams = 0
        self.projected_potential_size = self.M

        if is_attenuating:
            self.mip = mip
        else:
            self.mip = mip.real
        self.magnetisation = specimen_parameters['Mass Magnetization'] * specimen_parameters['Density'] * 1000  # A/m


        # Calculate wavelength from electron energy
        self.wavelength = (h / np.sqrt(2 * m0 * np.abs(e) * energy)) * (1 / np.sqrt(1 + np.abs(e) * energy / (2 * m0 * c * c)))

        #Initialise transfert function and specimen attributes
        self.transfer_function = None
        self.specimen = None
        self.specimen_size = None

        # Set regularisation parameters and construct kernels
        self.reg_tie = 0.1 / (self.image_width * self.image_size)
        self.reg_image = 0.1
        self.k_squared_kernel = self._construct_k_squared_kernel()
        self.inverse_k_squared_kernel = self._construct_inverse_k_squared_kernel()
        self.k_kernel = self._construct_k_kernel()

        if use_multislice and multislice_method == 'files':

            self.wave_multislice = np.conj(self._import_wavefield(path + 'image(' +
                                                        str(item) + ')', self.projected_potential_size))
            self.phase_exact = PhaseImagingSystem.extract_phase_from_wavefield(self.wave_multislice)
        else:
            # Construct specimen from file, project phases, and
            # downsample phase to the system's image size
            self._construct_specimen(specimen_file)
            self._project_phase()

        while len(self.phase_exact) > image_size:
            self.phase_exact = PhaseImagingSystem._downsample(self.phase_exact)
        while len(self.phase_exact) < image_size:
            self.phase_exact = PhaseImagingSystem._upsample(self.phase_exact)


        if use_multislice:
            if multislice_method == 'files':
                self.wave_multislice_hr = np.conj(self._import_wavefield(path + 'image_hr(' +
                                                        str(item) + ')', self.M))
            else:
                self.atom_locations = self.build_atom_locations()
                random_axis = PhaseImagingSystem.generate_random_axis()
                random_angle = rnd.uniform(0, 2 * PI)
                self.wave_multislice_hr = self.project_phase_ms(
                                             axis=random_axis,
                                             angle=random_angle)

    @staticmethod
    def scale(img, zoom_factor, **kwargs):
        """
        Courtesy of ali_m on StackOverflow
        https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
        """

        h, w = img.shape[:2]
        # width and height of the zoomed image

        # for multichannel images we don't want to apply the zoom factor to the RGB
        # dimension, so instead we create a tuple of zoom factors, one per array
        # dimension, with 1's for any trailing dimensions after the width and height.
        zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

        # zooming out
        if zoom_factor < 1:
            zh = int(np.round(zoom_factor * h))
            zw = int(np.round(zoom_factor * w))
            # bounding box of the clip region within the output array
            top = (h - zh) // 2
            left = (w - zw) // 2
            # pad with intensity of unity
            out = np.ones_like(img)
            out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, **kwargs)

        # zooming in
        elif zoom_factor > 1:
            # bounding box of the clip region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            out = zoom(img[top:top + zh +1, left:left + zw + 1], zoom_tuple, **kwargs)

            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = ((out.shape[0] - h) // 2)
            trim_left = ((out.shape[1] - w) // 2)
            out = out[trim_top:trim_top + h, trim_left:trim_left + w]

        # if zoom_factor == 1, just return the input array
        else:
            out = img


        return out
    def rotate_images(self, std, mode, n_images):
        if n_images == 3:
            if mode == 'uniform':
                angle = np.random.uniform(0, std)
            elif mode == 'gaussian':
                angle = np.random.normal(scale=std)
            self.image_under = rotate(input=self.image_under, angle=angle, reshape=False, cval=1.0)
        if mode == 'uniform':
            angle = np.random.uniform(0, std)
        elif mode == 'gaussian':
            angle = np.random.normal(scale=std)
        self.image_over = rotate(input=self.image_over, angle=angle, reshape=False, cval=1.0)
        if self.flipping:
            if n_images == 3:
                if mode == 'uniform':
                    angle = np.random.uniform(0, std)
                elif mode == 'gaussian':
                    angle = np.random.normal(scale=std)
                self.image_in_reverse = rotate(input=self.image_in_reverse, angle=angle, reshape=False, cval=1.0)
            if mode == 'uniform':
                angle = np.random.uniform(0, std)
            elif mode == 'gaussian':
                angle = np.random.normal(scale=std)
            self.image_under_reverse = rotate(input=self.image_under_reverse, angle=angle, reshape=False, cval=1.0)
            if mode == 'uniform':
                angle = np.random.uniform(0, std)
            elif mode == 'gaussian':
                angle = np.random.normal(scale=std)
            self.image_over_reverse = rotate(input=self.image_over, angle=angle, reshape=False, cval=1.0)
        return angle

    def scale_images(self, mode, std, n_images):
        if mode == 'gaussian':
            factor = np.random.normal(loc=1.0, scale=std)
        elif mode == 'uniform':
            factor = np.random.uniform(1.0-std, 1+std)
        if n_images == 3:
            self.image_under = PhaseImagingSystem.scale(self.image_under, factor, cval=1.0)
        self.image_over = PhaseImagingSystem.scale(self.image_over, factor, cval=1.0)
        if self.flipping:
            if n_images == 3:
                factor = np.random.normal(loc=1.0, scale=std)
                self.image_in_reverse = PhaseImagingSystem.scale(self.image_in_reverse, factor, cval=1.0)
            factor = np.random.normal(loc=1.0, scale=std)
            self.image_under_reverse = PhaseImagingSystem.scale(self.image_under_reverse, factor, cval=1.0)
            factor = np.random.normal(loc=1.0, scale=std)
            self.image_over_reverse = PhaseImagingSystem.scale(self.image_over_reverse, factor, cval=1.0)
        return factor

    @staticmethod
    def _random_vector(std, mode):

        angle = np.random.uniform(0, 2 * PI)

        if mode == 'uniform':
            r = np.random.uniform(0, std)
        elif mode == 'gaussian':
            r = np.abs(np.random.normal(scale=std))

        disp = [r * np.cos(angle), r * np.sin(angle)]
        return disp

    def shift_images(self, mode, std, n_images):
        disp = PhaseImagingSystem._random_vector(std=std, mode=mode)
        if n_images == 3:
            self.image_under = shift(input=self.image_under, shift=disp, cval=1.0)
        self.image_over = shift(input=self.image_over, shift=disp, cval=1.0)
        if self.flipping:
            if n_images == 3:
                disp = PhaseImagingSystem._random_vector(std)
                self.image_in_reverse = shift(input=self.image_in_reverse, shift=disp, cval=1.0)
            disp = PhaseImagingSystem._random_vector(std)
            self.image_under_reverse = shift(input=self.image_under_reverse, shift=disp, cval=1.0)
            disp = PhaseImagingSystem._random_vector(std)
            self.image_over_reverse = shift(input=self.image_over_reverse, shift=disp, cval=1.0)
        return disp

    @staticmethod
    def extract_phase_from_wavefield(wave):
        """
        Computes the unwrapped phase from a 2D complex-valued
        wavefunction. Assumes that the phase is equal on opposite
        sides of the phase map, but this can be changed by setting
        the wrap_around_axis arguments to False, which may be
        required if the object is not centred.
        :param wave:
        :return:
        """
        # Calculate wrapped phase from wavefield
        output = np.angle(wave)
        # Unwrap phase
        output = unwrap(output, wrap_around_axis_0=True,
                        wrap_around_axis_1=True)
        return output

    @staticmethod
    @jit
    def _upsample(input_image):
        nx = len(input_image)
        ny = len(input_image[0])
        output = np.zeros((2 * nx, 2 * ny), dtype=complex)
        for i in range(nx - 1):
            for j in range(ny - 1):
                output[2 * i, 2 * j] = input_image[i, j]
                output[2 * i + 1, 2 * j] = (input_image[i, j] + input_image[i + 1, j]) / 2
                output[2 * i, 2 * j + 1] = (input_image[i, j] + input_image[i, j + 1]) / 2
                output[2 * i + 1, 2 * j + 1] = (input_image[i, j] +
                                                input_image[i + 1, j] +
                                                input_image[i, j + 1] +
                                                input_image[i + 1, j + 1]) / 4
        return output




    @staticmethod
    def generate_random_axis():
        phi = rnd.uniform(0, 2 * PI)
        theta = rnd.uniform(0, PI)

        return np.array([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)])

    @jit
    def build_atom_locations(self):
        """
        Create a list containing atom locations and their associated atomic numbers.
        :return: location_list
        """

        # Size of unit cell in angstroms
        Sc = [8.32, 8.32, 8.32]

        # Length of each side of cubic region containing specimen
        aA = self.image_width * 1e10

        # Number of unit cells in each dimension
        Nc = ([int(aA/Sc[0]), int(aA/Sc[1]), int(aA/Sc[2])])

        # Number of voxels across one dimension
        M = len(self.specimen)

        atoms_read = 0
        atoms_in_cell = 56
        elements = 2
        n_atoms = Nc[0] * Nc[1] * Nc[2] * atoms_in_cell

        loc = [0, 0, 0]

        # Each pair is the atomic number followed by the number of atoms of this element
        atomic_numbers = [[26, 24], [8, 32]]

        # Array containing the atom locations within a cell. The crystal structure here
        # is magnetite.
        atom_locations = [
                            # Fe2 +
                            [0.000 * Sc[0], 0.000 * Sc[1], 0.000 * Sc[2]],
                            [0.500 * Sc[0], 0.500 * Sc[1], 0.000 * Sc[2]],
                            [0.000 * Sc[0], 0.500 * Sc[1], 0.500 * Sc[2]],
                            [0.500 * Sc[0], 0.000 * Sc[1], 0.500 * Sc[2]],

                            [0.250 * Sc[0], 0.250 * Sc[1], 0.250 * Sc[2]],
                            [0.250 * Sc[0], 0.750 * Sc[1], 0.750 * Sc[2]],
                            [0.750 * Sc[0], 0.750 * Sc[1], 0.250 * Sc[2]],
                            [0.750 * Sc[0], 0.250 * Sc[1], 0.750 * Sc[2]],

                            # Fe3 +
                            [0.125 * Sc[0], 0.375 * Sc[1], 0.875 * Sc[2]],
                            [0.375 * Sc[0], 0.125 * Sc[1], 0.875 * Sc[2]],
                            [0.625 * Sc[0], 0.875 * Sc[1], 0.875 * Sc[2]],
                            [0.875 * Sc[0], 0.625 * Sc[1], 0.875 * Sc[2]],

                            [0.125 * Sc[0], 0.125 * Sc[1], 0.625 * Sc[2]],
                            [0.375 * Sc[0], 0.375 * Sc[1], 0.625 * Sc[2]],
                            [0.625 * Sc[0], 0.625 * Sc[1], 0.625 * Sc[2]],
                            [0.875 * Sc[0], 0.875 * Sc[1], 0.625 * Sc[2]],

                            [0.125 * Sc[0], 0.875 * Sc[1], 0.375 * Sc[2]],
                            [0.375 * Sc[0], 0.625 * Sc[1], 0.375 * Sc[2]],
                            [0.625 * Sc[0], 0.375 * Sc[1], 0.375 * Sc[2]],
                            [0.875 * Sc[0], 0.125 * Sc[1], 0.375 * Sc[2]],

                            [0.125 * Sc[0], 0.625 * Sc[1], 0.125 * Sc[2]],
                            [0.375 * Sc[0], 0.875 * Sc[1], 0.125 * Sc[2]],
                            [0.625 * Sc[0], 0.125 * Sc[1], 0.125 * Sc[2]],
                            [0.875 * Sc[0], 0.375 * Sc[1], 0.125 * Sc[2]],

                            # O2 -
                            [0.125 * Sc[0], 0.125 * Sc[1], 0.875 * Sc[2]],
                            [0.125 * Sc[0], 0.625 * Sc[1], 0.875 * Sc[2]],
                            [0.375 * Sc[0], 0.375 * Sc[1], 0.875 * Sc[2]],
                            [0.375 * Sc[0], 0.875 * Sc[1], 0.875 * Sc[2]],
                            [0.625 * Sc[0], 0.125 * Sc[1], 0.875 * Sc[2]],
                            [0.625 * Sc[0], 0.625 * Sc[1], 0.875 * Sc[2]],
                            [0.875 * Sc[0], 0.375 * Sc[1], 0.875 * Sc[2]],
                            [0.875 * Sc[0], 0.875 * Sc[1], 0.875 * Sc[2]],

                            [0.125 * Sc[0], 0.375 * Sc[1], 0.625 * Sc[2]],
                            [0.125 * Sc[0], 0.875 * Sc[1], 0.625 * Sc[2]],
                            [0.375 * Sc[0], 0.125 * Sc[1], 0.625 * Sc[2]],
                            [0.375 * Sc[0], 0.625 * Sc[1], 0.625 * Sc[2]],
                            [0.625 * Sc[0], 0.375 * Sc[1], 0.625 * Sc[2]],
                            [0.625 * Sc[0], 0.875 * Sc[1], 0.625 * Sc[2]],
                            [0.875 * Sc[0], 0.125 * Sc[1], 0.625 * Sc[2]],
                            [0.875 * Sc[0], 0.625 * Sc[1], 0.625 * Sc[2]],

                            [0.125 * Sc[0], 0.125 * Sc[1], 0.375 * Sc[2]],
                            [0.125 * Sc[0], 0.625 * Sc[1], 0.375 * Sc[2]],
                            [0.375 * Sc[0], 0.375 * Sc[1], 0.375 * Sc[2]],
                            [0.375 * Sc[0], 0.875 * Sc[1], 0.375 * Sc[2]],
                            [0.625 * Sc[0], 0.125 * Sc[1], 0.375 * Sc[2]],
                            [0.625 * Sc[0], 0.625 * Sc[1], 0.375 * Sc[2]],
                            [0.875 * Sc[0], 0.375 * Sc[1], 0.375 * Sc[2]],
                            [0.875 * Sc[0], 0.875 * Sc[1], 0.375 * Sc[2]],

                            [0.125 * Sc[0], 0.375 * Sc[1], 0.125 * Sc[2]],
                            [0.125 * Sc[0], 0.875 * Sc[1], 0.125 * Sc[2]],
                            [0.375 * Sc[0], 0.125 * Sc[1], 0.125 * Sc[2]],
                            [0.375 * Sc[0], 0.625 * Sc[1], 0.125 * Sc[2]],
                            [0.625 * Sc[0], 0.375 * Sc[1], 0.125 * Sc[2]],
                            [0.625 * Sc[0], 0.875 * Sc[1], 0.125 * Sc[2]],
                            [0.875 * Sc[0], 0.125 * Sc[1], 0.125 * Sc[2]],
                            [0.875 * Sc[0], 0.625 * Sc[1], 0.125 * Sc[2]],
                            ]

        location_list = []
        print("Generating specimen...")
        progress_bar = pyprind.ProgBar(Nc[0], sys.stdout)
        for t1 in range(0, Nc[0]):
            for t2 in range(0, Nc[1]):
                for t3 in range(0, Nc[2]):
                    for element in range(0, elements):
                        for atom in range(0, atomic_numbers[element][1]):

                            loc[0] = -aA / 2 + float(t1) * Sc[0] + \
                                     atom_locations[atom + element * atomic_numbers[0][1]][0]
                            loc[1] = -aA / 2 + float(t2) * Sc[1] + \
                                     atom_locations[atom + element * atomic_numbers[0][1]][1]
                            loc[2] = -aA / 2 + float(t3) * Sc[2] + \
                                     atom_locations[atom + element * atomic_numbers[0][1]][2]
                            i = int(np.floor(loc[0] * M / aA + M / 2))
                            j = int(np.floor(loc[1] * M / aA + M / 2))
                            k = int(np.floor(loc[2] * M / aA + M / 2))

                            if self.specimen[i, j, k] != 0:
                                location_list.append(loc + [atomic_numbers[element][0]])
                                atoms_read += 1

            progress_bar.update()

        return location_list

    @staticmethod
    def rotation_matrix(axis, angle):

        r = np.array([[np.cos(angle) + axis[0] * axis[0] * (1 - np.cos(angle)),
                       axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle),
                       axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)],
                      [axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle),
                       np.cos(angle) + axis[1] * axis[1] * (1 - np.cos(angle)),
                       axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)],
                      [axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle),
                       axis[2] * axis[1] * (1 - np.cos(angle) + axis[0] * np.sin(angle)),
                       np.cos(angle) + axis[2] * axis[2] * (1 - np.cos(angle))]])
        return r

    @jit
    def freqn(self, M, aA, k2max):
        xo = np.arange(M) * aA / (M - 1)
        imid = M / 2
        ko = (np.arange(M) - imid) / aA
        ko2 = np.square(ko)
        window = np.zeros((M, M), dtype=complex)
        for i in range(M):
            for j in range(M):
                if ko2[i] + ko2[j] < k2max:
                    window[i, j] = 1
                    self.nbeams += 1

        window = fft.ifftshift(window)
        ko2 = fft.ifftshift(ko2)
        return ko2, window

    @staticmethod
    def sort_by_z(x, y, z, z_number):
        """
        Sorts atoms from lowest to highest z-depth
        :param x: x coord of atom
        :param y: y coord of atom
        :param z: z coord of atom
        :param z_number: atomic number of atom
        :return:
        """
        a = np.vstack((x, y, z, z_number))
        return a[:, a[2, :].argsort()]

    def sigma(self):
        """
        Interaction parameter sigma in radians/(kV-Angstrom)
        :param kev:
        :return:
        """
        emass = 510.99906  # Electron mass in keV
        wl = self.wavelength * 1e10  # Convert wavelength to Angstroms
        energy = self.energy
        x = (emass + energy) / (2 * emass + energy)
        return 2 * PI * x / (wl * self.energy)

    @staticmethod
    @jit
    def bessi0(x):

        i0a = [1.0, 3.5156229, 3.0899424, 1.2067492,
               0.2659732, 0.0360768, 0.0045813]

        i0b = [0.39894228, 0.01328592, 0.00225319,
               -0.00157565, 0.00916281, -0.02057706, 0.02635537,
               -0.01647633, 0.00392377]

        ax = np.abs(x)
        if ax <= 3.75:
            t = x / 3.75
            t = t * t
            sum = i0a[6]
            for i in [5, 4, 3, 2, 1, 0]:
                sum = sum * t + i0a[i]
        else:
            t = 3.75 / ax
            sum = i0b[8]
            for i in [7, 6, 5, 4, 3, 2, 1, 0]:
                sum = sum * t + i0b[i]
            sum = np.exp(ax) * sum / np.sqrt(ax)

        return sum

    @staticmethod
    @jit
    def bessk0(x):

        k0a = [-0.57721566, 0.42278420, 0.23069756,
               0.03488590, 0.00262698, 0.00010750, 0.00000740]
        k0b = [1.25331414, -0.07832358, 0.02189568,
               -0.01062446, 0.00587872, -0.00251540, 0.00053208]
        ax = np.abs(x)
        if ax > 0 and ax <= 2:
            x2 = ax / 2
            x2 = x2 * x2
            sum = k0a[6]
            for i in [5, 4, 3, 2, 1, 0]:
                sum = sum * x2 + k0a[i]
            sum = -np.log(ax / 2) * PhaseImagingSystem.bessi0(x) + sum
        elif ax > 2:
            x2 = 2 / ax
            sum = k0b[6]
            for i in [5, 4, 3, 2, 1, 0]:
                sum = sum * x2 + k0b[i]
            sum = np.exp(-ax) * sum / np.sqrt(ax)
        else:
            sum = 1e20

        return sum

    @staticmethod
    @jit
    def vzatom(z, radius):

        # Lorenzian, Gaussian constants
        al = 300.8242834
        ag = 150.4121417

        if z < nzmin or z > nzmax:
            return 0.0
        if PhaseImagingSystem.fe_table_read == 0:
            nfe = PhaseImagingSystem.read_fe_table()
        r = np.abs(radius)

        # Avoid singularity at r = 0
        if r < 1e-10:
            r = 1e-10

        suml = sumg = 0

        # Lorenzians
        x = 2 * PI * r
        for i in range(0, 2 * nl, 2):
            suml += PhaseImagingSystem.fparams[z][i] * PhaseImagingSystem.bessk0(x *
                                                                                 np.sqrt(PhaseImagingSystem.fparams[z][
                                                                                             i + 1]))

        # Gaussians
        x = PI * r
        x = x * x
        for i in range(2 * nl, 2 * (nl + ng), 2):
            sumg += PhaseImagingSystem.fparams[z][i] * np.exp(-x / PhaseImagingSystem.fparams[z][i + 1]) \
                    / PhaseImagingSystem.fparams[z][i + 1]

        return al * suml + ag * sumg

    @staticmethod
    @jit
    def splinh(x, y, b, c, d, n):

        if n < 4:
            return
        # Do the first end point (special case) and get starting values
        m5 = (y[3] - y[2]) / (x[3] - x[2])
        m4 = (y[2] - y[1]) / (x[2] - x[1])
        m3 = (y[1] - y[0]) / (x[1] - x[0])

        m2 = m3 + m3 - m4
        m1 = m2 + m2 - m3

        m54 = np.abs(m5 - m4)
        m43 = np.abs(m4 - m3)
        m32 = np.abs(m3 - m2)
        m21 = np.abs(m2 - m1)

        if m43 + m21 > SMALL:
            t1 = (m43 * m2 + m21 * m3) / (m43 + m21)
        else:
            t1 = 0.5 * (m2 + m3)

        # Do everything up to the last end points

        nm1 = n - 1
        nm4 = n - 4



        for i in range(nm1):
            if m54 + m32 > SMALL:
                t2 = (m54 * m3 + m32 * m4) / (m54 + m32)
            else:
                t2 = 0.5 * (m3 + m4)

            x43 = x[i + 1] - x[i]
            b[i] = t1
            c[i] = (3 * m3 - t1 - t1 - t2) / x43
            d[i] = (t1 + t2 - m3 - m3) / (x43 * x43)

            m1 = m2
            m2 = m3
            m3 = m4
            m4 = m5

            if i < nm4:
                m5 = (y[i + 4] - y[i + 3]) / (x[i + 4] - x[i + 3])
            else:
                m5 = m4 + m4 - m3

            m21 = m32
            m32 = m43
            m43 = m54
            m54 = np.abs(m5 - m4)
            t1 = t2
        return x, y, b, c, d

    @staticmethod
    @jit
    def seval(x, y, b, c, d, n, x0):
        # Exit if x0 is outside the spline range
        n = int(n)
        if x0 <= x[0]:
            i = 0
        elif x0 >= x[n - 2]:
            i = n - 2
        else:
            i = 0
            j = n

            k = int((i + j) / 2)
            if x0 < x[k]:
                j = k
            elif x0 >= x[k]:
                i = k
            while j - i > 1:
                k = int((i + j) / 2)
                if x0 < x[k]:
                    j = k
                elif x0 >= x[k]:
                    i = k
        z = x0 - x[i]
        seval1 = y[i] + (b[i] + (c[i] + d[i] * z) * z) * z
        return seval1

    @staticmethod
    @jit
    def vz_atom_lut(z, rsq):
        """
        return the (real space) projected atomic potential for atomic
        number Z at radius r (in Angstroms)
        :param z:
        :param rsq:
        :return:
        """
        rmin = 0.01  # r (in ang) range of LUT
        rmax = 5.0
        nrmax = 100  # Number of in lookup table

        z = int(z)  # Cast to int for using as list index

        # Spline interpolation coefficient
        spline_init = 0
        spliny = np.zeros((nzmax, nrmax))
        splinb = np.zeros((nzmax, nrmax))
        splinc = np.zeros((nzmax, nrmax))
        splind = np.zeros((nzmax, nrmax))

        if spline_init == 0:
            # Generate a set of logarithmic r values
            dlnr = np.log(rmax / rmin) / (nrmax - 1)
            iv = np.arange(nrmax)
            splinx = rmin * np.exp(iv * dlnr)
            splinx = splinx * splinx

            nspline = np.zeros(nzmax)
            spline_init = 1

        iz = z - 1
        if z < 1 or z > nzmax:
            print("Bad atomic number {0} in vz_atom_lut()".format(z))
        # If this atomic number has not been called before, generate the spline coefficients
        if nspline[iz] == 0:
            r = np.sqrt(splinx)
            for i in range(nrmax):
                spliny[iz, i] = PhaseImagingSystem.vzatom(z, r[i])
            nspline[iz] = nrmax
            splinx, spliny[iz], splinb[iz], splinc[iz], splind[iz] = PhaseImagingSystem.splinh(splinx,
                                                                                               spliny[iz],
                                                                                               splinb[iz],
                                                                                               splinc[iz],
                                                                                               splind[iz],
                                                                                               nrmax)

        # Now that everything is set up, find the scattering factor by interpolation
        # in the table
        vz = PhaseImagingSystem.seval(splinx,
                                      spliny[iz],
                                      splinb[iz],
                                      splinc[iz],
                                      splind[iz],
                                      nspline[iz],
                                      rsq)
        return vz

    @jit
    def tratoms_old(self, trans, x, y, z_number, scalex, scaley, idx, rmax2, rminsq, na):
        M = self.M
        ixo = x / scalex
        iyo = y / scaley
        ixo = ixo.astype(int)
        iyo = iyo.astype(int)
        nx1 = ixo - idx
        nx2 = ixo + idx
        ny1 = iyo - idx
        ny2 = iyo + idx



        prog_bar = pyprind.ProgBar(na, stream=sys.stdout)
        for i in range(na):
            prog_bar.update()
            for ix in range(nx1[i], nx2[i] + 1):
                rx2 = x[i] - float(ix) * scalex
                rx2 = rx2 * rx2
                ixw = ix % M
                for iy in range(ny1[i], ny2[i] + 1):
                    rsq = y[i] - float(iy) * scalex
                    rsq = rx2 + rsq * rsq
                    if rsq <= rmax2:
                        iyw = iy % M
                        if rsq < rminsq:
                            rsq = rminsq
                        vz = PhaseImagingSystem.vz_atom_lut(z_number[i], rsq)
                        trans[ixw, iyw] += vz


    @jit
    def tratoms(self, trans, x, y, z_number, scalex, scaley, idx, rmax2, rminsq, na):
        M = self.M
        ixo = x / scalex
        iyo = y / scaley
        ixo = ixo.astype(int)
        iyo = iyo.astype(int)
        nx1 = ixo - idx
        nx2 = ixo + idx
        ny1 = iyo - idx
        ny2 = iyo + idx

        prog_bar = pyprind.ProgBar(na, stream=sys.stdout)
        for i in range(na):
            ixv = np.arange(nx1[i], nx2[i] + 1)
            ixw = np.mod(ixv, M)
            iyv = np.arange(ny1[i], ny2[i] + 1)
            iyw = np.mod(iyv, M)
            prog_bar.update()
            rx2 = x[i] - ixv.astype(float) * scalex
            rx2 = rx2 * rx2
            iyv = np.arange(ny1[i], ny2[i] + 1)
            ry2 = y[i] - iyv.astype(float) * scaley
            ry2 = ry2 * ry2
            rsqx, rsqy = np.meshgrid(rx2, ry2, indexing='ij')
            rsq = rsqx + rsqy
            rsq = np.where(rsq >= rminsq, rsq, rminsq)

            for ix in range(nx1[i], nx2[i] + 1):
                for iy in range(ny1[i], ny2[i] + 1):
                    if rsq[ix - nx1[i], iy - ny1[i]] <= rmax2:
                        vz = PhaseImagingSystem.vz_atom_lut(z_number[i], rsq[ix - nx1[i], iy - ny1[i]])
                        trans[ixw[ix - nx1[i]], iyw[iy - ny1[i]]] += vz

    @jit
    def trlayer(self, x, y, z_number, na, aA, trans, window):

        M = self.M
        rmax = 3  # Maximum atomic radius in Angstrom
        rmax2 = rmax * rmax
        scale = self.sigma()  # in 1 / (volt-Anstroms)
        scalex = aA / M
        scaley = aA / M

        # Minimum radius (for regularisation)
        rmin = aA / M
        r = aA / M
        rmin = 0.25 * np.sqrt(0.5 * (rmin * rmin + r * r))
        rminsq = rmin * rmin

        idx = int(M * rmax / aA) + 1
        print("Building Transmission Layer")

        self.tratoms(trans, x, y, z_number, scalex, scaley, idx, rmax2, rminsq, na)

        # Convert phase to complex transmission function
        vz = scale * trans.real
        trans = np.cos(vz) + np.sin(vz)

        # Bandwidth limit the transmission function
        trans = fft.fft2(trans)
        trans = trans * window
        trans = fft.ifft2(trans)
        return trans


    @staticmethod
    @jit
    def read_fe_table():
        if PhaseImagingSystem.fe_table_read == 1:
            return 0
        fparams = PhaseImagingSystem.fparams

        fparams[1][0] = 4.20298324e-003
        fparams[1][1] = 2.25350888e-001
        fparams[1][2] = 6.27762505e-002
        fparams[1][3] = 2.25366950e-001
        fparams[1][4] = 3.00907347e-002
        fparams[1][5] = 2.25331756e-001
        fparams[1][6] = 6.77756695e-002
        fparams[1][7] = 4.38854001e+000
        fparams[1][8] = 3.56609237e-003
        fparams[1][9] = 4.03884823e-001
        fparams[1][10] = 2.76135815e-002
        fparams[1][11] = 1.44490166e+000
        fparams[2][0] = 1.87543704e-005
        fparams[2][1] = 2.12427997e-001
        fparams[2][2] = 4.10595800e-004
        fparams[2][3] = 3.32212279e-001
        fparams[2][4] = 1.96300059e-001
        fparams[2][5] = 5.17325152e-001
        fparams[2][6] = 8.36015738e-003
        fparams[2][7] = 3.66668239e-001
        fparams[2][8] = 2.95102022e-002
        fparams[2][9] = 1.37171827e+000
        fparams[2][10] = 4.65928982e-007
        fparams[2][11] = 3.75768025e+004
        fparams[3][0] = 7.45843816e-002
        fparams[3][1] = 8.81151424e-001
        fparams[3][2] = 7.15382250e-002
        fparams[3][3] = 4.59142904e-002
        fparams[3][4] = 1.45315229e-001
        fparams[3][5] = 8.81301714e-001
        fparams[3][6] = 1.12125769e+000
        fparams[3][7] = 1.88483665e+001
        fparams[3][8] = 2.51736525e-003
        fparams[3][9] = 1.59189995e-001
        fparams[3][10] = 3.58434971e-001
        fparams[3][11] = 6.12371000e+000
        fparams[4][0] = 6.11642897e-002
        fparams[4][1] = 9.90182132e-002
        fparams[4][2] = 1.25755034e-001
        fparams[4][3] = 9.90272412e-002
        fparams[4][4] = 2.00831548e-001
        fparams[4][5] = 1.87392509e+000
        fparams[4][6] = 7.87242876e-001
        fparams[4][7] = 9.32794929e+000
        fparams[4][8] = 1.58847850e-003
        fparams[4][9] = 8.91900236e-002
        fparams[4][10] = 2.73962031e-001
        fparams[4][11] = 3.20687658e+000
        fparams[5][0] = 1.25716066e-001
        fparams[5][1] = 1.48258830e-001
        fparams[5][2] = 1.73314452e-001
        fparams[5][3] = 1.48257216e-001
        fparams[5][4] = 1.84774811e-001
        fparams[5][5] = 3.34227311e+000
        fparams[5][6] = 1.95250221e-001
        fparams[5][7] = 1.97339463e+000
        fparams[5][8] = 5.29642075e-001
        fparams[5][9] = 5.70035553e+000
        fparams[5][10] = 1.08230500e-003
        fparams[5][11] = 5.64857237e-002
        fparams[6][0] = 2.12080767e-001
        fparams[6][1] = 2.08605417e-001
        fparams[6][2] = 1.99811865e-001
        fparams[6][3] = 2.08610186e-001
        fparams[6][4] = 1.68254385e-001
        fparams[6][5] = 5.57870773e+000
        fparams[6][6] = 1.42048360e-001
        fparams[6][7] = 1.33311887e+000
        fparams[6][8] = 3.63830672e-001
        fparams[6][9] = 3.80800263e+000
        fparams[6][10] = 8.35012044e-004
        fparams[6][11] = 4.03982620e-002
        fparams[7][0] = 5.33015554e-001
        fparams[7][1] = 2.90952515e-001
        fparams[7][2] = 5.29008883e-002
        fparams[7][3] = 1.03547896e+001
        fparams[7][4] = 9.24159648e-002
        fparams[7][5] = 1.03540028e+001
        fparams[7][6] = 2.61799101e-001
        fparams[7][7] = 2.76252723e+000
        fparams[7][8] = 8.80262108e-004
        fparams[7][9] = 3.47681236e-002
        fparams[7][10] = 1.10166555e-001
        fparams[7][11] = 9.93421736e-001
        fparams[8][0] = 3.39969204e-001
        fparams[8][1] = 3.81570280e-001
        fparams[8][2] = 3.07570172e-001
        fparams[8][3] = 3.81571436e-001
        fparams[8][4] = 1.30369072e-001
        fparams[8][5] = 1.91919745e+001
        fparams[8][6] = 8.83326058e-002
        fparams[8][7] = 7.60635525e-001
        fparams[8][8] = 1.96586700e-001
        fparams[8][9] = 2.07401094e+000
        fparams[8][10] = 9.96220028e-004
        fparams[8][11] = 3.03266869e-002
        fparams[9][0] = 2.30560593e-001
        fparams[9][1] = 4.80754213e-001
        fparams[9][2] = 5.26889648e-001
        fparams[9][3] = 4.80763895e-001
        fparams[9][4] = 1.24346755e-001
        fparams[9][5] = 3.95306720e+001
        fparams[9][6] = 1.24616894e-003
        fparams[9][7] = 2.62181803e-002
        fparams[9][8] = 7.20452555e-002
        fparams[9][9] = 5.92495593e-001
        fparams[9][10] = 1.53075777e-001
        fparams[9][11] = 1.59127671e+000
        fparams[10][0] = 4.08371771e-001
        fparams[10][1] = 5.88228627e-001
        fparams[10][2] = 4.54418858e-001
        fparams[10][3] = 5.88288655e-001
        fparams[10][4] = 1.44564923e-001
        fparams[10][5] = 1.21246013e+002
        fparams[10][6] = 5.91531395e-002
        fparams[10][7] = 4.63963540e-001
        fparams[10][8] = 1.24003718e-001
        fparams[10][9] = 1.23413025e+000
        fparams[10][10] = 1.64986037e-003
        fparams[10][11] = 2.05869217e-002
        fparams[11][0] = 1.36471662e-001
        fparams[11][1] = 4.99965301e-002
        fparams[11][2] = 7.70677865e-001
        fparams[11][3] = 8.81899664e-001
        fparams[11][4] = 1.56862014e-001
        fparams[11][5] = 1.61768579e+001
        fparams[11][6] = 9.96821513e-001
        fparams[11][7] = 2.00132610e+001
        fparams[11][8] = 3.80304670e-002
        fparams[11][9] = 2.60516254e-001
        fparams[11][10] = 1.27685089e-001
        fparams[11][11] = 6.99559329e-001
        fparams[12][0] = 3.04384121e-001
        fparams[12][1] = 8.42014377e-002
        fparams[12][2] = 7.56270563e-001
        fparams[12][3] = 1.64065598e+000
        fparams[12][4] = 1.01164809e-001
        fparams[12][5] = 2.97142975e+001
        fparams[12][6] = 3.45203403e-002
        fparams[12][7] = 2.16596094e-001
        fparams[12][8] = 9.71751327e-001
        fparams[12][9] = 1.21236852e+001
        fparams[12][10] = 1.20593012e-001
        fparams[12][11] = 5.60865838e-001
        fparams[13][0] = 7.77419424e-001
        fparams[13][1] = 2.71058227e+000
        fparams[13][2] = 5.78312036e-002
        fparams[13][3] = 7.17532098e+001
        fparams[13][4] = 4.26386499e-001
        fparams[13][5] = 9.13331555e-002
        fparams[13][6] = 1.13407220e-001
        fparams[13][7] = 4.48867451e-001
        fparams[13][8] = 7.90114035e-001
        fparams[13][9] = 8.66366718e+000
        fparams[13][10] = 3.23293496e-002
        fparams[13][11] = 1.78503463e-001
        fparams[14][0] = 1.06543892e+000
        fparams[14][1] = 1.04118455e+000
        fparams[14][2] = 1.20143691e-001
        fparams[14][3] = 6.87113368e+001
        fparams[14][4] = 1.80915263e-001
        fparams[14][5] = 8.87533926e-002
        fparams[14][6] = 1.12065620e+000
        fparams[14][7] = 3.70062619e+000
        fparams[14][8] = 3.05452816e-002
        fparams[14][9] = 2.14097897e-001
        fparams[14][10] = 1.59963502e+000
        fparams[14][11] = 9.99096638e+000
        fparams[15][0] = 1.05284447e+000
        fparams[15][1] = 1.31962590e+000
        fparams[15][2] = 2.99440284e-001
        fparams[15][3] = 1.28460520e-001
        fparams[15][4] = 1.17460748e-001
        fparams[15][5] = 1.02190163e+002
        fparams[15][6] = 9.60643452e-001
        fparams[15][7] = 2.87477555e+000
        fparams[15][8] = 2.63555748e-002
        fparams[15][9] = 1.82076844e-001
        fparams[15][10] = 1.38059330e+000
        fparams[15][11] = 7.49165526e+000
        fparams[16][0] = 1.01646916e+000
        fparams[16][1] = 1.69181965e+000
        fparams[16][2] = 4.41766748e-001
        fparams[16][3] = 1.74180288e-001
        fparams[16][4] = 1.21503863e-001
        fparams[16][5] = 1.67011091e+002
        fparams[16][6] = 8.27966670e-001
        fparams[16][7] = 2.30342810e+000
        fparams[16][8] = 2.33022533e-002
        fparams[16][9] = 1.56954150e-001
        fparams[16][10] = 1.18302846e+000
        fparams[16][11] = 5.85782891e+000
        fparams[17][0] = 9.44221116e-001
        fparams[17][1] = 2.40052374e-001
        fparams[17][2] = 4.37322049e-001
        fparams[17][3] = 9.30510439e+000
        fparams[17][4] = 2.54547926e-001
        fparams[17][5] = 9.30486346e+000
        fparams[17][6] = 5.47763323e-002
        fparams[17][7] = 1.68655688e-001
        fparams[17][8] = 8.00087488e-001
        fparams[17][9] = 2.97849774e+000
        fparams[17][10] = 1.07488641e-002
        fparams[17][11] = 6.84240646e-002
        fparams[18][0] = 1.06983288e+000
        fparams[18][1] = 2.87791022e-001
        fparams[18][2] = 4.24631786e-001
        fparams[18][3] = 1.24156957e+001
        fparams[18][4] = 2.43897949e-001
        fparams[18][5] = 1.24158868e+001
        fparams[18][6] = 4.79446296e-002
        fparams[18][7] = 1.36979796e-001
        fparams[18][8] = 7.64958952e-001
        fparams[18][9] = 2.43940729e+000
        fparams[18][10] = 8.23128431e-003
        fparams[18][11] = 5.27258749e-002
        fparams[19][0] = 6.92717865e-001
        fparams[19][1] = 7.10849990e+000
        fparams[19][2] = 9.65161085e-001
        fparams[19][3] = 3.57532901e-001
        fparams[19][4] = 1.48466588e-001
        fparams[19][5] = 3.93763275e-002
        fparams[19][6] = 2.64645027e-002
        fparams[19][7] = 1.03591321e-001
        fparams[19][8] = 1.80883768e+000
        fparams[19][9] = 3.22845199e+001
        fparams[19][10] = 5.43900018e-001
        fparams[19][11] = 1.67791374e+000
        fparams[20][0] = 3.66902871e-001
        fparams[20][1] = 6.14274129e-002
        fparams[20][2] = 8.66378999e-001
        fparams[20][3] = 5.70881727e-001
        fparams[20][4] = 6.67203300e-001
        fparams[20][5] = 7.82965639e+000
        fparams[20][6] = 4.87743636e-001
        fparams[20][7] = 1.32531318e+000
        fparams[20][8] = 1.82406314e+000
        fparams[20][9] = 2.10056032e+001
        fparams[20][10] = 2.20248453e-002
        fparams[20][11] = 9.11853450e-002
        fparams[21][0] = 3.78871777e-001
        fparams[21][1] = 6.98910162e-002
        fparams[21][2] = 9.00022505e-001
        fparams[21][3] = 5.21061541e-001
        fparams[21][4] = 7.15288914e-001
        fparams[21][5] = 7.87707920e+000
        fparams[21][6] = 1.88640973e-002
        fparams[21][7] = 8.17512708e-002
        fparams[21][8] = 4.07945949e-001
        fparams[21][9] = 1.11141388e+000
        fparams[21][10] = 1.61786540e+000
        fparams[21][11] = 1.80840759e+001
        fparams[22][0] = 3.62383267e-001
        fparams[22][1] = 7.54707114e-002
        fparams[22][2] = 9.84232966e-001
        fparams[22][3] = 4.97757309e-001
        fparams[22][4] = 7.41715642e-001
        fparams[22][5] = 8.17659391e+000
        fparams[22][6] = 3.62555269e-001
        fparams[22][7] = 9.55524906e-001
        fparams[22][8] = 1.49159390e+000
        fparams[22][9] = 1.62221677e+001
        fparams[22][10] = 1.61659509e-002
        fparams[22][11] = 7.33140839e-002
        fparams[23][0] = 3.52961378e-001
        fparams[23][1] = 8.19204103e-002
        fparams[23][2] = 7.46791014e-001
        fparams[23][3] = 8.81189511e+000
        fparams[23][4] = 1.08364068e+000
        fparams[23][5] = 5.10646075e-001
        fparams[23][6] = 1.39013610e+000
        fparams[23][7] = 1.48901841e+001
        fparams[23][8] = 3.31273356e-001
        fparams[23][9] = 8.38543079e-001
        fparams[23][10] = 1.40422612e-002
        fparams[23][11] = 6.57432678e-002
        fparams[24][0] = 1.34348379e+000
        fparams[24][1] = 1.25814353e+000
        fparams[24][2] = 5.07040328e-001
        fparams[24][3] = 1.15042811e+001
        fparams[24][4] = 4.26358955e-001
        fparams[24][5] = 8.53660389e-002
        fparams[24][6] = 1.17241826e-002
        fparams[24][7] = 6.00177061e-002
        fparams[24][8] = 5.11966516e-001
        fparams[24][9] = 1.53772451e+000
        fparams[24][10] = 3.38285828e-001
        fparams[24][11] = 6.62418319e-001
        fparams[25][0] = 3.26697613e-001
        fparams[25][1] = 8.88813083e-002
        fparams[25][2] = 7.17297000e-001
        fparams[25][3] = 1.11300198e+001
        fparams[25][4] = 1.33212464e+000
        fparams[25][5] = 5.82141104e-001
        fparams[25][6] = 2.80801702e-001
        fparams[25][7] = 6.71583145e-001
        fparams[25][8] = 1.15499241e+000
        fparams[25][9] = 1.26825395e+001
        fparams[25][10] = 1.11984488e-002
        fparams[25][11] = 5.32334467e-002
        fparams[26][0] = 3.13454847e-001
        fparams[26][1] = 8.99325756e-002
        fparams[26][2] = 6.89290016e-001
        fparams[26][3] = 1.30366038e+001
        fparams[26][4] = 1.47141531e+000
        fparams[26][5] = 6.33345291e-001
        fparams[26][6] = 1.03298688e+000
        fparams[26][7] = 1.16783425e+001
        fparams[26][8] = 2.58280285e-001
        fparams[26][9] = 6.09116446e-001
        fparams[26][10] = 1.03460690e-002
        fparams[26][11] = 4.81610627e-002
        fparams[27][0] = 3.15878278e-001
        fparams[27][1] = 9.46683246e-002
        fparams[27][2] = 1.60139005e+000
        fparams[27][3] = 6.99436449e-001
        fparams[27][4] = 6.56394338e-001
        fparams[27][5] = 1.56954403e+001
        fparams[27][6] = 9.36746624e-001
        fparams[27][7] = 1.09392410e+001
        fparams[27][8] = 9.77562646e-003
        fparams[27][9] = 4.37446816e-002
        fparams[27][10] = 2.38378578e-001
        fparams[27][11] = 5.56286483e-001
        fparams[28][0] = 1.72254630e+000
        fparams[28][1] = 7.76606908e-001
        fparams[28][2] = 3.29543044e-001
        fparams[28][3] = 1.02262360e-001
        fparams[28][4] = 6.23007200e-001
        fparams[28][5] = 1.94156207e+001
        fparams[28][6] = 9.43496513e-003
        fparams[28][7] = 3.98684596e-002
        fparams[28][8] = 8.54063515e-001
        fparams[28][9] = 1.04078166e+001
        fparams[28][10] = 2.21073515e-001
        fparams[28][11] = 5.10869330e-001
        fparams[29][0] = 3.58774531e-001
        fparams[29][1] = 1.06153463e-001
        fparams[29][2] = 1.76181348e+000
        fparams[29][3] = 1.01640995e+000
        fparams[29][4] = 6.36905053e-001
        fparams[29][5] = 1.53659093e+001
        fparams[29][6] = 7.44930667e-003
        fparams[29][7] = 3.85345989e-002
        fparams[29][8] = 1.89002347e-001
        fparams[29][9] = 3.98427790e-001
        fparams[29][10] = 2.29619589e-001
        fparams[29][11] = 9.01419843e-001
        fparams[30][0] = 5.70893973e-001
        fparams[30][1] = 1.26534614e-001
        fparams[30][2] = 1.98908856e+000
        fparams[30][3] = 2.17781965e+000
        fparams[30][4] = 3.06060585e-001
        fparams[30][5] = 3.78619003e+001
        fparams[30][6] = 2.35600223e-001
        fparams[30][7] = 3.67019041e-001
        fparams[30][8] = 3.97061102e-001
        fparams[30][9] = 8.66419596e-001
        fparams[30][10] = 6.85657228e-003
        fparams[30][11] = 3.35778823e-002
        fparams[31][0] = 6.25528464e-001
        fparams[31][1] = 1.10005650e-001
        fparams[31][2] = 2.05302901e+000
        fparams[31][3] = 2.41095786e+000
        fparams[31][4] = 2.89608120e-001
        fparams[31][5] = 4.78685736e+001
        fparams[31][6] = 2.07910594e-001
        fparams[31][7] = 3.27807224e-001
        fparams[31][8] = 3.45079617e-001
        fparams[31][9] = 7.43139061e-001
        fparams[31][10] = 6.55634298e-003
        fparams[31][11] = 3.09411369e-002
        fparams[32][0] = 5.90952690e-001
        fparams[32][1] = 1.18375976e-001
        fparams[32][2] = 5.39980660e-001
        fparams[32][3] = 7.18937433e+001
        fparams[32][4] = 2.00626188e+000
        fparams[32][5] = 1.39304889e+000
        fparams[32][6] = 7.49705041e-001
        fparams[32][7] = 6.89943350e+000
        fparams[32][8] = 1.83581347e-001
        fparams[32][9] = 3.64667232e-001
        fparams[32][10] = 9.52190743e-003
        fparams[32][11] = 2.69888650e-002
        fparams[33][0] = 7.77875218e-001
        fparams[33][1] = 1.50733157e-001
        fparams[33][2] = 5.93848150e-001
        fparams[33][3] = 1.42882209e+002
        fparams[33][4] = 1.95918751e+000
        fparams[33][5] = 1.74750339e+000
        fparams[33][6] = 1.79880226e-001
        fparams[33][7] = 3.31800852e-001
        fparams[33][8] = 8.63267222e-001
        fparams[33][9] = 5.85490274e+000
        fparams[33][10] = 9.59053427e-003
        fparams[33][11] = 2.33777569e-002
        fparams[34][0] = 9.58390681e-001
        fparams[34][1] = 1.83775557e-001
        fparams[34][2] = 6.03851342e-001
        fparams[34][3] = 1.96819224e+002
        fparams[34][4] = 1.90828931e+000
        fparams[34][5] = 2.15082053e+000
        fparams[34][6] = 1.73885956e-001
        fparams[34][7] = 3.00006024e-001
        fparams[34][8] = 9.35265145e-001
        fparams[34][9] = 4.92471215e+000
        fparams[34][10] = 8.62254658e-003
        fparams[34][11] = 2.12308108e-002
        fparams[35][0] = 1.14136170e+000
        fparams[35][1] = 2.18708710e-001
        fparams[35][2] = 5.18118737e-001
        fparams[35][3] = 1.93916682e+002
        fparams[35][4] = 1.85731975e+000
        fparams[35][5] = 2.65755396e+000
        fparams[35][6] = 1.68217399e-001
        fparams[35][7] = 2.71719918e-001
        fparams[35][8] = 9.75705606e-001
        fparams[35][9] = 4.19482500e+000
        fparams[35][10] = 7.24187871e-003
        fparams[35][11] = 1.99325718e-002
        fparams[36][0] = 3.24386970e-001
        fparams[36][1] = 6.31317973e+001
        fparams[36][2] = 1.31732163e+000
        fparams[36][3] = 2.54706036e-001
        fparams[36][4] = 1.79912614e+000
        fparams[36][5] = 3.23668394e+000
        fparams[36][6] = 4.29961425e-003
        fparams[36][7] = 1.98965610e-002
        fparams[36][8] = 1.00429433e+000
        fparams[36][9] = 3.61094513e+000
        fparams[36][10] = 1.62188197e-001
        fparams[36][11] = 2.45583672e-001
        fparams[37][0] = 2.90445351e-001
        fparams[37][1] = 3.68420227e-002
        fparams[37][2] = 2.44201329e+000
        fparams[37][3] = 1.16013332e+000
        fparams[37][4] = 7.69435449e-001
        fparams[37][5] = 1.69591472e+001
        fparams[37][6] = 1.58687000e+000
        fparams[37][7] = 2.53082574e+000
        fparams[37][8] = 2.81617593e-003
        fparams[37][9] = 1.88577417e-002
        fparams[37][10] = 1.28663830e-001
        fparams[37][11] = 2.10753969e-001
        fparams[38][0] = 1.37373086e-002
        fparams[38][1] = 1.87469061e-002
        fparams[38][2] = 1.97548672e+000
        fparams[38][3] = 6.36079230e+000
        fparams[38][4] = 1.59261029e+000
        fparams[38][5] = 2.21992482e-001
        fparams[38][6] = 1.73263882e-001
        fparams[38][7] = 2.01624958e-001
        fparams[38][8] = 4.66280378e+000
        fparams[38][9] = 2.53027803e+001
        fparams[38][10] = 1.61265063e-003
        fparams[38][11] = 1.53610568e-002
        fparams[39][0] = 6.75302747e-001
        fparams[39][1] = 6.54331847e-002
        fparams[39][2] = 4.70286720e-001
        fparams[39][3] = 1.06108709e+002
        fparams[39][4] = 2.63497677e+000
        fparams[39][5] = 2.06643540e+000
        fparams[39][6] = 1.09621746e-001
        fparams[39][7] = 1.93131925e-001
        fparams[39][8] = 9.60348773e-001
        fparams[39][9] = 1.63310938e+000
        fparams[39][10] = 5.28921555e-003
        fparams[39][11] = 1.66083821e-002
        fparams[40][0] = 2.64365505e+000
        fparams[40][1] = 2.20202699e+000
        fparams[40][2] = 5.54225147e-001
        fparams[40][3] = 1.78260107e+002
        fparams[40][4] = 7.61376625e-001
        fparams[40][5] = 7.67218745e-002
        fparams[40][6] = 6.02946891e-003
        fparams[40][7] = 1.55143296e-002
        fparams[40][8] = 9.91630530e-002
        fparams[40][9] = 1.76175995e-001
        fparams[40][10] = 9.56782020e-001
        fparams[40][11] = 1.54330682e+000
        fparams[41][0] = 6.59532875e-001
        fparams[41][1] = 8.66145490e-002
        fparams[41][2] = 1.84545854e+000
        fparams[41][3] = 5.94774398e+000
        fparams[41][4] = 1.25584405e+000
        fparams[41][5] = 6.40851475e-001
        fparams[41][6] = 1.22253422e-001
        fparams[41][7] = 1.66646050e-001
        fparams[41][8] = 7.06638328e-001
        fparams[41][9] = 1.62853268e+000
        fparams[41][10] = 2.62381591e-003
        fparams[41][11] = 8.26257859e-003
        fparams[42][0] = 6.10160120e-001
        fparams[42][1] = 9.11628054e-002
        fparams[42][2] = 1.26544000e+000
        fparams[42][3] = 5.06776025e-001
        fparams[42][4] = 1.97428762e+000
        fparams[42][5] = 5.89590381e+000
        fparams[42][6] = 6.48028962e-001
        fparams[42][7] = 1.46634108e+000
        fparams[42][8] = 2.60380817e-003
        fparams[42][9] = 7.84336311e-003
        fparams[42][10] = 1.13887493e-001
        fparams[42][11] = 1.55114340e-001
        fparams[43][0] = 8.55189183e-001
        fparams[43][1] = 1.02962151e-001
        fparams[43][2] = 1.66219641e+000
        fparams[43][3] = 7.64907000e+000
        fparams[43][4] = 1.45575475e+000
        fparams[43][5] = 1.01639987e+000
        fparams[43][6] = 1.05445664e-001
        fparams[43][7] = 1.42303338e-001
        fparams[43][8] = 7.71657112e-001
        fparams[43][9] = 1.34659349e+000
        fparams[43][10] = 2.20992635e-003
        fparams[43][11] = 7.90358976e-003
        fparams[44][0] = 4.70847093e-001
        fparams[44][1] = 9.33029874e-002
        fparams[44][2] = 1.58180781e+000
        fparams[44][3] = 4.52831347e-001
        fparams[44][4] = 2.02419818e+000
        fparams[44][5] = 7.11489023e+000
        fparams[44][6] = 1.97036257e-003
        fparams[44][7] = 7.56181595e-003
        fparams[44][8] = 6.26912639e-001
        fparams[44][9] = 1.25399858e+000
        fparams[44][10] = 1.02641320e-001
        fparams[44][11] = 1.33786087e-001
        fparams[45][0] = 4.20051553e-001
        fparams[45][1] = 9.38882628e-002
        fparams[45][2] = 1.76266507e+000
        fparams[45][3] = 4.64441687e-001
        fparams[45][4] = 2.02735641e+000
        fparams[45][5] = 8.19346046e+000
        fparams[45][6] = 1.45487176e-003
        fparams[45][7] = 7.82704517e-003
        fparams[45][8] = 6.22809600e-001
        fparams[45][9] = 1.17194153e+000
        fparams[45][10] = 9.91529915e-002
        fparams[45][11] = 1.24532839e-001
        fparams[46][0] = 2.10475155e+000
        fparams[46][1] = 8.68606470e+000
        fparams[46][2] = 2.03884487e+000
        fparams[46][3] = 3.78924449e-001
        fparams[46][4] = 1.82067264e-001
        fparams[46][5] = 1.42921634e-001
        fparams[46][6] = 9.52040948e-002
        fparams[46][7] = 1.17125900e-001
        fparams[46][8] = 5.91445248e-001
        fparams[46][9] = 1.07843808e+000
        fparams[46][10] = 1.13328676e-003
        fparams[46][11] = 7.80252092e-003
        fparams[47][0] = 2.07981390e+000
        fparams[47][1] = 9.92540297e+000
        fparams[47][2] = 4.43170726e-001
        fparams[47][3] = 1.04920104e-001
        fparams[47][4] = 1.96515215e+000
        fparams[47][5] = 6.40103839e-001
        fparams[47][6] = 5.96130591e-001
        fparams[47][7] = 8.89594790e-001
        fparams[47][8] = 4.78016333e-001
        fparams[47][9] = 1.98509407e+000
        fparams[47][10] = 9.46458470e-002
        fparams[47][11] = 1.12744464e-001
        fparams[48][0] = 1.63657549e+000
        fparams[48][1] = 1.24540381e+001
        fparams[48][2] = 2.17927989e+000
        fparams[48][3] = 1.45134660e+000
        fparams[48][4] = 7.71300690e-001
        fparams[48][5] = 1.26695757e-001
        fparams[48][6] = 6.64193880e-001
        fparams[48][7] = 7.77659202e-001
        fparams[48][8] = 7.64563285e-001
        fparams[48][9] = 1.66075210e+000
        fparams[48][10] = 8.61126689e-002
        fparams[48][11] = 1.05728357e-001
        fparams[49][0] = 2.24820632e+000
        fparams[49][1] = 1.51913507e+000
        fparams[49][2] = 1.64706864e+000
        fparams[49][3] = 1.30113424e+001
        fparams[49][4] = 7.88679265e-001
        fparams[49][5] = 1.06128184e-001
        fparams[49][6] = 8.12579069e-002
        fparams[49][7] = 9.94045620e-002
        fparams[49][8] = 6.68280346e-001
        fparams[49][9] = 1.49742063e+000
        fparams[49][10] = 6.38467475e-001
        fparams[49][11] = 7.18422635e-001
        fparams[50][0] = 2.16644620e+000
        fparams[50][1] = 1.13174909e+001
        fparams[50][2] = 6.88691021e-001
        fparams[50][3] = 1.10131285e-001
        fparams[50][4] = 1.92431751e+000
        fparams[50][5] = 6.74464853e-001
        fparams[50][6] = 5.65359888e-001
        fparams[50][7] = 7.33564610e-001
        fparams[50][8] = 9.18683861e-001
        fparams[50][9] = 1.02310312e+001
        fparams[50][10] = 7.80542213e-002
        fparams[50][11] = 9.31104308e-002
        fparams[51][0] = 1.73662114e+000
        fparams[51][1] = 8.84334719e-001
        fparams[51][2] = 9.99871380e-001
        fparams[51][3] = 1.38462121e-001
        fparams[51][4] = 2.13972409e+000
        fparams[51][5] = 1.19666432e+001
        fparams[51][6] = 5.60566526e-001
        fparams[51][7] = 6.72672880e-001
        fparams[51][8] = 9.93772747e-001
        fparams[51][9] = 8.72330411e+000
        fparams[51][10] = 7.37374982e-002
        fparams[51][11] = 8.78577715e-002
        fparams[52][0] = 2.09383882e+000
        fparams[52][1] = 1.26856869e+001
        fparams[52][2] = 1.56940519e+000
        fparams[52][3] = 1.21236537e+000
        fparams[52][4] = 1.30941993e+000
        fparams[52][5] = 1.66633292e-001
        fparams[52][6] = 6.98067804e-002
        fparams[52][7] = 8.30817576e-002
        fparams[52][8] = 1.04969537e+000
        fparams[52][9] = 7.43147857e+000
        fparams[52][10] = 5.55594354e-001
        fparams[52][11] = 6.17487676e-001
        fparams[53][0] = 1.60186925e+000
        fparams[53][1] = 1.95031538e-001
        fparams[53][2] = 1.98510264e+000
        fparams[53][3] = 1.36976183e+001
        fparams[53][4] = 1.48226200e+000
        fparams[53][5] = 1.80304795e+000
        fparams[53][6] = 5.53807199e-001
        fparams[53][7] = 5.67912340e-001
        fparams[53][8] = 1.11728722e+000
        fparams[53][9] = 6.40879878e+000
        fparams[53][10] = 6.60720847e-002
        fparams[53][11] = 7.86615429e-002
        fparams[54][0] = 1.60015487e+000
        fparams[54][1] = 2.92913354e+000
        fparams[54][2] = 1.71644581e+000
        fparams[54][3] = 1.55882990e+001
        fparams[54][4] = 1.84968351e+000
        fparams[54][5] = 2.22525983e-001
        fparams[54][6] = 6.23813648e-002
        fparams[54][7] = 7.45581223e-002
        fparams[54][8] = 1.21387555e+000
        fparams[54][9] = 5.56013271e+000
        fparams[54][10] = 5.54051946e-001
        fparams[54][11] = 5.21994521e-001
        fparams[55][0] = 2.95236854e+000
        fparams[55][1] = 6.01461952e+000
        fparams[55][2] = 4.28105721e-001
        fparams[55][3] = 4.64151246e+001
        fparams[55][4] = 1.89599233e+000
        fparams[55][5] = 1.80109756e-001
        fparams[55][6] = 5.48012938e-002
        fparams[55][7] = 7.12799633e-002
        fparams[55][8] = 4.70838600e+000
        fparams[55][9] = 4.56702799e+001
        fparams[55][10] = 5.90356719e-001
        fparams[55][11] = 4.70236310e-001
        fparams[56][0] = 3.19434243e+000
        fparams[56][1] = 9.27352241e+000
        fparams[56][2] = 1.98289586e+000
        fparams[56][3] = 2.28741632e-001
        fparams[56][4] = 1.55121052e-001
        fparams[56][5] = 3.82000231e-002
        fparams[56][6] = 6.73222354e-002
        fparams[56][7] = 7.30961745e-002
        fparams[56][8] = 4.48474211e+000
        fparams[56][9] = 2.95703565e+001
        fparams[56][10] = 5.42674414e-001
        fparams[56][11] = 4.08647015e-001
        fparams[57][0] = 2.05036425e+000
        fparams[57][1] = 2.20348417e-001
        fparams[57][2] = 1.42114311e-001
        fparams[57][3] = 3.96438056e-002
        fparams[57][4] = 3.23538151e+000
        fparams[57][5] = 9.56979169e+000
        fparams[57][6] = 6.34683429e-002
        fparams[57][7] = 6.92443091e-002
        fparams[57][8] = 3.97960586e+000
        fparams[57][9] = 2.53178406e+001
        fparams[57][10] = 5.20116711e-001
        fparams[57][11] = 3.83614098e-001
        fparams[58][0] = 3.22990759e+000
        fparams[58][1] = 9.94660135e+000
        fparams[58][2] = 1.57618307e-001
        fparams[58][3] = 4.15378676e-002
        fparams[58][4] = 2.13477838e+000
        fparams[58][5] = 2.40480572e-001
        fparams[58][6] = 5.01907609e-001
        fparams[58][7] = 3.66252019e-001
        fparams[58][8] = 3.80889010e+000
        fparams[58][9] = 2.43275968e+001
        fparams[58][10] = 5.96625028e-002
        fparams[58][11] = 6.59653503e-002
        fparams[59][0] = 1.58189324e-001
        fparams[59][1] = 3.91309056e-002
        fparams[59][2] = 3.18141995e+000
        fparams[59][3] = 1.04139545e+001
        fparams[59][4] = 2.27622140e+000
        fparams[59][5] = 2.81671757e-001
        fparams[59][6] = 3.97705472e+000
        fparams[59][7] = 2.61872978e+001
        fparams[59][8] = 5.58448277e-002
        fparams[59][9] = 6.30921695e-002
        fparams[59][10] = 4.85207954e-001
        fparams[59][11] = 3.54234369e-001
        fparams[60][0] = 1.81379417e-001
        fparams[60][1] = 4.37324793e-002
        fparams[60][2] = 3.17616396e+000
        fparams[60][3] = 1.07842572e+001
        fparams[60][4] = 2.35221519e+000
        fparams[60][5] = 3.05571833e-001
        fparams[60][6] = 3.83125763e+000
        fparams[60][7] = 2.54745408e+001
        fparams[60][8] = 5.25889976e-002
        fparams[60][9] = 6.02676073e-002
        fparams[60][10] = 4.70090742e-001
        fparams[60][11] = 3.39017003e-001
        fparams[61][0] = 1.92986811e-001
        fparams[61][1] = 4.37785970e-002
        fparams[61][2] = 2.43756023e+000
        fparams[61][3] = 3.29336996e-001
        fparams[61][4] = 3.17248504e+000
        fparams[61][5] = 1.11259996e+001
        fparams[61][6] = 3.58105414e+000
        fparams[61][7] = 2.46709586e+001
        fparams[61][8] = 4.56529394e-001
        fparams[61][9] = 3.24990282e-001
        fparams[61][10] = 4.94812177e-002
        fparams[61][11] = 5.76553100e-002
        fparams[62][0] = 2.12002595e-001
        fparams[62][1] = 4.57703608e-002
        fparams[62][2] = 3.16891754e+000
        fparams[62][3] = 1.14536599e+001
        fparams[62][4] = 2.51503494e+000
        fparams[62][5] = 3.55561054e-001
        fparams[62][6] = 4.44080845e-001
        fparams[62][7] = 3.11953363e-001
        fparams[62][8] = 3.36742101e+000
        fparams[62][9] = 2.40291435e+001
        fparams[62][10] = 4.65652543e-002
        fparams[62][11] = 5.52266819e-002
        fparams[63][0] = 2.59355002e+000
        fparams[63][1] = 3.82452612e-001
        fparams[63][2] = 3.16557522e+000
        fparams[63][3] = 1.17675155e+001
        fparams[63][4] = 2.29402652e-001
        fparams[63][5] = 4.76642249e-002
        fparams[63][6] = 4.32257780e-001
        fparams[63][7] = 2.99719833e-001
        fparams[63][8] = 3.17261920e+000
        fparams[63][9] = 2.34462738e+001
        fparams[63][10] = 4.37958317e-002
        fparams[63][11] = 5.29440680e-002
        fparams[64][0] = 3.19144939e+000
        fparams[64][1] = 1.20224655e+001
        fparams[64][2] = 2.55766431e+000
        fparams[64][3] = 4.08338876e-001
        fparams[64][4] = 3.32681934e-001
        fparams[64][5] = 5.85819814e-002
        fparams[64][6] = 4.14243130e-002
        fparams[64][7] = 5.06771477e-002
        fparams[64][8] = 2.61036728e+000
        fparams[64][9] = 1.99344244e+001
        fparams[64][10] = 4.20526863e-001
        fparams[64][11] = 2.85686240e-001
        fparams[65][0] = 2.59407462e-001
        fparams[65][1] = 5.04689354e-002
        fparams[65][2] = 3.16177855e+000
        fparams[65][3] = 1.23140183e+001
        fparams[65][4] = 2.75095751e+000
        fparams[65][5] = 4.38337626e-001
        fparams[65][6] = 2.79247686e+000
        fparams[65][7] = 2.23797309e+001
        fparams[65][8] = 3.85931001e-002
        fparams[65][9] = 4.87920992e-002
        fparams[65][10] = 4.10881708e-001
        fparams[65][11] = 2.77622892e-001
        fparams[66][0] = 3.16055396e+000
        fparams[66][1] = 1.25470414e+001
        fparams[66][2] = 2.82751709e+000
        fparams[66][3] = 4.67899094e-001
        fparams[66][4] = 2.75140255e-001
        fparams[66][5] = 5.23226982e-002
        fparams[66][6] = 4.00967160e-001
        fparams[66][7] = 2.67614884e-001
        fparams[66][8] = 2.63110834e+000
        fparams[66][9] = 2.19498166e+001
        fparams[66][10] = 3.61333817e-002
        fparams[66][11] = 4.68871497e-002
        fparams[67][0] = 2.88642467e-001
        fparams[67][1] = 5.40507687e-002
        fparams[67][2] = 2.90567296e+000
        fparams[67][3] = 4.97581077e-001
        fparams[67][4] = 3.15960159e+000
        fparams[67][5] = 1.27599505e+001
        fparams[67][6] = 3.91280259e-001
        fparams[67][7] = 2.58151831e-001
        fparams[67][8] = 2.48596038e+000
        fparams[67][9] = 2.15400972e+001
        fparams[67][10] = 3.37664478e-002
        fparams[67][11] = 4.50664323e-002
        fparams[68][0] = 3.15573213e+000
        fparams[68][1] = 1.29729009e+001
        fparams[68][2] = 3.11519560e-001
        fparams[68][3] = 5.81399387e-002
        fparams[68][4] = 2.97722406e+000
        fparams[68][5] = 5.31213394e-001
        fparams[68][6] = 3.81563854e-001
        fparams[68][7] = 2.49195776e-001
        fparams[68][8] = 2.40247532e+000
        fparams[68][9] = 2.13627616e+001
        fparams[68][10] = 3.15224214e-002
        fparams[68][11] = 4.33253257e-002
        fparams[69][0] = 3.15591970e+000
        fparams[69][1] = 1.31232407e+001
        fparams[69][2] = 3.22544710e-001
        fparams[69][3] = 5.97223323e-002
        fparams[69][4] = 3.05569053e+000
        fparams[69][5] = 5.61876773e-001
        fparams[69][6] = 2.92845100e-002
        fparams[69][7] = 4.16534255e-002
        fparams[69][8] = 3.72487205e-001
        fparams[69][9] = 2.40821967e-001
        fparams[69][10] = 2.27833695e+000
        fparams[69][11] = 2.10034185e+001
        fparams[70][0] = 3.10794704e+000
        fparams[70][1] = 6.06347847e-001
        fparams[70][2] = 3.14091221e+000
        fparams[70][3] = 1.33705269e+001
        fparams[70][4] = 3.75660454e-001
        fparams[70][5] = 7.29814740e-002
        fparams[70][6] = 3.61901097e-001
        fparams[70][7] = 2.32652051e-001
        fparams[70][8] = 2.45409082e+000
        fparams[70][9] = 2.12695209e+001
        fparams[70][10] = 2.72383990e-002
        fparams[70][11] = 3.99969597e-002
        fparams[71][0] = 3.11446863e+000
        fparams[71][1] = 1.38968881e+001
        fparams[71][2] = 5.39634353e-001
        fparams[71][3] = 8.91708508e-002
        fparams[71][4] = 3.06460915e+000
        fparams[71][5] = 6.79919563e-001
        fparams[71][6] = 2.58563745e-002
        fparams[71][7] = 3.82808522e-002
        fparams[71][8] = 2.13983556e+000
        fparams[71][9] = 1.80078788e+001
        fparams[71][10] = 3.47788231e-001
        fparams[71][11] = 2.22706591e-001
        fparams[72][0] = 3.01166899e+000
        fparams[72][1] = 7.10401889e-001
        fparams[72][2] = 3.16284788e+000
        fparams[72][3] = 1.38262192e+001
        fparams[72][4] = 6.33421771e-001
        fparams[72][5] = 9.48486572e-002
        fparams[72][6] = 3.41417198e-001
        fparams[72][7] = 2.14129678e-001
        fparams[72][8] = 1.53566013e+000
        fparams[72][9] = 1.55298698e+001
        fparams[72][10] = 2.40723773e-002
        fparams[72][11] = 3.67833690e-002
        fparams[73][0] = 3.20236821e+000
        fparams[73][1] = 1.38446369e+001
        fparams[73][2] = 8.30098413e-001
        fparams[73][3] = 1.18381581e-001
        fparams[73][4] = 2.86552297e+000
        fparams[73][5] = 7.66369118e-001
        fparams[73][6] = 2.24813887e-002
        fparams[73][7] = 3.52934622e-002
        fparams[73][8] = 1.40165263e+000
        fparams[73][9] = 1.46148877e+001
        fparams[73][10] = 3.33740596e-001
        fparams[73][11] = 2.05704486e-001
        fparams[74][0] = 9.24906855e-001
        fparams[74][1] = 1.28663377e-001
        fparams[74][2] = 2.75554557e+000
        fparams[74][3] = 7.65826479e-001
        fparams[74][4] = 3.30440060e+000
        fparams[74][5] = 1.34471170e+001
        fparams[74][6] = 3.29973862e-001
        fparams[74][7] = 1.98218895e-001
        fparams[74][8] = 1.09916444e+000
        fparams[74][9] = 1.35087534e+001
        fparams[74][10] = 2.06498883e-002
        fparams[74][11] = 3.38918459e-002
        fparams[75][0] = 1.96952105e+000
        fparams[75][1] = 4.98830620e+001
        fparams[75][2] = 1.21726619e+000
        fparams[75][3] = 1.33243809e-001
        fparams[75][4] = 4.10391685e+000
        fparams[75][5] = 1.84396916e+000
        fparams[75][6] = 2.90791978e-002
        fparams[75][7] = 2.84192813e-002
        fparams[75][8] = 2.30696669e-001
        fparams[75][9] = 1.90968784e-001
        fparams[75][10] = 6.08840299e-001
        fparams[75][11] = 1.37090356e+000
        fparams[76][0] = 2.06385867e+000
        fparams[76][1] = 4.05671697e+001
        fparams[76][2] = 1.29603406e+000
        fparams[76][3] = 1.46559047e-001
        fparams[76][4] = 3.96920673e+000
        fparams[76][5] = 1.82561596e+000
        fparams[76][6] = 2.69835487e-002
        fparams[76][7] = 2.84172045e-002
        fparams[76][8] = 2.31083999e-001
        fparams[76][9] = 1.79765184e-001
        fparams[76][10] = 6.30466774e-001
        fparams[76][11] = 1.38911543e+000
        fparams[77][0] = 2.21522726e+000
        fparams[77][1] = 3.24464090e+001
        fparams[77][2] = 1.37573155e+000
        fparams[77][3] = 1.60920048e-001
        fparams[77][4] = 3.78244405e+000
        fparams[77][5] = 1.78756553e+000
        fparams[77][6] = 2.44643240e-002
        fparams[77][7] = 2.82909938e-002
        fparams[77][8] = 2.36932016e-001
        fparams[77][9] = 1.70692368e-001
        fparams[77][10] = 6.48471412e-001
        fparams[77][11] = 1.37928390e+000
        fparams[78][0] = 9.84697940e-001
        fparams[78][1] = 1.60910839e-001
        fparams[78][2] = 2.73987079e+000
        fparams[78][3] = 7.18971667e-001
        fparams[78][4] = 3.61696715e+000
        fparams[78][5] = 1.29281016e+001
        fparams[78][6] = 3.02885602e-001
        fparams[78][7] = 1.70134854e-001
        fparams[78][8] = 2.78370726e-001
        fparams[78][9] = 1.49862703e+000
        fparams[78][10] = 1.52124129e-002
        fparams[78][11] = 2.83510822e-002
        fparams[79][0] = 9.61263398e-001
        fparams[79][1] = 1.70932277e-001
        fparams[79][2] = 3.69581030e+000
        fparams[79][3] = 1.29335319e+001
        fparams[79][4] = 2.77567491e+000
        fparams[79][5] = 6.89997070e-001
        fparams[79][6] = 2.95414176e-001
        fparams[79][7] = 1.63525510e-001
        fparams[79][8] = 3.11475743e-001
        fparams[79][9] = 1.39200901e+000
        fparams[79][10] = 1.43237267e-002
        fparams[79][11] = 2.71265337e-002
        fparams[80][0] = 1.29200491e+000
        fparams[80][1] = 1.83432865e-001
        fparams[80][2] = 2.75161478e+000
        fparams[80][3] = 9.42368371e-001
        fparams[80][4] = 3.49387949e+000
        fparams[80][5] = 1.46235654e+001
        fparams[80][6] = 2.77304636e-001
        fparams[80][7] = 1.55110144e-001
        fparams[80][8] = 4.30232810e-001
        fparams[80][9] = 1.28871670e+000
        fparams[80][10] = 1.48294351e-002
        fparams[80][11] = 2.61903834e-002
        fparams[81][0] = 3.75964730e+000
        fparams[81][1] = 1.35041513e+001
        fparams[81][2] = 3.21195904e+000
        fparams[81][3] = 6.66330993e-001
        fparams[81][4] = 6.47767825e-001
        fparams[81][5] = 9.22518234e-002
        fparams[81][6] = 2.76123274e-001
        fparams[81][7] = 1.50312897e-001
        fparams[81][8] = 3.18838810e-001
        fparams[81][9] = 1.12565588e+000
        fparams[81][10] = 1.31668419e-002
        fparams[81][11] = 2.48879842e-002
        fparams[82][0] = 1.00795975e+000
        fparams[82][1] = 1.17268427e-001
        fparams[82][2] = 3.09796153e+000
        fparams[82][3] = 8.80453235e-001
        fparams[82][4] = 3.61296864e+000
        fparams[82][5] = 1.47325812e+001
        fparams[82][6] = 2.62401476e-001
        fparams[82][7] = 1.43491014e-001
        fparams[82][8] = 4.05621995e-001
        fparams[82][9] = 1.04103506e+000
        fparams[82][10] = 1.31812509e-002
        fparams[82][11] = 2.39575415e-002
        fparams[83][0] = 1.59826875e+000
        fparams[83][1] = 1.56897471e-001
        fparams[83][2] = 4.38233925e+000
        fparams[83][3] = 2.47094692e+000
        fparams[83][4] = 2.06074719e+000
        fparams[83][5] = 5.72438972e+001
        fparams[83][6] = 1.94426023e-001
        fparams[83][7] = 1.32979109e-001
        fparams[83][8] = 8.22704978e-001
        fparams[83][9] = 9.56532528e-001
        fparams[83][10] = 2.33226953e-002
        fparams[83][11] = 2.23038435e-002
        fparams[84][0] = 1.71463223e+000
        fparams[84][1] = 9.79262841e+001
        fparams[84][2] = 2.14115960e+000
        fparams[84][3] = 2.10193717e-001
        fparams[84][4] = 4.37512413e+000
        fparams[84][5] = 3.66948812e+000
        fparams[84][6] = 2.16216680e-002
        fparams[84][7] = 1.98456144e-002
        fparams[84][8] = 1.97843837e-001
        fparams[84][9] = 1.33758807e-001
        fparams[84][10] = 6.52047920e-001
        fparams[84][11] = 7.80432104e-001
        fparams[85][0] = 1.48047794e+000
        fparams[85][1] = 1.25943919e+002
        fparams[85][2] = 2.09174630e+000
        fparams[85][3] = 1.83803008e-001
        fparams[85][4] = 4.75246033e+000
        fparams[85][5] = 4.19890596e+000
        fparams[85][6] = 1.85643958e-002
        fparams[85][7] = 1.81383503e-002
        fparams[85][8] = 2.05859375e-001
        fparams[85][9] = 1.33035404e-001
        fparams[85][10] = 7.13540948e-001
        fparams[85][11] = 7.03031938e-001
        fparams[86][0] = 6.30022295e-001
        fparams[86][1] = 1.40909762e-001
        fparams[86][2] = 3.80962881e+000
        fparams[86][3] = 3.08515540e+001
        fparams[86][4] = 3.89756067e+000
        fparams[86][5] = 6.51559763e-001
        fparams[86][6] = 2.40755100e-001
        fparams[86][7] = 1.08899672e-001
        fparams[86][8] = 2.62868577e+000
        fparams[86][9] = 6.42383261e+000
        fparams[86][10] = 3.14285931e-002
        fparams[86][11] = 2.42346699e-002
        fparams[87][0] = 5.23288135e+000
        fparams[87][1] = 8.60599536e+000
        fparams[87][2] = 2.48604205e+000
        fparams[87][3] = 3.04543982e-001
        fparams[87][4] = 3.23431354e-001
        fparams[87][5] = 3.87759096e-002
        fparams[87][6] = 2.55403596e-001
        fparams[87][7] = 1.28717724e-001
        fparams[87][8] = 5.53607228e-001
        fparams[87][9] = 5.36977452e-001
        fparams[87][10] = 5.75278889e-003
        fparams[87][11] = 1.29417790e-002
        fparams[88][0] = 1.44192685e+000
        fparams[88][1] = 1.18740873e-001
        fparams[88][2] = 3.55291725e+000
        fparams[88][3] = 1.01739750e+000
        fparams[88][4] = 3.91259586e+000
        fparams[88][5] = 6.31814783e+001
        fparams[88][6] = 2.16173519e-001
        fparams[88][7] = 9.55806441e-002
        fparams[88][8] = 3.94191605e+000
        fparams[88][9] = 3.50602732e+001
        fparams[88][10] = 4.60422605e-002
        fparams[88][11] = 2.20850385e-002
        fparams[89][0] = 1.45864127e+000
        fparams[89][1] = 1.07760494e-001
        fparams[89][2] = 4.18945405e+000
        fparams[89][3] = 8.89090649e+001
        fparams[89][4] = 3.65866182e+000
        fparams[89][5] = 1.05088931e+000
        fparams[89][6] = 2.08479229e-001
        fparams[89][7] = 9.09335557e-002
        fparams[89][8] = 3.16528117e+000
        fparams[89][9] = 3.13297788e+001
        fparams[89][10] = 5.23892556e-002
        fparams[89][11] = 2.08807697e-002
        fparams[90][0] = 1.19014064e+000
        fparams[90][1] = 7.73468729e-002
        fparams[90][2] = 2.55380607e+000
        fparams[90][3] = 6.59693681e-001
        fparams[90][4] = 4.68110181e+000
        fparams[90][5] = 1.28013896e+001
        fparams[90][6] = 2.26121303e-001
        fparams[90][7] = 1.08632194e-001
        fparams[90][8] = 3.58250545e-001
        fparams[90][9] = 4.56765664e-001
        fparams[90][10] = 7.82263950e-003
        fparams[90][11] = 1.62623474e-002
        fparams[91][0] = 4.68537504e+000
        fparams[91][1] = 1.44503632e+001
        fparams[91][2] = 2.98413708e+000
        fparams[91][3] = 5.56438592e-001
        fparams[91][4] = 8.91988061e-001
        fparams[91][5] = 6.69512914e-002
        fparams[91][6] = 2.24825384e-001
        fparams[91][7] = 1.03235396e-001
        fparams[91][8] = 3.04444846e-001
        fparams[91][9] = 4.27255647e-001
        fparams[91][10] = 9.48162708e-003
        fparams[91][11] = 1.77730611e-002
        fparams[92][0] = 4.63343606e+000
        fparams[92][1] = 1.63377267e+001
        fparams[92][2] = 3.18157056e+000
        fparams[92][3] = 5.69517868e-001
        fparams[92][4] = 8.76455075e-001
        fparams[92][5] = 6.88860012e-002
        fparams[92][6] = 2.21685477e-001
        fparams[92][7] = 9.84254550e-002
        fparams[92][8] = 2.72917100e-001
        fparams[92][9] = 4.09470917e-001
        fparams[92][10] = 1.11737298e-002
        fparams[92][11] = 1.86215410e-002
        fparams[93][0] = 4.56773888e+000
        fparams[93][1] = 1.90992795e+001
        fparams[93][2] = 3.40325179e+000
        fparams[93][3] = 5.90099634e-001
        fparams[93][4] = 8.61841923e-001
        fparams[93][5] = 7.03204851e-002
        fparams[93][6] = 2.19728870e-001
        fparams[93][7] = 9.36334280e-002
        fparams[93][8] = 2.38176903e-001
        fparams[93][9] = 3.93554882e-001
        fparams[93][10] = 1.38306499e-002
        fparams[93][11] = 1.94437286e-002
        fparams[94][0] = 5.45671123e+000
        fparams[94][1] = 1.01892720e+001
        fparams[94][2] = 1.11687906e-001
        fparams[94][3] = 3.98131313e-002
        fparams[94][4] = 3.30260343e+000
        fparams[94][5] = 3.14622212e-001
        fparams[94][6] = 1.84568319e-001
        fparams[94][7] = 1.04220860e-001
        fparams[94][8] = 4.93644263e-001
        fparams[94][9] = 4.63080540e-001
        fparams[94][10] = 3.57484743e+000
        fparams[94][11] = 2.19369542e+001
        fparams[95][0] = 5.38321999e+000
        fparams[95][1] = 1.07289857e+001
        fparams[95][2] = 1.23343236e-001
        fparams[95][3] = 4.15137806e-002
        fparams[95][4] = 3.46469090e+000
        fparams[95][5] = 3.39326208e-001
        fparams[95][6] = 1.75437132e-001
        fparams[95][7] = 9.98932346e-002
        fparams[95][8] = 3.39800073e+000
        fparams[95][9] = 2.11601535e+001
        fparams[95][10] = 4.69459519e-001
        fparams[95][11] = 4.51996970e-001
        fparams[96][0] = 5.38402377e+000
        fparams[96][1] = 1.11211419e+001
        fparams[96][2] = 3.49861264e+000
        fparams[96][3] = 3.56750210e-001
        fparams[96][4] = 1.88039547e-001
        fparams[96][5] = 5.39853583e-002
        fparams[96][6] = 1.69143137e-001
        fparams[96][7] = 9.60082633e-002
        fparams[96][8] = 3.19595016e+000
        fparams[96][9] = 1.80694389e+001
        fparams[96][10] = 4.64393059e-001
        fparams[96][11] = 4.36318197e-001
        fparams[97][0] = 3.66090688e+000
        fparams[97][1] = 3.84420906e-001
        fparams[97][2] = 2.03054678e-001
        fparams[97][3] = 5.48547131e-002
        fparams[97][4] = 5.30697515e+000
        fparams[97][5] = 1.17150262e+001
        fparams[97][6] = 1.60934046e-001
        fparams[97][7] = 9.21020329e-002
        fparams[97][8] = 3.04808401e+000
        fparams[97][9] = 1.73525367e+001
        fparams[97][10] = 4.43610295e-001
        fparams[97][11] = 4.27132359e-001
        fparams[98][0] = 3.94150390e+000
        fparams[98][1] = 4.18246722e-001
        fparams[98][2] = 5.16915345e+000
        fparams[98][3] = 1.25201788e+001
        fparams[98][4] = 1.61941074e-001
        fparams[98][5] = 4.81540117e-002
        fparams[98][6] = 4.15299561e-001
        fparams[98][7] = 4.24913856e-001
        fparams[98][8] = 2.91761325e+000
        fparams[98][9] = 1.90899693e+001
        fparams[98][10] = 1.51474927e-001
        fparams[98][11] = 8.81568925e-002
        fparams[99][0] = 4.09780623e+000
        fparams[99][1] = 4.46021145e-001
        fparams[99][2] = 5.10079393e+000
        fparams[99][3] = 1.31768613e+001
        fparams[99][4] = 1.74617289e-001
        fparams[99][5] = 5.02742829e-002
        fparams[99][6] = 2.76774658e+000
        fparams[99][7] = 1.84815393e+001
        fparams[99][8] = 1.44496639e-001
        fparams[99][9] = 8.46232592e-002
        fparams[99][10] = 4.02772109e-001
        fparams[99][11] = 4.17640100e-001
        fparams[100][0] = 4.24934820e+000
        fparams[100][1] = 4.75263933e-001
        fparams[100][2] = 5.03556594e+000
        fparams[100][3] = 1.38570834e+001
        fparams[100][4] = 1.88920613e-001
        fparams[100][5] = 5.26975158e-002
        fparams[100][6] = 3.94356058e-001
        fparams[100][7] = 4.11193751e-001
        fparams[100][8] = 2.61213100e+000
        fparams[100][9] = 1.78537905e+001
        fparams[100][10] = 1.38001927e-001
        fparams[100][11] = 8.12774434e-002
        fparams[101][0] = 2.00942931e-001
        fparams[101][1] = 5.48366518e-002
        fparams[101][2] = 4.40119869e+000
        fparams[101][3] = 5.04248434e-001
        fparams[101][4] = 4.97250102e+000
        fparams[101][5] = 1.45721366e+001
        fparams[101][6] = 2.47530599e+000
        fparams[101][7] = 1.72978308e+001
        fparams[101][8] = 3.86883197e-001
        fparams[101][9] = 4.05043898e-001
        fparams[101][10] = 1.31936095e-001
        fparams[101][11] = 7.80821071e-002
        fparams[102][0] = 2.16052899e-001
        fparams[102][1] = 5.83584058e-002
        fparams[102][2] = 4.91106799e+000
        fparams[102][3] = 1.53264212e+001
        fparams[102][4] = 4.54862870e+000
        fparams[102][5] = 5.34434760e-001
        fparams[102][6] = 2.36114249e+000
        fparams[102][7] = 1.68164803e+001
        fparams[102][8] = 1.26277292e-001
        fparams[102][9] = 7.50304633e-002
        fparams[102][10] = 3.81364501e-001
        fparams[102][11] = 3.99305852e-001
        fparams[103][0] = 4.86738014e+000
        fparams[103][1] = 1.60320520e+001
        fparams[103][2] = 3.19974401e-001
        fparams[103][3] = 6.70871138e-002
        fparams[103][4] = 4.58872425e+000
        fparams[103][5] = 5.77039373e-001
        fparams[103][6] = 1.21482448e-001
        fparams[103][7] = 7.22275899e-002
        fparams[103][8] = 2.31639872e+000
        fparams[103][9] = 1.41279737e+001
        fparams[103][10] = 3.79258137e-001
        fparams[103][11] = 3.89973484e-001

        PhaseImagingSystem.fe_table_read = 1
        PhaseImagingSystem.fparams = fparams
        return (nzmax - nzmin + 1) * npmax

    def transmit_wave(self, wave, trans):
        wave = wave.real * trans.real - wave.imag * trans.imag + (wave.real * trans.imag + wave.imag
                                                                 * trans.real) * 1j
        return wave

    @jit
    def propagate_wave(self, wave, propx, propy, k2, k2max):
        M = self.M

        for i in range(M):
            if True:#k2[i] < k2max:
                pxr = propx[i].real
                pxi = propx[i].imag
                for j in range(M):
                    if k2[i] + k2[j] < k2max:
                        pyr = propy[j].real
                        pyi = propy[j].imag
                        wr = wave[i, j].real
                        wi = wave[i, j].imag
                        tr = wr * pyr - wi * pyi
                        ti = wr * pyi + wi * pyr
                        wave[i, j] = tr * pxr - ti * pxi + (tr * pxi + ti * pxr) * 1j
                    else:
                        wave[i, j] = 0 + 0j
            else:
                for j in range(M):
                    wave[i, j] = 0 + 0j

        return wave

    @staticmethod
    def calculate_integrated_intensity(wave):
        M = len(wave)
        scale = 1 / (M * M)
        tot = 0
        for i in range(M):
            for j in range(M):
                tot += wave[i, j].real * wave[i, j].real + wave[i, j].imag * wave[i, j].imag
        tot *= scale

        return tot


    def project_phase_ms(self, axis, angle):

        """
        WARNING. This function is not working at all. It is being worked on in my phaseimaging library, which
        will be used here when it is finished.


        Multislice simulation method
        Note that this code is based on the theory described in
        "Advanced Computing in Electrons Microscopy" (Plenum 1998, Springer 2010) by Earl J. Kirkland,
        and much of it is adapted from Kirkland's C code, available at
        http://people.ccmr.cornell.edu/~kirkland/cdownloads.html
        :return:
        """
        # Convert list of attoms to numpy array for consistency
        location_list = self.atom_locations
        width = self.image_width
        aA = width * 1e10

        M = self.M
        xhat = np.array([1, 0, 0])
        yhat = np.array([0, 1, 0])
        zhat = np.array([0, 0, 1])

        exit_wave = np.zeros((M, M), dtype=complex)

        rotation_center = aA / 2

        loc = np.zeros(3)
        loc_final = np.zeros(3)
        num_atoms_total = len(location_list)
        z_number = np.array([])
        x = np.array([])
        y = np.array([])
        z = np.array([])

        wobble = np.array([])


        num_atoms = 0
        print("Reading in Atoms...")
        prog_bar = pyprind.ProgBar(num_atoms_total)
        for atoms_read in range(num_atoms_total):
            prog_bar.update()
            loc = np.array(location_list[atoms_read][:3])
            r = self.rotation_matrix(axis, angle)

            # TODO: Remove the second line and check that rotation is working
            loc_final = np.dot(r, loc - rotation_center) + rotation_center
            loc_final = loc

            atom_in_range = True
            for i in range(3):
                if loc_final[i] <= 0 or loc_final[i] >= aA:
                    atom_in_range = False
                    break
            if atom_in_range:

                z_number = np.append(z_number, location_list[atoms_read][3])
                x = np.append(x, loc_final[0])
                y = np.append(y, loc_final[1])
                z = np.append(z, loc_final[2])

                wobble = np.append(wobble, 0.078)
                num_atoms += 1
        print("There are a total of {0} atoms in the specimen".format(num_atoms))
        deltaz = 5  # Slice thickness in Angstrom
        wave = np.zeros(list((M, M)), dtype=complex)  # Incident beam
        trans = np.zeros(list((M, M)), dtype=complex)  # Transmission function
        temp = np.zeros(list((M, M)), dtype=complex)  # Scratch wavefunction

        cs3 = 5e6  # Spherical abberation in Angstrom
        cs5 = 0
        aobj = 100e-3  # Objective aperture in radians

        wavelength = self.wavelength * 1e10  # Convert to Angstroms

        xmin = xmax = x[0]
        ymin = ymax = y[0]
        zmin = zmax = z[0]
        for i in range(num_atoms):
            if x[i] < xmin:
                xmin = x[i]
            elif x[i] > xmax:
                xmax = x[i]
            if y[i] < ymin:
                ymin = y[i]
            elif y[i] > ymax:
                ymax = y[i]
            if z[i] < zmin:
                zmin = z[i]
            elif z[i] > zmax:
                zmax = z[i]
        k1 = 1 / aA
        k2 = k1 * k1

        bandwidth_limit = 2 / 3
        k2max = bandwidth_limit * M / (2 * aA)
        print("Bandwidth limited to a real space resolution of {0} Angstroms".format(1 / k2max))
        k2max = k2max * k2max

        k2, window = self.freqn(M, aA, k2max)
        propx = np.zeros(M, dtype=complex)
        propy = np.zeros(M, dtype=complex)

        scale = PI * deltaz

        for i in range(M):
            t = scale * k2[i] * wavelength
            propx[i] = np.cos(t) - np.sin(t) * 1j
            propy[i] = np.cos(t) - np.sin(t) * 1j

        # Initialise incident wave to unity everywhere
        for i in range(M):
            for j in range(M):
                wave[i, j] = 1 + 0j

        x, y, z, z_number = self.sort_by_z(x, y, z, z_number)
        zmin = z[0]
        zmax = z[num_atoms-1]

        print("Transmitting wave...")
        zslice = 0.75 * deltaz  # Start a little before top of unit cell
        istart = 0
        approx_num_slices = int(aA / deltaz)

        prog_bar = pyprind.ProgBar(approx_num_slices, sys.stdout)

        while zslice < aA:
            prog_bar.update()
            if istart < num_atoms:
                na = 0
                for i in range(istart, num_atoms):
                    if z[i] < zslice:
                        na += 1
                    else:
                        break
                # Calculate transmission function. Skip if layer is empty
                if na > 0:
                    trans = self.trlayer(x[istart:num_atoms],
                                         y[istart:num_atoms],
                                         z_number[istart:num_atoms],
                                         na, aA, trans, window)

                    print("Transmit wave...")
                    wave = self.transmit_wave(wave, trans)
            wave = fft.fft2(wave)
            wave = self.propagate_wave(wave, propx, propy, k2, k2max)

            wave = fft.ifft2(wave)
            zslice += deltaz
            istart += na

        print("Calculate lii...")
        integrated_intensity = PhaseImagingSystem.calculate_integrated_intensity(wave)

        print("Integrated Intensity: ", integrated_intensity)
        if integrated_intensity < self.lowest_integrated_intensity:
            self.lowest_integrated_intensity = integrated_intensity
        return fft.ifftshift(wave)

    def _add_noise(self, image):

        try:
            for i in range(len(image)):
                for j in range(len(image[0])):
                    if image[i, j] >= 0:
                        image[i, j] = np.random.poisson(image[i, j] /
                                          (self.noise_level *
                                           self.noise_level *
                                           self.image_intensity)) * (self.noise_level *
                                                                     self.noise_level *
                                                                     self.image_intensity)
        except ValueError:
            print("lam: ", print(image[i, j] /
                                      (self.noise_level *
                                       self.noise_level *
                                       self.image_intensity)))
            warnings.warn("value too high in add_noise()")


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
        if self.use_multislice:
            size = self.M
        else:
            size = self.image_size
        kernel = np.zeros(list((size, size)), dtype=complex)
        for i in range(size):
            for j in range(size):
                i0 = i - size / 2
                j0 = j - size / 2
                da = 1 / self.image_width
                kernel[i, j] = (i0 * i0) * da * da + j0 * j0 * da * da

        return kernel

    def _construct_k_kernel(self):
        kernel = np.zeros(list((self.image_size, self.image_size, 2)), dtype=complex)
        da = 1 / self.image_width
        for i in range(self.image_size):
            for j in range(self.image_size):
                i0 = i - self.image_size / 2
                j0 = j - self.image_size / 2
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

    def _import_wavefield(self, wavefield_file, pix):

        with open(wavefield_file, 'r') as f:
            f.seek(0)
            wave_temp = np.zeros((pix, pix), dtype=complex)

            for i in range(pix):
                line = f.readline().split()
                for j in range(pix):
                    wave_temp[i, j] = float(line[j * 2]) + float(line[j * 2 + 1]) * 1j
        return wave_temp

    @staticmethod
    def _downsample(input):
        input_len = len(input)
        ds_len = int(input_len / 2)
        ds = np.zeros((ds_len, ds_len), dtype=complex)
        for i in range(ds_len):
            for j in range(ds_len):
                sum_ = 0
                for p in range(2):
                    for q in range(2):
                        sum_ += input[2*i+p, 2*j+q]
                ds[i, j] = sum_ / 4

        return ds

    def _project_phase(self):
        if self.specimen_parameters['Use Electrostatic/Magnetic Potential'][0] and \
                self.specimen_parameters['Use Electrostatic/Magnetic Potential'][1]:
            self.phase_electrostatic = self._project_electrostatic_phase()
            self.phase_magnetic = self._project_magnetic_phase()
            self.phase_exact = self.phase_electrostatic + self.phase_magnetic
            if self.flipping:
                self.phase_reverse = self.phase_electrostatic - self.phase_magnetic
        elif self.specimen_parameters['Use Electrostatic/Magnetic Potential'][0]:
            self.phase_exact = self._project_electrostatic_phase()
        elif self.specimen_parameters['Use Electrostatic/Magnetic Potential'][1]:
            self.phase_exact = self._project_magnetic_phase()

    def _project_electrostatic_phase(self):
        """
        Computes the image plane phase shift of the specimen.

        :param self:
        :param orientation: either 0, 1, or 2, describes the axis of projection
        :return:
        """
        energy = self.energy
        wavelength = self.wavelength
        dz = self.image_width / self.specimen_size
        return np.sum(self.specimen,
                                  axis=0) * PI/(energy * wavelength) * self.mip * dz

    def _project_magnetic_phase(self):

        mhatcrossk_z = np.cross(PhaseImagingSystem.generate_random_axis(), self.k_kernel)[:, :, 2]
        D0 = fft.fftshift(fft.fftn(self.specimen))[int(self.image_size/2), :, :] * (self.image_width / self.image_size)

        phase = (1j * PI * mu0 * self.magnetisation / phi0) * D0 * self.inverse_k_squared_kernel * mhatcrossk_z
        phase = fft.ifftshift(phase)
        return fft.ifft2(phase)

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

    def _transfer_image(self, defocus, wavefunction, image):
        """
        Uses the defocus exact_phase (at the image plane) to produce an out of focus
        image or wavefield.

        :param self:
        :param defocus:
        :return:
        """
        self._set_transfer_function(defocus=defocus)

        wavefunction = fft.fft2(wavefunction)
        wavefunction = fft.fftshift(wavefunction)
        wavefunction = self.transfer_function * wavefunction
        wavefunction = fft.ifftshift(wavefunction)
        wavefunction = fft.ifft2(wavefunction)
        if image:
            return np.absolute(wavefunction) * np.absolute(wavefunction)
        else:
            return wavefunction

    def generate_images(self, n_images):
        assert n_images == 2 or n_images == 3
        """
        Compute images at under-, in-, and over-focus
        :return:
        """
        if self.use_multislice:
            wavefunction = self.wave_multislice_hr
        else:
            wavefunction = np.exp(1j * self.phase_exact)
        self.image_over = self._transfer_image(defocus=self.defocus, wavefunction=wavefunction, image=True)
        self.image_under = self._transfer_image(defocus=-self.defocus, wavefunction=wavefunction, image=True)
        if n_images == 3:
            self.image_in = self._transfer_image(defocus=0, wavefunction=wavefunction, image=True)
            while len(self.image_in) > self.image_size:
                self.image_in = PhaseImagingSystem._downsample(self.image_in)
        while len(self.image_under) > self.image_size:
            self.image_under = PhaseImagingSystem._downsample(self.image_under)
        while len(self.image_over) > self.image_size:
            self.image_over = PhaseImagingSystem._downsample(self.image_over)
        if self.flipping:
            if self.use_multislice:
                wavefunction = self.wave_multislice_hr
            else:
                wavefunction = np.exp(1j * self.phase_reverse)
            self.image_over_reverse = self._transfer_image(defocus=self.defocus, wavefunction=wavefunction, image=True)
            self.image_under_reverse = self._transfer_image(defocus=-self.defocus, wavefunction=wavefunction, image=True)
            if n_images == 3:
                self.image_in_reverse = self._transfer_image(defocus=0, wavefunction=wavefunction, image=True)
                while len(self.image_in_reverse) > self.image_size:
                    self.image_in_reverse = PhaseImagingSystem._downsample(self.image_in_reverse)
            while len(self.image_under_reverse) > self.image_size:
                self.image_under_reverse = PhaseImagingSystem._downsample(self.image_under_reverse)
            while len(self.image_over) > self.image_size:
                self.image_over_reverse = PhaseImagingSystem._downsample(self.image_over_reverse)
        return

    def approximate_in_focus(self):
        self.image_in = (self.image_over + self.image_under) / 2
        if self.flipping:
            self.image_in_reverse = (self.image_over_reverse + self.image_under_reverse) / 2

    def add_noise_to_micrographs(self):
        if self.flipping:
            images = [self.image_under,
                      self.image_in,
                      self.image_over,
                      self.image_under_reverse,
                      self.image_in_reverse,
                      self.image_over_reverse]
        else:
            images = [self.image_under,
                  self.image_in,
                  self.image_over]
        if self.noise_level != 0:
            for image in images:
                self._add_noise(image)
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

    def intensity_derivative(self, dir):
        if dir == 'forward':
            return (self.image_under - self.image_over) / (2 * self.defocus)
        elif dir == 'reverse':
            return (self.image_under_reverse - self.image_over_reverse) / (2 * self.defocus)

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
        if self.flipping:
            self.image_under_reverse = self.apodise(self.image_under_reverse, rad_sup)
            self.image_in_reverse = self.apodise(self.image_in_reverse, rad_sup)
            self.image_over_reverse = self.apodise(self.image_over_reverse, rad_sup)

    def apodise_phases(self, rad_sup):
        self.phase_exact = self.apodise(self.phase_exact, rad_sup)
        if self.flipping:
            self.phase_reverse = self.apodise(self.phase_reverse, rad_sup)
        self.phase_retrieved = self.apodise(self.phase_retrieved, rad_sup)

    def remove_offset(self):
        if self.flipping:
            if self.simulation_parameters['Retrieve Phase Component'] == 'electrostatic':
                offset = self.phase_retrieved - self.phase_electrostatic
            elif self.simulation_parameters['Retrieve Phase Component'] == 'magnetic':
                offset = self.phase_retrieved - self.phase_magnetic
        else:
            offset = self.phase_retrieved - self.phase_exact
        offset_avg = np.mean(np.mean(offset))
        self.phase_retrieved = self.phase_retrieved - offset_avg

    def retrieve_phase(self, method):
        if method == 'TIE':
            self.retrieve_phase_TIE()
        elif method == 'GS':
            self.retrieve_phase_GS()
        else:
            raise ValueError('Unknown Phase Retrieval Method')

    def retrieve_phase_GS(self):
        """
        Retrieve phase from under- and over-focus images using
        the Gerchberg-Saxton-Misell [1] algorithm.

        ##########################################################################
        ######                                                              ######
        ###### Note: this function has not been tested under any reasonable ######
        ###### set of parameters. Please test that it is working before     ######
        ###### using it.                                                    ######
        ######                                                              ######
        ##########################################################################

        [1] D L Misell 1973 J. Phys. D: Appl. Phys. 6 2200
        :return:
        """

        print(self.retrieve_phase_GS.__doc__)
        iterations = 200

        under_phase = np.random.uniform(0, 2 * PI, (self.image_size, self.image_size))
        over_amplitude = np.sqrt(np.clip(self.image_over, 0, None))
        under_amplitude = np.sqrt(np.clip(self.image_under, 0, None))

        under_prime = under_amplitude * np.exp(1j * under_phase)

        for i in range(iterations):
            over_prime = self._transfer_image(self.defocus * 2, under_prime, image=False)
            over_phase = np.angle(over_prime)
            over_prime = over_amplitude * np.exp(1j * over_phase)
            under_prime = self._transfer_image(-1 * self.defocus * 2, over_prime, image=False)
            under_phase = np.angle(under_prime)
            under_prime = under_amplitude * np.exp(1j * under_phase)

        result = np.angle(self._transfer_image(self.defocus, under_prime, image=False))
        self.phase_retrieved = unwrap(result, wrap_around_axis_0=True, wrap_around_axis_1=True)

    def retrieve_phase_TIE(self):
        """
        Utilise the transport-of-intensity equation to compute the phase from
        the micrographs.
        :return:
        """

        if self.is_attenuating:
            regularised_inverse_intensity = self.image_in / (self.image_in * self.image_in
                                                             + self.reg_image * self.reg_image)
            prefactor = (1. / self.wavelength) / (2. * PI)
            derivative = self.intensity_derivative('forward')
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


        else:
            prefactor = (1. / self.wavelength) / (2 * PI * self.image_intensity)

            filtered = self.convolve(self.intensity_derivative('forward'),
                                     self.inverse_k_squared_kernel)
            self.phase_retrieved = prefactor * filtered
        if self.flipping:
            if self.is_attenuating:
                regularised_inverse_intensity = self.image_in_reverse / (self.image_in_reverse * self.image_in_reverse
                                                                 + self.reg_image * self.reg_image)
                prefactor = (1. / self.wavelength) / (2. * PI)
                derivative = self.intensity_derivative('reverse')
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
                if self.simulation_parameters['Retrieve Phase Component'] == 'magnetic':
                    self.phase_retrieved = (self.phase_retrieved - prefactor * fft.ifft2(fft.ifftshift(derivative))) / 2
                elif self.simulation_parameters['Retrieve Phase Component'] == 'electrostatic':
                    self.phase_retrieved = (self.phase_retrieved + prefactor * fft.ifft2(fft.ifftshift(derivative))) / 2


            else:
                prefactor = (1. / self.wavelength) / (2 * PI * self.image_intensity)

                filtered = self.convolve(self.intensity_derivative('reverse'),
                                         self.inverse_k_squared_kernel)
                if self.simulation_parameters['Retrieve Phase Component'] == 'magnetic':
                    self.phase_retrieved = (self.phase_retrieved - prefactor * filtered) / 2
                elif self.simulation_parameters['Retrieve Phase Component'] == 'electrostatic':
                    self.phase_retrieved = (self.phase_retrieved + prefactor * filtered) / 2
        return












