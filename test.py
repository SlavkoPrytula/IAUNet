# # from configs import cfg

# # print(cfg.dataset.train)


# n = 20
# groups = 2
# for i in range(n):
#     for j in range(groups+1):
#         if j == 0:
#             pass
#         else:
#             channel_idx = n // groups * (j-1) + i
#             print(channel_idx)

#     print()

from scipy.constants import c, h, Planck
import numpy as np

# Constants
mass_electron = 9.31e-31  # Kg
velocity_light = c  # m/s

# EX1: Kinetic Energy = (1/2) * m * v^2
KE_electron = (1/2) * mass_electron * velocity_light ** 2

# EX2: Kinetic Energy of a particle having a wavelength λ = h / sqrt(2mE)
# E = (h^2) / (2mλ^2)
wavelength = 2e-12  # meters
KE_particle = (h ** 2) / (2 * mass_electron * (wavelength ** 2))

# EX3: Hounsfield Unit (HU) = 1000 * ((μ - μ_water) / (μ_water))
attenuation_brain = 0.237
attenuation_water = 0.214
HU_brain = 1000 * ((attenuation_brain - attenuation_water) / attenuation_water)

# EX5: Larmor Frequency = γ * B
# γ (gyromagnetic ratio) for proton = 42.58 MHz/T
B_field = 1.5  # T
gamma_proton = 42.58  # MHz/T
Larmor_frequency = gamma_proton * B_field

# EX7: Slice resolution Δz = γ * B0 * Gz
# Δf = γ * Gz * Δz
Gz = 10e-3  # T/m
slice_resolution = 10e-3  # m
step_size = gamma_proton * 1e6 * Gz * slice_resolution  # in Hz

# EX10: Numerical Aperture NA = n * sin(θ)
# sin(θ) = NA / n, for air n=1
NA = 0.5
n_air = 1
angular_aperture = np.arcsin(NA / n_air) * (180/np.pi)  # Convert from radians to degrees

# Conversion factors
joule_to_femtojoule = 1e15
joule_to_picojoule = 1e12

# Convert to appropriate units and round to match options
KE_electron_fJ = KE_electron * joule_to_femtojoule
KE_particle_pJ = KE_particle * joule_to_picojoule

print(KE_electron_fJ, KE_particle_pJ, HU_brain, Larmor_frequency, step_size, angular_aperture)


print(0.5 * 9.31 * 10e-31 * (3 * 10e8) ** 2)



R_fp = 10 / 100  # Convert cm to m
theta = np.deg2rad(15)  # Convert degrees to radians
speed_of_sound = 1540  # Speed of sound in soft tissue in m/s

# Calculate the propagation time t_j for the jth transducer for beamforming in ultrasound scanning
# First, we calculate the distance d_j from the jth transducer to point P using trigonometry.
# Given that R_fp is the hypotenuse of the right-angled triangle and theta is the angle at O,
# we can calculate the adjacent side (d_j), which is the direct path from the jth transducer to point P.
d_j = R_fp / np.cos(theta)  # The direct path distance
print(d_j, R_fp, np.cos(theta))

# Calculate the propagation time t_j
t_j = d_j / speed_of_sound  # time = distance / speed
print(t_j)

# Convert the propagation time to microseconds (us)
t_j_microseconds = t_j * 1e6  # Convert seconds to microseconds
t_j_microseconds
print(t_j_microseconds)

print( (13.5 ** 2 + 0.1 ** 2) ** 0.5 )

# Constants
R_fp = 0.1  # 10cm in meters
theta_degrees = 15  # angle in degrees
theta_radians = np.deg2rad(theta_degrees)  # convert angle to radians
speed_of_sound = 1540  # m/s in soft tissue
transducer_width = 1e-3  # transducer length in meters

# Calculate the horizontal distance from the origin to the point directly above P (x_j)
x_j = R_fp * np.tan(theta_radians)

# Calculate the direct distance d_j the sound wave would travel from the transducer to point P
d_j = np.sqrt(R_fp**2 + x_j**2)

# Calculate the propagation time t_j
t_j = d_j / speed_of_sound

# Convert propagation time to microseconds
t_j_microseconds = t_j * 1e6  # from seconds to microseconds

x_j, d_j, t_j, t_j_microseconds

print(x_j, d_j, t_j, t_j_microseconds)