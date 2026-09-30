# Physical constants
G = 9.81                  # gravity (m/s²)
RHO = 1.225               # air density (kg/m³)
CD = 0.47                 # drag coefficient (sphere baseline)
AREA = 0.01               # cross-sectional area (m²)
MASS = 10.0               # projectile mass (kg)

# Default simulation parameters
DEFAULT_VELOCITY = 100.0  # m/s
DEFAULT_ANGLES = [30.0, 45.0, 60.0]  # launch angles in degrees

# Wind settings
# Option 1: constant wind
# WIND_X = 10.0          # horizontal wind along x (m/s)
# WIND_Y = 5.0           # lateral wind along z (m/s)

# Option 2: altitude-dependent wind tables (preferred)
ALT_LEVELS = [0, 500, 1000, 2000, 3000]  # altitudes in meters
WIND_X = [0, 5, 10, 15, 20]              # horizontal wind (x) at altitudes
WIND_Y = [0, 0, 5, 5, 10]                # lateral wind (z) at altitudes