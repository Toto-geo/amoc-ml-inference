"""Figure package: one generator per paper figure under ``scripts``."""

import warnings

# netCDF4's wheel is compiled against a slightly older NumPy C-ABI than the pinned
# runtime, which emits a harmless "numpy.ndarray size changed" RuntimeWarning on first
# import. Data and results are unaffected; silence only this specific message.
warnings.filterwarnings("ignore", message="numpy.ndarray size changed", category=RuntimeWarning)
