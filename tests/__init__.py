"""Test package.

The unittest runner resets the warning filters at run time, so a filter set on
import of the project packages does not survive into the test run. netCDF4's wheel
is compiled against a slightly older NumPy C-ABI than the pinned runtime and emits a
harmless "numpy.ndarray size changed" RuntimeWarning the first time its compiled
extension loads. Load it here, once, with that single message filtered, so the
already-imported module is reused silently during the tests.
"""

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed", category=RuntimeWarning)
    import netCDF4  # noqa: F401  (preloaded so the one-time ABI warning is filtered)
