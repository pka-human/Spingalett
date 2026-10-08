# SPDX-License-Identifier: MIT
"""Builds the spingalett wheel. The package is pure Python over ctypes; when a shared library has been
copied into spingalett/ (libspingalett.so.*, libspingalett.*.dylib or libspingalett.dll, with the
runtime libraries it loads), the wheel carries it and is tagged for the platform but for any
Python 3 (py3-none-<platform>), since nothing in it is built against Python. Without a library the
wheel is pure and finds the library at run time (see spingalett/__init__.py)."""
import glob
import os

from setuptools import setup
from setuptools.dist import Distribution

HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARIES = [os.path.basename(p) for pattern in ("*.so*", "*.dylib", "*.dll")
             for p in glob.glob(os.path.join(HERE, "spingalett", pattern))]

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:                     # setuptools before 70.1
    from wheel.bdist_wheel import bdist_wheel


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = not LIBRARIES

    def get_tag(self):
        python, abi, platform = super().get_tag()
        return ("py3", "none", platform) if LIBRARIES else (python, abi, platform)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return bool(LIBRARIES)


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": PlatformWheel},
      package_data={"spingalett": ["py.typed"] + LIBRARIES})
