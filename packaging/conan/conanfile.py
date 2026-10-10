# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""Conan 2 recipe for the library of this source tree (the shared libraries and their headers):

    conan create packaging/conan --build=missing                         # OpenMP, no GPU backend
    conan create packaging/conan --build=missing -o "&:vulkan=True"       # with the Vulkan backend
    conan create packaging/conan --build=missing -o "&:runtime_only=True" # the runtime alone

Consumers use find_package(Spingalett) and Spingalett::spingalett or Spingalett::runtime (CMakeDeps),
or pkg-config's spingalett and spingalett-runtime (PkgConfigDeps). The kernels are chosen at run time
(SPINGALETT_NATIVE_ARCH off), so the package runs on any processor of its architecture.
"""
import os
import re

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy, load, rmdir

required_conan_version = ">=2.0"


class SpingalettConan(ConanFile):
    name = "spingalett"
    description = ("Neural networks in C: training on the CPU and on GPUs through Vulkan, deployment "
                   "models from FP32 down to INT2, an inference engine for microcontrollers")
    license = "MIT"
    url = "https://github.com/pka-human/Spingalett"
    homepage = "https://github.com/pka-human/Spingalett"
    topics = ("neural-network", "deep-learning", "machine-learning", "inference", "quantization", "vulkan")
    package_type = "shared-library"
    settings = "os", "arch", "compiler", "build_type"
    options = {"openmp": [True, False], "vulkan": [True, False], "runtime_only": [True, False]}
    default_options = {"openmp": True, "vulkan": False, "runtime_only": False}

    @property
    def _root(self):
        return os.path.join(self.recipe_folder, "..", "..")

    def set_version(self):
        text = load(self, os.path.join(self._root, "CMakeLists.txt"))
        self.version = re.search(r"project\(Spingalett VERSION ([\d.]+)", text).group(1)

    def export_sources(self):
        for pattern in ("CMakeLists.txt", "LICENSE", "cmake/*", "Include/*", "Src/*"):
            copy(self, pattern, self._root, self.export_sources_folder)

    def configure(self):
        # a C library
        self.settings.rm_safe("compiler.libcxx")
        self.settings.rm_safe("compiler.cppstd")
        if self.options.runtime_only:
            self.options.rm_safe("vulkan")      # the runtime computes on the CPU

    def layout(self):
        cmake_layout(self)

    def validate(self):
        if self.settings.compiler == "msvc":
            raise ConanInvalidConfiguration(
                "the library is C23 built with GCC or Clang (MinGW on Windows); MSVC programs link its DLL")

    def requirements(self):
        if self.options.get_safe("vulkan"):
            self.requires("vulkan-headers/[>=1.3.290.0 <2]")

    def build_requirements(self):
        self.tool_requires("cmake/[>=3.21 <5]")
        if self.options.get_safe("vulkan"):
            # glslc compiles the kernels to SPIR-V; 2025 on knows bfloat16 (the matrix units' kernel)
            self.tool_requires("shaderc/[>=2025.3]")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["SPINGALETT_NATIVE_ARCH"] = False
        tc.cache_variables["BUILD_EXAMPLE"] = False
        tc.cache_variables["BUILD_TESTS"] = False
        tc.cache_variables["BUILD_APPS"] = False
        tc.cache_variables["BUILD_WITH_OPENMP"] = bool(self.options.openmp)
        tc.cache_variables["SPINGALETT_VULKAN"] = "ON" if self.options.get_safe("vulkan") else "OFF"
        tc.cache_variables["SPINGALETT_RUNTIME_ONLY"] = bool(self.options.runtime_only)
        # the build tree's outputs stay in the build folder (by default they go to Bin/ and Lib/ of the sources)
        tc.cache_variables["SPINGALETT_BIN_DIR"] = os.path.join(self.build_folder, "Bin").replace("\\", "/")
        tc.cache_variables["SPINGALETT_LIB_DIR"] = os.path.join(self.build_folder, "Lib").replace("\\", "/")
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(self, "LICENSE", self.source_folder, os.path.join(self.package_folder, "licenses"))
        CMake(self).install()
        # Conan's generators write the consumers' CMake and pkg-config files
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))
        rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "Spingalett")
        if not self.options.runtime_only:
            library = self.cpp_info.components["library"]
            library.set_property("cmake_target_name", "Spingalett::spingalett")
            library.set_property("pkg_config_name", "spingalett")
            library.libs = ["spingalett"]
            if self.options.get_safe("vulkan"):
                library.requires = ["vulkan-headers::vulkan-headers"]
        runtime = self.cpp_info.components["runtime"]
        runtime.set_property("cmake_target_name", "Spingalett::runtime")
        runtime.set_property("pkg_config_name", "spingalett-runtime")
        runtime.libs = ["spingalett-runtime"]
