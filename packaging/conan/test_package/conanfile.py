# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
import os

from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.cmake import CMake, cmake_layout


class TestPackageConan(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    generators = "CMakeDeps", "CMakeToolchain", "VirtualRunEnv"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        if can_run(self):
            # the library writes xor.slett, which the runtime then runs
            model = []
            if not self.dependencies["spingalett"].options.runtime_only:
                self.run(os.path.join(self.cpp.build.bindir, "test_package"), env="conanrun")
                model = ["xor.slett"]
            self.run(" ".join([os.path.join(self.cpp.build.bindir, "test_runtime")] + model), env="conanrun")
