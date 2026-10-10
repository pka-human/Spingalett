# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
#
# Homebrew: brew tap pka-human/spingalett https://github.com/pka-human/Spingalett
#           brew install spingalett
class Spingalett < Formula
  desc "Neural networks in C: training on CPUs and GPUs, transformers, INT8 to INT2 deployment"
  homepage "https://github.com/pka-human/Spingalett"
  url "https://github.com/pka-human/Spingalett/archive/refs/tags/v1.2.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"
  head "https://github.com/pka-human/Spingalett.git", branch: "main"

  depends_on "cmake" => :build
  depends_on "ninja" => :build
  depends_on "shaderc" => :build
  depends_on "vulkan-headers" => :build

  on_macos do
    depends_on "libomp"
  end

  def install
    args = %W[
      -DSPINGALETT_NATIVE_ARCH=OFF
      -DBUILD_WITH_OPENMP=ON
      -DSPINGALETT_VULKAN=ON
      -DBUILD_APPS=OFF
      -DBUILD_TESTS=OFF
      -DSPINGALETT_BIN_DIR=#{buildpath}/out/Bin
      -DSPINGALETT_LIB_DIR=#{buildpath}/out/Lib
    ]
    if OS.mac?
      omp = Formula["libomp"]
      args += [
        "-DOpenMP_C_FLAGS=-Xclang -fopenmp -I#{omp.opt_include}",
        "-DOpenMP_C_LIB_NAMES=omp",
        "-DOpenMP_omp_LIBRARY=#{omp.opt_lib}/libomp.dylib",
      ]
    end
    system "cmake", "-S", ".", "-B", "build", "-G", "Ninja", *args, *std_cmake_args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  test do
    (testpath/"test.c").write <<~C
      #include <Spingalett/Spingalett.h>
      #include <stdio.h>
      int main(void) {
          SpingalettNetwork *net = spingalett_network_new(.loss_func = SPINGALETT_LOSS_MSE);
          spingalett_layer(.net = net, .neurons_amount = 2);
          spingalett_layer(.net = net, .neurons_amount = 1, .act_func = SPINGALETT_ACT_SIGMOID);
          printf("%s %u\\n", spingalett_version(), spingalett_layer_count(net));
          spingalett_network_free(net);
          return 0;
      }
    C
    system ENV.cc, "-std=c11", "test.c", "-I#{include}", "-L#{lib}", "-lspingalett", "-o", "test"
    assert_match "#{version} 2", shell_output("./test")
  end
end
