# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
#
# Spingalett for Nix: `nix build github:pka-human/Spingalett` (the library, its runtime, tools, headers,
# pkg-config and CMake files), `nix develop` for a shell to build it in.
{
  description = "Spingalett: neural networks in C, trained on CPUs and GPUs (CUDA, Vulkan), deployed down to INT2";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in {
      packages = forAll (pkgs: rec {
        spingalett = pkgs.stdenv.mkDerivation {
          pname = "spingalett";
          version = "1.2.0";
          src = self;
          nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.shaderc ];
          buildInputs = [ pkgs.vulkan-headers ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [ pkgs.llvmPackages.openmp ];
          # the baseline instruction set (later ones' kernels chosen at run time); the GPU backends load their
          # drivers when used (Vulkan's loader, CUDA's libcuda)
          cmakeFlags = [
            "-DSPINGALETT_NATIVE_ARCH=OFF"
            "-DBUILD_WITH_OPENMP=ON"
            "-DSPINGALETT_VULKAN=ON"
            "-DBUILD_APPS=OFF"
            "-DBUILD_TESTS=OFF"
          ];
          meta = with pkgs.lib; {
            description = "Neural networks in C: training on CPUs and GPUs, transformers, INT8 to INT2 deployment";
            homepage = "https://github.com/pka-human/Spingalett";
            license = licenses.mit;
            platforms = platforms.unix;
          };
        };
        default = spingalett;
      });

      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = [ pkgs.cmake pkgs.ninja pkgs.shaderc pkgs.vulkan-headers pkgs.clang pkgs.python3
                       pkgs.python3Packages.numpy ];
        };
      });
    };
}
