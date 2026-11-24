# abla-grad's dev shell: abla-graphics' (Vulkan loader, validation layers) plus the SPIR-V compilers the GLSL kernels need.
{ pkgs ? import (builtins.fetchTarball {
    name = "nixpkgs-5e2a59a5b1a82f89f2c7e598302a9cacebb72a67";
    url = "https://github.com/NixOS/nixpkgs/archive/5e2a59a5b1a82f89f2c7e598302a9cacebb72a67.tar.gz";
    sha256 = "02rf59xj2v1ml7v35x4j64ax1fb7xx96ld7va6rifyxazmwsr4rb";
  }) {} }:

pkgs.mkShell {
  packages = with pkgs; [
    gnumake
    glslang          # glslangValidator: GLSL -> SPIR-V at build time (tools/build-kernels.sh)
    spirv-tools      # spirv-val, spirv-dis
    vulkan-loader
    vulkan-tools
    vulkan-validation-layers
    ripgrep
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export NIX_LDFLAGS="$NIX_LDFLAGS -rpath ${pkgs.vulkan-loader}/lib"
  '';
}
