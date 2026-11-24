# Runtime NVRTC compilation and the installed NVIDIA driver.
{ pkgs ? import (builtins.fetchTarball {
    name = "nixpkgs-5e2a59a5b1a82f89f2c7e598302a9cacebb72a67";
    url = "https://github.com/NixOS/nixpkgs/archive/5e2a59a5b1a82f89f2c7e598302a9cacebb72a67.tar.gz";
    sha256 = "02rf59xj2v1ml7v35x4j64ax1fb7xx96ld7va6rifyxazmwsr4rb";
  }) { config.allowUnfree = true; } }:

let
  nvrtc = pkgs.cudaPackages_12_9.cuda_nvrtc.lib;
  driverLib = "/run/opengl-driver/lib";
in pkgs.mkShell {
  packages = [ pkgs.gnumake pkgs.ripgrep ];
  shellHook = ''
    export LIBRARY_PATH="${nvrtc}/lib:${driverLib}''${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${nvrtc}/lib:${driverLib}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export NIX_LDFLAGS="$NIX_LDFLAGS -L${driverLib} -L${nvrtc}/lib -rpath ${driverLib} -rpath ${nvrtc}/lib"
  '';
}
