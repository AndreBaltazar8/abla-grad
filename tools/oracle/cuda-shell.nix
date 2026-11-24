let
  pkgs = import <nixpkgs> { config.allowUnfree = true; };
  cuda = pkgs.cudaPackages_12_9;
in
(pkgs.mkShell.override { stdenv = cuda.backendStdenv; }) {
  nativeBuildInputs = [ pkgs.cmake pkgs.ninja cuda.cuda_nvcc pkgs.autoAddDriverRunpath ];
  buildInputs = [ cuda.cuda_cccl cuda.cuda_cudart cuda.cuda_nvml_dev cuda.libcublas ];
}
