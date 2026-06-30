import os
import warnings

from setuptools import setup
import torch.utils.cpp_extension as torch_cpp_extension

if os.environ.get("CHAMFER3D_ALLOW_CUDA_MISMATCH") == "1":

    def _skip_cuda_version_check(compiler_name, compiler_version):
        warnings.warn(
            "Skipping PyTorch CUDA version check for chamfer3D. This is used "
            "when the PyTorch wheel reports a newer CUDA runtime but this "
            "extension is intentionally compiled with the CUDA 12.9 toolchain.",
            RuntimeWarning,
            stacklevel=2,
        )

    torch_cpp_extension._check_cuda_version = _skip_cuda_version_check

BuildExtension = torch_cpp_extension.BuildExtension
CUDAExtension = torch_cpp_extension.CUDAExtension

extra_compile_args = {
    "cxx": [
        "-DCCCL_IGNORE_DEPRECATED_CUDA_BELOW_12",
        "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
    ],
    "nvcc": [
        "-DCCCL_IGNORE_DEPRECATED_CUDA_BELOW_12",
        "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
    ],
}
setup(
    name="chamfer3D",
    packages=["chamfer3D"],
    package_dir={"chamfer3D": "."},
    ext_modules=[
        CUDAExtension(
            name="chamfer3D._C",
            sources=[
                "/".join(
                    __file__.split("/")[:-1] + ["chamfer3D_cuda.cpp"]
                ),  # must named as xxx_cuda.cpp
                "/".join(__file__.split("/")[:-1] + ["chamfer3D.cu"]),
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    version="1.0.6",
)
