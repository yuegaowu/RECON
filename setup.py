"""Install Python code; optionally build the upstream MTR CUDA extensions."""
import os
from setuptools import find_packages, setup

extensions = []
commands = {}
if os.environ.get('RECON_BUILD_CUDA') == '1':
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    commands['build_ext'] = BuildExtension
    for name, files in {
        'knn': ['knn.cpp', 'knn_gpu.cu', 'knn_api.cpp'],
        'attention': ['attention_api.cpp', 'attention_func_v2.cpp', 'attention_func.cpp',
                      'attention_value_computation_kernel_v2.cu',
                      'attention_value_computation_kernel.cu',
                      'attention_weight_computation_kernel_v2.cu',
                      'attention_weight_computation_kernel.cu'],
    }.items():
        module = 'unitraj.models.mtr.ops.' + name
        extensions.append(CUDAExtension(
            name=module + '.' + name + '_cuda',
            sources=[os.path.join(*module.split('.'), 'src', f) for f in files]))

setup(
    name='recon-unitraj', version='0.1.0',
    description='RECON and trajectory prediction baselines built on UniTraj',
    license='AGPL-3.0-or-later',
    packages=find_packages(include=['unitraj', 'unitraj.*']),
    package_data={'unitraj': ['configs/*.yaml', 'configs/method/*.yaml',
                             'configs/checkpoint_catalog.json']},
    cmdclass=commands, ext_modules=extensions,
)
