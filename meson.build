project('glmnet', 'cpp',
  version : run_command('python', '-c', 'import setuptools_scm; print(setuptools_scm.get_version())', check : true).stdout().strip(),
  default_options : ['warning_level=3', 'cpp_std=c++17'])

py = import('python').find_installation(pure: false)

# Eigen is vendored in src/eigen
inc = include_directories('eigen', 'src/glmnetpp/include', 'src/glmnetpp/src', 'src/glmnetpp/test')

# Pybind11 dependency
pybind11_dep = dependency('pybind11')

# Source files
sources_elnet_point = [
  'src/elnet_point.cpp',
  'src/internal.cpp',
  'src/update_pb.cpp'
]
sources_lognet = [
  'src/lognet.cpp',
  'src/internal.cpp',
  'src/update_pb.cpp'
]
sources_fishnet = [
  'src/fishnet.cpp',
  'src/internal.cpp',
  'src/update_pb.cpp'
]
sources_gaussnet = [
  'src/gaussnet.cpp',
  'src/internal.cpp',
  'src/update_pb.cpp'
]
sources_coxnet = [
  'src/coxnet.cpp',
  'src/internal.cpp',
  'src/update_pb.cpp',
  'src/glmnetpp/coxdev/cpp/src/coxdev.cpp'
]

# Cox additionally needs the coxdev headers (nested submodule of glmnetpp)
inc_cox = include_directories('eigen', 'src/glmnetpp/include', 'src/glmnetpp/src', 'src/glmnetpp/test',
                              'src/glmnetpp/coxdev/cpp/include')

sources_multigaussnet = [
  'src/multigaussnet.cpp',
  'src/internal.cpp',
  'src/update_pb.cpp'
]

# Extension modules
py.extension_module(
  '_elnet_point',
  sources_elnet_point,
  include_directories : inc,
  dependencies : pybind11_dep,
  install : true,
  subdir : 'glmnet'
)

py.extension_module(
  '_lognet',
  sources_lognet,
  include_directories : inc,
  dependencies : pybind11_dep,
  install : true,
  subdir : 'glmnet'
)

py.extension_module(
  '_fishnet',
  sources_fishnet,
  include_directories : inc,
  dependencies : pybind11_dep,
  install : true,
  subdir : 'glmnet'
)

py.extension_module(
  '_gaussnet',
  sources_gaussnet,
  include_directories : inc,
  dependencies : pybind11_dep,
  install : true,
  subdir : 'glmnet'
)

py.extension_module(
  '_multigaussnet',
  sources_multigaussnet,
  include_directories : inc,
  dependencies : pybind11_dep,
  install : true,
  subdir : 'glmnet'
)

py.extension_module(
  '_coxnet',
  sources_coxnet,
  include_directories : inc_cox,
  dependencies : pybind11_dep,
  install : true,
  subdir : 'glmnet'
)

# Install Python sources
install_subdir(
    'glmnet',
    install_dir: py.get_install_dir(),
    exclude_directories: ['__pycache__', '.ipynb_checkpoints'],
    exclude_files: ['.gitignore']
)
