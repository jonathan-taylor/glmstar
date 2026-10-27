project('glmnet', 'cpp',
  # the python of the build, as meson-python's native file gives it to find_program
  # (a bare 'python' is looked up on the PATH, which may have none, or another)
  version : run_command(find_program('python', 'python3'), '-c',
                        'import setuptools_scm; print(setuptools_scm.get_version())',
                        check : true).stdout().strip(),
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
  'src/update_pb.cpp'
]

# Cox additionally needs coxdev's headers: header-only since coxdev 0.4.0,
# installed with its Python package (coxdev.get_include(), a build
# requirement), and compiled into _coxnet
coxdev_incdir = run_command(py, ['-c', 'import coxdev; print(coxdev.get_include())'],
                            check : true).stdout().strip()
# (as a compile argument, not include_directories: the environment, and so
# coxdev's headers, may be in the source tree, e.g. uv's .venv, an absolute
# path there that include_directories refuses)
coxdev_dep = declare_dependency(compile_args : ['-I' + coxdev_incdir])
inc_cox = include_directories('eigen', 'src/glmnetpp/include', 'src/glmnetpp/src', 'src/glmnetpp/test')

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
  dependencies : [pybind11_dep, coxdev_dep],
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
