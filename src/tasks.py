import os
import glob
import invoke
import logging

logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

os.chdir(os.path.dirname(__file__))


@invoke.task
def build_lib(context):
    infiles = ['combinatorics']
    LOG.info('Compiling lib..')
    invoke.run(
        ('g++ -c -O3 -Wall -Werror -std=c++17 -fPIC'
         ' `python3 -m pybind11 --includes`'
         ' -o ../libpb2q.o ')
        + ' '.join(f"{src}.cc" for src in infiles)
    )


@invoke.task(build_lib)
def build_phi4(context):
    LOG.info('Compiling phi4..')
    invoke.run(
        'g++ -O3 -Wall -Werror -shared -std=c++17 -fPIC'
        ' `python3 -m pybind11 --includes`'
        ' -I.'
        ' -L. -lpb2q'
        ' -o ../pb2q/phi4`python3-config --extension-suffix`'
        ' phi4.cc'
    )


@invoke.task
def clean(context):
    for file in glob.glob('*.o'):
        os.remove(file)
    for file in glob.glob('../pb2q/*.so'):
        os.remove(file)
