import sys
import argparse
from pathlib import Path

def parse():
    argv = sys.argv[1:]
    if "--" in argv:
        pos = argv.index("--")
        python_args = argv[:pos]
        tci_args = argv[pos+1:]
    else:
        python_args = argv
        tci_args = []

    h5files_tci = {
        "ishigami" : "tci/Ishigami.h5",
        "legendre" : "tci/LegendrePoly3.h5",
        "morris" : "tci/Morris2006.h5",
    }


    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", type=str, default="ishigami",choices=["ishigami", "legendre", "morris"], help="Selects the test function")
    parser.add_argument("--tci", type=Path, help="Optional C++ TCI executable")
    parser.add_argument('--gp-seed', type=int, default=12572601, help='Random seed for generating the GP training points.')
    parser.add_argument('--gp-sizes', nargs='+', type=int, default=[128, 256, 512], help='Number of function evaluations used to training the GP model (powers of two).')
    parser.add_argument('--gp-sobol-samples', type=int, default=2**14, help='Number of sample pairs used to estimate Sobol indices from the GP (powers of two).')
    parser.add_argument('--gp-restarts', type=int, default=10, help='Number of additional attempts to optimize the GP kernel parameters.')
    parser.add_argument('--alpha', type=float, default=1e-8, help='Small value added to the GP covariance diagonal for numerical stability.')
    parser.add_argument("--tci-h5", type=str, default=None, help="TCI HDF5 file path (relative to this .py file) containing prediction points")


    if any(s == '--function' or s.startswith('--function=') for s in tci_args):
        raise RuntimeError('C++ "--function" is automatically copied from python args before "--" ')

    
    args = parser.parse_args(python_args)
    if tci_args and args.tci is None:
        parser.error(f"TCI arguments passed without accompanying --tci=... : {tci_args}")

    if args.alpha < 0 or args.gp_restarts < 0:
        parser.error("alpha and gp-restarts must be nonnegative")
    
    if args.tci_h5 is None:
        args.tci_h5 = h5files_tci[args.function]
    
    return args, python_args, tci_args


