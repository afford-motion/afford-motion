import os, sys
sys.path.append(os.path.abspath('.'))
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='HumanML3D')
args = parser.parse_args()

if args.dataset == 'HumanML3D':
    from prepare.datasets.HumanML3D.HumanML3D import HumanML3D as Dataset
elif args.dataset == 'PROX':
    from prepare.datasets.PROX.PROX import PROX as Dataset
elif args.dataset == 'HUMANISE':
    from prepare.datasets.HUMANISE.HUMANISE import HUMANISE as Dataset
else:
    raise NotImplementedError

Dataset(args.data_dir).process()