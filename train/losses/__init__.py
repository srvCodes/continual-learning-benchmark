"""
Author: https://github.com/yulu0724/SDC-IL/blob/master/losses/__init__.py
"""


from .BinBranchLoss import BinBranchLoss
from .BinDevianceLoss import BinDevianceLoss
from .msloss import MultiSimilarityLoss
from .angular import AngularLoss

__factory = {
    'binbranch': BinBranchLoss,
    'bin': BinDevianceLoss,
    'msloss': MultiSimilarityLoss,
    'angular': AngularLoss
}

def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.
    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)
