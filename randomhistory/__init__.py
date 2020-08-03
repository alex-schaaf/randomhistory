import numpy as np
from scipy.stats import distributions
import pynoddy.history
import scipy.stats
from typing import Iterable, List, Optional, Tuple, Dict, Union
import logging


class RandomHistory:
    def __init__(
        self, extent: Iterable[float], verbose: bool = False
    ) -> None:
        self.extent = extent,

        self._verbose = verbose
        if self._verbose:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

        self.history = []

    def sample_events(self, seed: int = None):
        """Generate a sample list of event properties."""
        sample_history = []
        for event in self.history:
            event_sample = sample_event_properties(event, seed=seed)
            sample_history.append(
                (event.get('type'), event_sample)
            )
        return sample_history

    def sample_history(self, random_seed: int = None):
        raise NotImplementedError


def random_positions(
    extent: Tuple[float],
    z_offset: float = 0
) -> tuple:
    """Random within-extent position generator.

    Args:
        extent (Tuple[float]): Model extent x,X,y,Y,z,Z
        z_offset (float, optional): Vertical offset from bottom (z).
            Defaults to 0.

    Returns:
        (tuple) of X,Y,Z uniform distributions.
    """
    return (
        scipy.stats.uniform(extent[0], extent[1] - extent[0]),
        scipy.stats.uniform(extent[2], extent[3] - extent[2]),
        scipy.stats.uniform(extent[4] + z_offset, extent[5] - extent[4])
    )


def _parse_distribution(parameter: dict) -> Optional[scipy.stats.skewnorm]:
    """Parse given parameter dictionary into scipy.stats distribution object.

    Args:
        parameter (dict): Parameter dictionary with 'distribution' k/v-pair
            providing the distribution type. Standard scipy.stats arguments
            are expected as keys for respective parametrization.

    Returns:
        Optional[scipy.stats.skewnorm]: Parametrized distribution.
    """
    distribution_type = parameter.get('distribution')
    if distribution_type == 'norm':
        # NORMAL DISTRIBUTION
        loc = parameter.get('value')
        scale = parameter.get('scale')
        skew = parameter.get('skew', 0)
        return scipy.stats.skewnorm(a=skew, loc=loc, scale=scale)
    else:
        print(f'Distribution type "{distribution_type}" not supported.')


def sample_event_properties(event: dict, seed: int = None) -> dict:
    # TODO: Stratigraphy event handling
    event_sample = {}
    parameters = event.get('parameters')
    if seed:
        np.random.seed(seed)

    for pname, p in parameters.items():
        if p.get('uncertain'):
            # is uncertain
            distribution = _parse_distribution(p)
            if not distribution:
                value = p.get('value')
            else:
                value = distribution.rvs()
        else:
            value = p.get('value')

        event_sample[pname] = value

    # pop and merge X,Y,Z into noddy pos parameter [X, Y, Z]
    keys = event_sample.keys()
    if 'X' in keys and 'Y' in keys and 'Z' in keys:
        pos = []
        for coord in ['X', 'Y', 'Z']:
            pos.append(event_sample.pop(coord))
        event_sample['pos'] = pos

    return event_sample
