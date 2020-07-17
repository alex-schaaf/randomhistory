import numpy as np
import pynoddy.history
import scipy.stats
from typing import Iterable, List, Tuple, Dict, Union
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

    # def add_event(self, event_type: str, properties: dict) -> None:
    #     self.history.append((event_type, properties))
    #     # logging.info(f"Added '{event_type}' event to stochastic history.")
    #
    # def add_events(self, events: List[tuple]) -> None:
    #     for event_type, properties in events:
    #         self.add_event(event_type, properties)

    def sample_events(self, seed: int = None):
        """Generate a sample list of event properties."""
        sample_history = []
        for event in self.history:
            event_sample = sample_properties(event, seed=seed)
            sample_history.append(
                (event.get('type'), event_sample)
            )
        return sample_history

    def sample_history(self, random_seed: int = None):
        pass

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
    return (scipy.stats.uniform(extent[0], extent[1] - extent[0]),
            scipy.stats.uniform(extent[2], extent[3] - extent[2]),
            scipy.stats.uniform(extent[4] + z_offset, extent[5] - extent[4]))


def sample_properties(event: dict, seed: int = None):
    """Draw from parameter distribution dictionary and return parametrized
    one.

    Args:
        dist_dict: Dictionary of parameter distributions for stochastic
            event parameters.

    Returns:
        (dict) Sample from parameter distribution dictionary.
    """
    np.random.seed(seed)
    event_params = event.get('parameters')

    sample_dict = {}
    for property_name, value in event_params.items():
        # if value is a collection (e.g. x,y,z position, layer thicknesses)
        if type(value) in (list, tuple):
            samples = []
            for v in value:
                # if it's a scipy.stats distribution sample
                # else just add the the value itself
                if hasattr(v, "rvs"):
                    v = v.rvs()  # sample value
                samples.append(v)  # append sampled value
            sample_dict[property_name] = samples
        # if the value is a string (e.g. layer name property)
        elif type(value) in [str, int, float]:
            sample_dict[property_name] = value
        # if the value is a distribution -> sample
        elif hasattr(value, "rvs"):
            sample_dict[property_name] = value.rvs()
    return sample_dict
