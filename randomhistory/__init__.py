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
        self.events = self.history
        self.rock_library = None
        self.rock_sample = []

    def sample_events(self, seed: int = None) -> List[Tuple[str, dict]]:
        self.rock_sample = []  # reset rock/lith sample
        sample_history = []
        np.random.seed(seed) if seed else None

        for i, event in enumerate(self.history):
            uncertain = event.get('uncertain')
            if uncertain:
                probability = event.get('probability')
                if not scipy.stats.bernoulli.rvs(probability):
                    continue  # skip if not happening

            event_family = event.get('event_family')
            if event_family:
                bounds = event.get('nEvents')
                n_events = int(scipy.stats.uniform(bounds[0], bounds[1] - bounds[0]).rvs())
                seeds = np.random.randint(0, 1000000, size=n_events)
            else:
                seeds = [seed]

            for s in seeds:
                event_sample = self.sample_event_properties(event, seed=s)
                sample_history.append(
                    (event.get('type'), event_sample)
                )

        return sample_history

    def sample_event_properties(self, event: dict, seed: int = None):
        """Sample properties of an event in the event history."""
        parameters = event.get('parameters')
        event_type = event.get('type')

        np.random.seed(seed) if seed else None

        event_sample = {}
        if event_type in ['stratigraphy', 'unconformity']:
            if parameters.get('num_layers').get('uncertain'):
                event_sample.update(
                    self.sample_stratigraphy(
                        event.get('parameters'),
                        seed=seed
                    ))
            else:
                for pname, param in parameters.items():
                    event_sample[pname] = param.get('value')

        for pname, param in parameters.items():
            if pname in ['num_layers', 'layer_thickness', 'layer_names', 'lithology']:
                continue  # skip stratigraphic parameters
            if param.get('uncertain'):
                # is uncertain
                distribution = _parse_distribution(param)
                if not distribution:
                    value = param.get('value')
                else:
                    value = distribution.rvs()
            else:
                value = param.get('value')
            event_sample[pname] = value

        # pop and merge X,Y,Z into noddy pos parameter [X, Y, Z]
        keys = event_sample.keys()
        if 'X' in keys and 'Y' in keys and 'Z' in keys:
            pos = []
            for coord in ['X', 'Y', 'Z']:
                pos.append(event_sample.pop(coord))
            event_sample['pos'] = pos

        return event_sample

    def sample_stratigraphy(self, parameters, seed: int = None):
        """Sample stratigraphy-related parameters for stratigraphy or unconformity
        events."""
        stratigraphy = {}
        np.random.seed(seed) if seed else None
        num_layers = int(_parse_distribution(parameters.get('num_layers')).rvs())  # sample number of layers
        layer_thickness = list(_parse_distribution(  # sample thickness for all layers
            parameters.get('layer_thickness')
        ).rvs(size=num_layers))
        layer_names = [f'Layer {layer + 1}' for layer in range(num_layers)]  # generate layer names
        stratigraphy.update({
            "num_layers": num_layers,
            "layer_thickness": layer_thickness,
            "layer_names": layer_names
        })
        if self.rock_library:
            lithologies = [rock.get('name') for rock in self.rock_library]
            lithology = list(np.random.choice(lithologies, size=num_layers, replace=True))
            self.rock_sample += lithology

        return stratigraphy

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
        # TODO: truncnorm to cut off negative values & out of bounds values for angles
    elif distribution_type == 'uniform':
        # UNIFORM DISTRIBUTION
        low = parameter.get('low')
        high = parameter.get('high')
        return scipy.stats.uniform(low, high - low)
    else:
        print(f'Distribution type "{distribution_type}" not supported.')
