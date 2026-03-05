# License: MIT
# This file is partially built on SMAC3(https://github.com/automl/SMAC3), which is licensed as follows,

# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)
# Author: Aaron Klein, Marius Lindauer

import abc
import time
import warnings
from typing import Iterable, List, Union, Tuple, Optional,Any
import random
import scipy.optimize
import numpy as np

from openbox import logger
from openbox.acquisition_function.acquisition import AbstractAcquisitionFunction
from openbox.utils.config_space import get_one_exchange_neighbourhood, \
    Configuration, ConfigurationSpace
from openbox.utils.history import History, MultiStartHistory
from openbox.utils.util_funcs import get_types
from openbox.utils.constants import MAXINT
from ..compressor.sampling import SamplingStrategy, StandardSamplingStrategy
from . import generator
from . import base
from . import selector

class AcquisitionFunctionMaximizer(object, metaclass=abc.ABCMeta):
    """Abstract class for acquisition maximization.

    In order to use this class it has to be subclassed and the method
    ``_maximize`` must be implemented.

    Parameters
    ----------
    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy = None,
            rng: Union[bool, np.random.RandomState] = None,
            turbo_length=None,
    ):
        if sampling_strategy is None:
            sampling_strategy = StandardSamplingStrategy(config_space, seed=getattr(rng, "randint", lambda *_: None)(MAXINT))
        self.sampling_strategy = sampling_strategy
        self.config_space = config_space
        self.turbo_length=turbo_length
        
        self.turbo_state=False
        if self.turbo_length is not None:
            self.turbo_state=True
        
        if rng is None:
            logger.debug('no rng given, using default seed of 1')
            self.rng = np.random.RandomState(seed=1)
        else:
            self.rng = rng

    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Configuration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        **kwargs

        Returns
        -------
        iterable
            An iterable consisting of :class:`openbox.config_space.Configuration`.
        """
        return [t[1] for t in self._maximize(acquisition_function, history, num_points, **kwargs)]

    @abc.abstractmethod
    def _maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        """Implements acquisition function maximization.

        In contrast to ``maximize``, this method returns an iterable of tuples,
        consisting of the acquisition function value and the configuration. This
        allows to plug together different acquisition function maximizers.

        Parameters
        ----------
        acquisition_function: AbstractAcquisitionFunction
            acquisition function
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        **kwargs

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`openbox.config_space.Configuration`).
        """
        raise NotImplementedError()

    def _sort_configs_by_acq_value(
            self,
            acquisition_function,
            configs: List[Configuration]
    ) -> List[Tuple[float, Configuration]]:
        """Sort the given configurations by acquisition value

        Parameters
        ----------
        acquisition_function: AbstractAcquisitionFunction
            acquisition function
        configs : list(Configuration)

        Returns
        -------
        list: (acquisition value, Candidate solutions),
                ordered by their acquisition function value
        """

        acq_values = acquisition_function(configs)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]

    def fliter(
            self,
            incumbent_config,
            challengers):
        x_center = incumbent_config.get_array()
        lower_bounds = x_center - self.turbo_length / 2.0
        upper_bounds = x_center + self.turbo_length / 2.0
        filtered_challengers = []
        for config in challengers:
            config_array = config.get_array()
            if np.all(config_array >= lower_bounds) and np.all(config_array <= upper_bounds):
                filtered_challengers.append(config)
        return filtered_challengers


class CMAESMaximizer(AcquisitionFunctionMaximizer):
    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy = None,
            rng: Union[bool, np.random.RandomState] = None,
            rand_prob=0.25,
            turbo_length=None,
    ):
        super().__init__(config_space, sampling_strategy, rng, turbo_length=turbo_length)
        
        self.cmaes_generator=generator.CMAESGenerator(sampling_strategy=self.sampling_strategy,
                                                      config_space=self.config_space,
                                                      sigma=0.99)
        
    def _maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        challengers=self.maximize(acquisition_function=acquisition_function
                                  ,history=history
                                  ,num_points=num_points
                                  ,kwargs=kwargs)
        acq_config=[]
        for config in challengers:
            val=acquisition_function(config)
            acq_config.append((val,config))
        return acq_config

    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Configuration]:
        challengers=self.cmaes_generator.generate(num_points=num_points,
                                                  history=history,
                                                  rng=self.rng,
                                                  acq_function=acquisition_function)
        
        if self.turbo_state and history:
            incumbent_config = self.rng.choice(history.get_incumbent_configs())
            flitered_challengers=self.fliter(incumbent_config=incumbent_config,challengers=challengers)
            return flitered_challengers
        
        return challengers


class LocalSearchMaximizer(AcquisitionFunctionMaximizer):
    """Implementation of openbox's local search.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional

    max_steps: int
        Maximum number of iterations that the local search will perform

    n_steps_plateau_walk: int
        number of steps during a plateau walk before local search terminates

    """

    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy = None,
            rng: Union[bool, np.random.RandomState] = None,
            remove_duplicates: bool = True,
            max_steps: Optional[int] = None,
            n_steps_plateau_walk: int = 10,
            turbo_length=None,
    ):
        super().__init__(config_space, sampling_strategy, rng, turbo_length)
        
        self.local_generator=generator.LocalSearchGenerator(max_steps=max_steps,
                                                            n_steps_plateau_walk=n_steps_plateau_walk,
                                                            remove_duplicates=remove_duplicates,
                                                            config_space=config_space,
                                                            sampling_strategy=sampling_strategy)

    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> List[Configuration]:
        """Starts a local search from the given startpoint and quits
        if either the max number of steps is reached or no neighbor
        with an higher improvement was found.

        Parameters
        ----------
        acquisition_function: AbstractAcquisitionFunction
            acquisition function
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        **kwargs:
            Additional parameters that will be passed to the
            acquisition function

        Returns
        -------
        incumbent: np.array(1, D)
            The best found configuration
        acq_val_incumbent: np.array(1,1)
            The acquisition value of the incumbent

        """
        challengers = self.local_generator.generate(history=history,
                                                    num_points=num_points,
                                                    rng=self.rng,
                                                    acq_function=acquisition_function,
                                                    **kwargs)
        
        if self.turbo_state and history:
            incumbent_config = self.rng.choice(history.get_incumbent_configs())
            flitered_challengers=self.fliter(incumbent_config=incumbent_config,challengers=challengers)
            return flitered_challengers
            
        
        return challengers

    def _maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> List[Tuple[float,Configuration]]:
        challengers = self.maximize(acquisition_function=acquisition_function,
                                    history=history,
                                    num_points=num_points,
                                    **kwargs)
        acq_config=[]
        for config in challengers:
            val=acquisition_function(config)
            acq_config.append((val,config))
        return acq_config


class RandomSearchMaximizer(AcquisitionFunctionMaximizer):
    """Get candidate solutions via random sampling of configurations.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """
    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy = None,
            rng: Union[bool, np.random.RandomState] = None,
            random_state=None,
            batch_size=None,
            turbo_length=None,
    ):
        super().__init__(config_space, sampling_strategy, rng, turbo_length)
        
        self.random_state=random_state
        self.ramdom_generator=generator.RandomSearchGenerator(sampling_strategy=self.sampling_strategy,
                                                              batch_size=batch_size,
                                                              random_state=self.random_state,
                                                              config_space=self.config_space)
            
    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> List[Configuration]:
        """Randomly sampled configurations

        Parameters
        ----------
        acquisition_function: AbstractAcquisitionFunction
            acquisition function
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        _sorted: bool
            whether random configurations are sorted according to acquisition function
        **kwargs
            not used

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`openbox.config_space.Configuration`).
        """
        challengers = self.ramdom_generator.generate(num_points=num_points,
                                                     history=history,
                                                     rng=self.rng,
                                                     acq_function=acquisition_function,
                                                     **kwargs)
        
        if self.turbo_state and history:
            incumbent_config = self.rng.choice(history.get_incumbent_configs())
            flitered_challengers=self.fliter(incumbent_config=incumbent_config,challengers=challengers)
            return flitered_challengers
        
        return challengers
    
    def _maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            _sort=False,
            **kwargs
    ) -> List[Tuple[float,Configuration]]:
        """Randomly sampled configurations

        Parameters
        ----------
        acquisition_function: AbstractAcquisitionFunction
            acquisition function
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        _sorted: bool
            whether random configurations are sorted according to acquisition function
        **kwargs
            not used

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`openbox.config_space.Configuration`).
        """
        challengers = self.maximize(acquisition_function=acquisition_function,
                                    history=history,
                                    num_points=num_points,
                                    **kwargs)
        val_config=[]
        if _sort:
            for config in challengers:
                val_config=self._sort_configs_by_acq_value(acquisition_function=acquisition_function,configs=challengers)
        else:
            for config in challengers:
                val_config.append((0,config))
        return val_config


class ScipyMaximizer(AcquisitionFunctionMaximizer):
    """
    Wraps scipy optimizer. Only on continuous dims.

    Parameters
    ----------
    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy = None,
            rand_prob: float = 0.0,
            rng: Union[bool, np.random.RandomState] = None,
            method='local',
            turbo_length=None,
    ):
        super().__init__(config_space, sampling_strategy, rng, turbo_length)
        self.method=method
        self.scipy_generator=generator.ScipySearchGenerator(sampling_strategy=self.sampling_strategy,
                                                            method=self.method,
                                                            config_space=self.config_space)
    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            initial_configs=None,
            **kwargs
    ) -> List[Configuration]:
        
        challengers = self.scipy_generator.generate(history=history,
                                                    num_points=1,
                                                    rng=self.rng,
                                                    acq_function=acquisition_function,
                                                    initial_configs=initial_configs,
                                                    **kwargs)
        return challengers

    def _maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        challengers = self.maximize(acquisition_function=acquisition_function,
                                    history=history,
                                    num_points=num_points,
                                    **kwargs)
        acq_config=[]
        for config in challengers:
            val=acquisition_function(config)
            acq_config.append((val,config))
        return acq_config


class RandomScipyMaximizer(AcquisitionFunctionMaximizer):
    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy = None,
            rng: Union[bool, np.random.RandomState] = None,
            turbo_length=None,
    ):
        super().__init__(config_space, sampling_strategy, rng, turbo_length)
        self.random_generator = generator.RandomSearchGenerator(sampling_strategy=self.sampling_strategy,
                                                                config_space=config_space)
        self.scipy_generator = generator.ScipySearchGenerator(sampling_strategy=self.sampling_strategy,
                                                              config_space=config_space)
        self.strategy = [self.random_generator, self.scipy_generator]
        
        
    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> List[Configuration]:
        configs = []
        challengers = self.random_generator.generate(num_points=num_points,
                                                     history=history,
                                                     rng=self.rng,
                                                     acq_function=acquisition_function,
                                                     **kwargs)
        
        if self.turbo_state and history:
            incumbent_config = self.rng.choice(history.get_incumbent_configs())
            challengers=self.fliter(incumbent_config=incumbent_config,challengers=challengers)
        
        for config in challengers:
            updated_config = self.scipy_generator.generate(history=history,
                                                           num_points=1,
                                                           rng=self.rng,
                                                           acq_function=acquisition_function,
                                                           initial_configs=[config],
                                                           **kwargs)
            configs.extend(updated_config)
        return configs
    
    def _maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> List[Tuple[float,Configuration]]:
        challengers = self.maximize(acquisition_function=acquisition_function,
                                    history=history,
                                    num_points=num_points,
                                    **kwargs)
        acq_config = []
        for config in challengers:
            val=acquisition_function(config)
            acq_config.append((val,config))
        return acq_config