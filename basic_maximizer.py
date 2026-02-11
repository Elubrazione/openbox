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
from my_openbox.compressor.sampling.base import SamplingStrategy
import generator

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
            sampling_strategy:SamplingStrategy,
            rng: Union[bool, np.random.RandomState] = None
    ):
        self.sampling_strategy=sampling_strategy
        self.config_space = config_space

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


class CMAESMaximizer(AcquisitionFunctionMaximizer):
    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy,
            rng: Union[bool, np.random.RandomState] = None,
            rand_prob=0.25,
    ):
        super().__init__(config_space,sampling_strategy, rng)
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
            sampling_strategy:SamplingStrategy,
            rng: Union[bool, np.random.RandomState] = None,
            remove_duplicates: bool = True,
            max_steps: Optional[int] = None,
            n_steps_plateau_walk: int = 10,
    ):
        super().__init__(config_space, sampling_strategy, rng)
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
        challengers=self.local_generator.generate(history=history,
                                                  num_points=num_points,
                                                  rng=self.rng,
                                                  acq_function=acquisition_function,
                                                  kwargs=kwargs)
        return challengers

    def _maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> List[Tuple[float,Configuration]]:
        challengers=self.maximize(acquisition_function=acquisition_function
                                  ,history=history
                                  ,num_points=num_points
                                  ,kwargs=kwargs)
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
            sampling_strategy:SamplingStrategy,
            rng: Union[bool, np.random.RandomState] = None
    ):
        super().__init__(config_space,sampling_strategy, rng)
        self.ramdom_generator=generator.RandomSearchGenerator(sampling_strategy=self.sampling_strategy,
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
        challengers=self.ramdom_generator.generate(num_points=num_points,
                                                   history=history,
                                                   rng=self.rng,
                                                   acq_function=acquisition_function,
                                                   kwargs=kwargs)
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
        challengers=self.maximize(acquisition_function=acquisition_function,
                                  history=history,
                                  num_points=num_points,
                                  kwargs=kwargs)
        val_config=[]
        if _sort:
            for config in challengers:
                val_config=self._sort_configs_by_acq_value(acquisition_function=acquisition_function,configs=challengers)
        else:
            for config in challengers:
                val_config.append((0,config))
        return val_config


class InterleavedLocalAndRandomSearchMaximizer(AcquisitionFunctionMaximizer):
    """Implements openbox's default acquisition function optimization.

    This acq_optimizer performs local search from the previous best points
    according, to the acquisition function, uses the acquisition function to
    sort randomly sampled configurations and interleaves unsorted, randomly
    sampled configurations in between.

    Parameters
    ----------
    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional

    max_steps: int
        [LocalSearchMaximizer] Maximum number of steps that the local search will perform

    n_steps_plateau_walk: int
        [LocalSearchMaximizer] number of steps during a plateau walk before local search terminates

    n_sls_iterations: int
        [LocalSearchMaximizer] number of local search iterations

    """

    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy,
            rng: Union[bool, np.random.RandomState] = None,
            max_steps: Optional[int] = None,
            n_steps_plateau_walk: int = 10,
            n_sls_iterations: int = 10,
            rand_prob=0.25
    ):
        super().__init__(config_space, sampling_strategy,rng)
        self.random_generator = generator.RandomSearchGenerator(
            config_space=config_space,
            sampling_strategy=sampling_strategy,
            rng=rng
        )
        self.local_generator = generator.LocalSearchGenerator(
            config_space=config_space,
            sampling_strategy=sampling_strategy,
            rng=rng,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk
        )
        self.n_sls_iterations = n_sls_iterations

        # =======================================================================
        # self.local_search = DiffOpt(
        #     acquisition_function=acquisition_function,
        #     config_space=config_space,
        #     rng=rng
        # )
        # =======================================================================

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
            passed to acquisition function

        Returns
        -------
        Iterable[Configuration]
            List of configurations.
        """

        next_configs_by_local_search = self.local_generator.generate(history=history,
                                                                     num_points=self.n_sls_iterations,
                                                                     rng=self.rng,
                                                                     acq_function=acquisition_function,
                                                                     kwargs=kwargs)

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = self.random_generator.generate(
            history=history,
            num_points=num_points-self.n_sls_iterations,
            rng=self.rng,
            acq_function=acquisition_function,
            kwargs=kwargs
        )

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of openbox. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs = (
                next_configs_by_random_search_sorted
                + next_configs_by_local_search
        )

        return next_configs

    def _maximize(
            self,
            history: History,
            acquisition_function:AbstractAcquisitionFunction,
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
            sampling_strategy:SamplingStrategy,
            rand_prob: float = 0.0,
            rng: Union[bool, np.random.RandomState] = None,
    ):
        super().__init__(config_space,sampling_strategy, rng)
        self.scipy_generator=generator.ScipySearchGenerator(sampling_strategy=self.sampling_strategy,
                                                            config_space=self.config_space)
    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            initial_config=None,
            **kwargs
    ) -> List[Configuration]:
        
        challengers=self.scipy_generator.generate(history=history,
                                                 num_points=1,
                                                 rng=self.rng,
                                                 acq_function=acquisition_function,
                                                 initial_config=initial_config,
                                                 kwargs=kwargs)
        
        return challengers

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


class RandomScipyMaximizer(AcquisitionFunctionMaximizer):
    """
    Use scipy.optimize with start points chosen by random search. Only on continuous dims.

    Parameters
    ----------
    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy,
            rand_prob: float = 0.0,
            rng: Union[bool, np.random.RandomState] = None,
    ):
        super().__init__(config_space, sampling_strategy,rng)
        
        self.random_scipy_generator=generator.RandomScipySearchGenerator(sampling_strategy=sampling_strategy,
                                                                         config_space=config_space)

    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            num_trials=10,
            **kwargs
    ) -> List[Configuration]:
        
        challengers=self.random_scipy_generator.generate(num_points=num_points,
                                                         history=history,
                                                         rng=self.rng,
                                                         acq_function=acquisition_function,
                                                         num_trials=num_trials,
                                                         kwargs=kwargs)
        return challengers
        
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


class ScipyGlobalMaximizer(AcquisitionFunctionMaximizer):
    """
    Wraps scipy global optimizer. Only on continuous dims.

    Parameters
    ----------
    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy,
            rand_prob: float = 0.0,
            rng: Union[bool, np.random.RandomState] = None,
    ):
        super().__init__(config_space,sampling_strategy, rng)
        self.scipy_global_generator=generator.ScipyGlobalGenerator(sampling_strategy=sampling_strategy,
                                                                   config_space=config_space)

    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            initial_config=None,
            **kwargs
    ) -> List[Configuration]:
        challengers=self.scipy_global_generator.generate(num_points=1,
                                                         history=history,
                                                         rng=self.rng,
                                                         acq_function=acquisition_function,
                                                         kwargs=kwargs)
        return challengers

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


class MESMO_Maximizer(AcquisitionFunctionMaximizer):
    """Implements Scipy optimizer for MESMO. Only on continuous dims

    Parameters
    ----------
    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional

    """

    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy,
            rng: Union[bool, np.random.RandomState] = None,
            num_mc=1000,
            num_opt=1000,
            rand_prob=0.0
    ):
        super().__init__(config_space, sampling_strategy,rng)
        self.num_mc = num_mc
        self.num_opt = num_opt
        self.mesmo_generator=generator.MESMO_Generator(sampling_strategy=sampling_strategy,
                                                       config_space=config_space,)

    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,  # todo useless
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
            passed to acquisition function

        Returns
        -------
        Iterable[Configuration]
            List of configurations.
        """
        challengers=self.mesmo_generator.generate(num_points=num_points,
                                                  history=history,
                                                  rng=self.rng,
                                                  acq_function=acquisition_function,
                                                  kwargs=kwargs)
        return challengers

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


class batchMCMaximizer(AcquisitionFunctionMaximizer):
    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy,
            rng: Union[bool, np.random.RandomState] = None,
            batch_size=None,
            rand_prob=0.0
    ):
        super().__init__(config_space, sampling_strategy,rng)
        self.batch_generator=generator.BatchMCGenerator(sampling_strategy=sampling_strategy,config_space=config_space,batch_size=batch_size)

    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: Union[History, MultiStartHistory],
            num_points: int,
            _sorted: bool = True,
            **kwargs
    ) -> List[Configuration]:
        """Randomly sampled configurations

        Parameters
        ----------
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        _sorted: bool
            whether random configurations are sorted according to acquisition function
        **kwargs
            turbo_state: TurboState
                provide turbo state to use trust region

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`openbox.config_space.Configuration`).
        """
        challengers=self.batch_generator.generate(num_points=num_points,
                                                  history=history,
                                                  rng=self.rng,
                                                  acq_function=acquisition_function,
                                                  kwargs=kwargs)

        return challengers

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


class InitialMaximizer(AcquisitionFunctionMaximizer):
    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy,
            random_state='high',
            batch_size=None,
            rng: Union[bool, np.random.RandomState] = None
    ):
        super().__init__(config_space,sampling_strategy, rng)
        if random_state =='high':
            self.initial_generator=generator.RandomSearchGenerator(sampling_strategy=sampling_strategy,
                                                                   config_space=config_space)
        elif random_state == 'medium':
            self.initial_generator=generator.BatchMCGenerator(sampling_strategy=sampling_strategy,
                                                              config_space=config_space,
                                                              batch_size=batch_size)
        elif random_state =='low':
            self.initial_generator=generator.MESMO_Generator(sampling_strategy=sampling_strategy,
                                                             config_space=config_space)
            
    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> List[Configuration]:
        challengers=self.initial_generator.generate(num_points=num_points,
                                                   history=history,
                                                   rng=self.rng,
                                                   acq_function=acquisition_function,
                                                   kwargs=kwargs)
        return challengers
    
    def _maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            batch_size=None,
            random_state='high',
            _sort=False,
            **kwargs
    ) -> List[Tuple[float,Configuration]]:
        challengers=self.maximize(acquisition_function=acquisition_function
                                  ,history=history
                                  ,num_points=num_points
                                  ,kwargs=kwargs)
        acq_config=[]
        for config in challengers:
            val=acquisition_function(config)
            acq_config.append((val,config))
        return acq_config

    
class RandomScipyMaximizer(AcquisitionFunctionMaximizer):
    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy,
            rng: Union[bool, np.random.RandomState] = None
    ):
        super().__init__(config_space,sampling_strategy, rng)
        self.random_generator=generator.RandomSearchGenerator(sampling_strategy=sampling_strategy,
                                                                   config_space=config_space)
        self.scipy_generator=generator.ScipySearchGenerator(sampling_strategy=sampling_strategy,
                                                            config_space=config_space)
            
    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> List[Configuration]:
        configs=[]
        challengers=self.random_generator.generate(num_points=num_points,
                                                   history=history,
                                                   rng=self.rng,
                                                   acq_function=acquisition_function,
                                                   kwargs=kwargs)
        for config in challengers:
            updated_config=self.scipy_generator.generate(history=history,
                                                         num_points=1,
                                                         rng=self.rng,
                                                         acq_function=acquisition_function,
                                                         initial_config=config,
                                                         kwargs=kwargs)
            configs.extend(updated_config)
        return configs
    
    def _maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> List[Tuple[float,Configuration]]:
        challengers=self.maximize(acquisition_function=acquisition_function
                                  ,history=history
                                  ,num_points=num_points
                                  ,kwargs=kwargs)
        acq_config=[]
        for config in challengers:
            val=acquisition_function(config)
            acq_config.append((val,config))
        return acq_config