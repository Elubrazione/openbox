# License: MIT
# Author: Huaijun Jiang
# Date: 2022-12-13

import os
import copy
import json
from datetime import datetime
from functools import partial
from typing import List, Tuple, Union, Optional
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, CategoricalHyperparameter, OrdinalHyperparameter
from openbox import logger
from openbox.utils.constants import SUCCESS
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.config_space.space_utils import get_config_from_dict, get_config_values, get_config_numerical_values
from openbox.utils.transform import get_transform_function
from openbox.utils.multi_objective import get_pareto_front, Hypervolume
from openbox.utils.util_funcs import transform_to_1d_list, deprecate_kwarg


class Observation(object):
    @deprecate_kwarg('objs', 'objectives', 'a future version')
    def __init__(
            self,
            config: Configuration,
            objectives: Union[List[float], np.ndarray],
            constraints: Optional[Union[List[float], np.ndarray]] = None,
            trial_state: Optional['State'] = SUCCESS,
            elapsed_time: Optional[float] = None,
            extra_info: Optional[dict] = None,
    ):
        self.config = config
        self.objectives = objectives
        self.constraints = constraints
        self.trial_state = trial_state
        self.elapsed_time = elapsed_time
        self.create_time = datetime.now()
        if extra_info is None:
            extra_info = dict()
        assert isinstance(extra_info, dict)
        self.extra_info = extra_info

        self.objectives = transform_to_1d_list(self.objectives, hint='objectives')
        if self.constraints is not None:
            self.constraints = transform_to_1d_list(self.constraints, hint='constraints')

    def __str__(self):
        items = [f'config={self.config}', f'objectives={self.objectives}']
        if self.constraints is not None:
            items.append(f'constraints={self.constraints}')
        items.append(f'trial_state={self.trial_state}')
        if self.elapsed_time is not None:
            items.append(f'elapsed_time={self.elapsed_time}')
        items.append(f'create_time={self.create_time}')
        if self.extra_info:
            items.append(f'extra_info={self.extra_info}')
        return f'Observation({", ".join(items)})'

    __repr__ = __str__

    def to_dict(self, compact: bool = True) -> dict:
        """Serialize observation to dict.

        Parameters
        ----------
        compact : bool, default=True
            If True, omit fields that are None / empty / default.
            If False, include all fields (legacy behavior for ``include_optional=True``).
        """
        data = {
            'config': self.config.get_dictionary(),
            'objectives': self.objectives,
        }

        # trial_state: only store if not SUCCESS (SUCCESS is the common case)
        if not compact or self.trial_state != SUCCESS:
            data['trial_state'] = self.trial_state

        # Optional fields — only stored when they carry information
        if not compact or self.constraints is not None:
            data['constraints'] = self.constraints
        if not compact or self.elapsed_time is not None:
            data['elapsed_time'] = self.elapsed_time
        if not compact or self.create_time is not None:
            data['create_time'] = self.create_time.isoformat()
        if not compact or self.extra_info:
            data['extra_info'] = self.extra_info

        return copy.deepcopy(data)

    @classmethod
    def from_dict(cls, data: dict, config_space: ConfigurationSpace):
        config = data['config']
        if isinstance(config, dict):
            assert config_space is not None, 'config_space must be provided if config is a dict'
            data['config'] = get_config_from_dict(config_space, config)
        else:
            assert isinstance(config, Configuration), 'config must be a dict or Configuration'

        create_time = data.pop('create_time', None)

        # Default trial_state to SUCCESS if omitted (compact format)
        data.setdefault('trial_state', SUCCESS)

        observation = cls(**data)

        if isinstance(create_time, str):
            observation.create_time = datetime.fromisoformat(create_time)
        elif create_time is not None:
            logger.warning(f'Unable to parse create_time ({create_time}) from dict.')
        return observation

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False
        return self.to_dict(compact=False) == other.to_dict(compact=False)


class History(object):
    """
    A history object stores the observations of the optimization process.

    Parameters
    ----------
    task_id: str
        Task id.
    num_objectives: int, default=1
        Number of objectives.
    num_constraints: int, default=0
        Number of constraints.
    config_space: ConfigurationSpace, optional
        Configuration space.
    ref_point: list or np.ndarray, optional
        Reference point for multi-objective hypervolume calculation.
    meta_info: dict, optional
        Meta information. Structured as::

            {
                "meta_feature": [...],         # workload feature vector
                "space": {
                    "original": {...},          # original config space definition
                    "dimension": {...},         # dimension info per param
                    "range": {...},             # range info per param
                },
                "compressor": {
                    "strategy": "none",
                    "original_params": 51,
                    "compressed_params": 51,
                    "computed_params": 0,
                    "range_compression_details": {},
                },
                "warm_start": {
                    "strategy": "none",
                    "source_tasks": [],
                    "init_configs": [],
                },
                "transfer": {
                    "strategy": "none",
                    "source_task_ids": [],
                },
                "random": {...},               # random state / seed info
            }
    """
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            task_id: str = 'OpenBox',
            num_objectives: int = 1,
            num_constraints: int = 0,
            config_space: Optional[ConfigurationSpace] = None,
            ref_point: Optional[Union[List[float], np.ndarray]] = None,
            meta_info: Optional[dict] = None,
    ):
        self.task_id = task_id
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.config_space = config_space
        if meta_info is None:
            meta_info = dict()
        assert isinstance(meta_info, dict)
        self.meta_info = meta_info

        self.observations = []
        self.global_start_time = datetime.now()

        # multi-objective
        self._ref_point = None
        self.ref_point = ref_point  # type: Optional[List[float]]

    # ========================================================================
    # Meta-info helpers — structured setters for the meta_info dict
    # ========================================================================

    def set_meta_feature(self, feature: Union[List[float], np.ndarray]) -> None:
        """Set workload meta-feature vector."""
        if isinstance(feature, np.ndarray):
            feature = feature.tolist()
        self.meta_info['meta_feature'] = feature

    def set_space_info(self,
                       original: Optional[dict] = None,
                       dimension: Optional[dict] = None,
                       range_info: Optional[dict] = None) -> None:
        """Set configuration space metadata.

        Parameters
        ----------
        original : dict, optional
            Full original config space definition.
        dimension : dict, optional
            Per-parameter dimension info (e.g. type, num_values).
        range_info : dict, optional
            Per-parameter range info (e.g. lower, upper, log).
        """
        space = {}
        if original is not None:
            space['original'] = original
        if dimension is not None:
            space['dimension'] = dimension
        if range_info is not None:
            space['range'] = range_info
        if space:
            self.meta_info['space'] = space

    def set_compressor_info(self,
                            strategy: str = 'none',
                            original_params: int = 0,
                            compressed_params: int = 0,
                            computed_params: int = 0,
                            range_compression_details: Optional[dict] = None) -> None:
        """Set compression metadata."""
        info = {
            'strategy': strategy,
            'original_params': original_params,
            'compressed_params': compressed_params,
            'computed_params': computed_params,
        }
        if range_compression_details:
            info['range_compression_details'] = range_compression_details
        self.meta_info['compressor'] = info

    def set_warm_start_info(self,
                            strategy: str = 'none',
                            source_tasks: Optional[List[str]] = None,
                            init_configs: Optional[List[dict]] = None) -> None:
        """Set warm-start metadata."""
        info = {'strategy': strategy}
        if source_tasks:
            info['source_tasks'] = source_tasks
        if init_configs:
            info['init_configs'] = init_configs
        self.meta_info['warm_start'] = info

    def set_transfer_info(self,
                          strategy: str = 'none',
                          source_task_ids: Optional[List[str]] = None) -> None:
        """Set transfer learning metadata."""
        info = {'strategy': strategy}
        if source_task_ids:
            info['source_task_ids'] = source_task_ids
        self.meta_info['transfer'] = info

    # ========================================================================
    # Core properties & methods (unchanged)
    # ========================================================================

    def __len__(self):
        return len(self.observations)

    def empty(self):
        return len(self) == 0

    @property
    def configurations(self) -> List[Configuration]:
        return [obs.config for obs in self.observations]

    @property
    def objectives(self) -> List[List[float]]:
        return [obs.objectives for obs in self.observations]

    @property
    def constraints(self) -> List[Optional[List[float]]]:
        return [obs.constraints for obs in self.observations]

    # alias
    configs = configurations
    objs = objectives
    constrs = cons = constraints

    @property
    def trial_states(self) -> List['State']:
        return [obs.trial_state for obs in self.observations]

    @property
    def elapsed_times(self) -> List[Optional[float]]:
        return [obs.elapsed_time for obs in self.observations]

    @property
    def create_times(self) -> List[datetime]:
        return [obs.create_time for obs in self.observations]

    @property
    def extra_infos(self) -> List[dict]:
        return [obs.extra_info for obs in self.observations]

    @property
    def ref_point(self) -> Optional[List[float]]:
        return self._ref_point

    @ref_point.setter
    def ref_point(self, ref_point: Optional[Union[List[float], np.ndarray]]):
        if ref_point is not None:
            assert self.num_objectives > 1, 'Reference point is only used for multi-objective optimization!'
        self._ref_point = self.check_ref_point(ref_point)  # type: Optional[List[float]]

    def check_ref_point(self, ref_point: Optional[Union[List[float], np.ndarray]]) -> Optional[List[float]]:
        """check and standardize the reference point"""
        if ref_point is not None:
            assert self.num_objectives > 1, 'Reference point is only used for multi-objective optimization!'
            ref_point = transform_to_1d_list(ref_point, hint='ref_point')
            assert len(ref_point) == self.num_objectives, 'Length of ref_point must be equal to num_objectives'
        return ref_point

    @staticmethod
    def _has_invalid_value(x: np.ndarray) -> bool:
        """Check if x has invalid value (nan, inf, -inf)."""
        x = np.asarray(x, dtype=np.float64)
        return np.any(np.isnan(x)) or np.any(np.isinf(x))

    def is_valid_observation(self, obs: Observation, raise_error=True):
        """Check if the observation is valid. If raise_error=True, raise ValueError if invalid."""
        try:
            if not isinstance(obs, Observation):
                raise ValueError(f'observation must be an instance of Observation, got {type(obs)}')
            if not isinstance(obs.config, Configuration):
                raise ValueError(f'config must be an instance of Configuration, got {type(obs.config)}')
            if len(obs.objectives) != self.num_objectives:
                raise ValueError(f'num objectives must be {self.num_objectives}, got {len(obs.objectives)}')
            if obs.trial_state == SUCCESS and self._has_invalid_value(obs.objectives):
                raise ValueError(f'invalid values (inf, nan) are not allowed in objectives in a SUCCESS trial, '
                                 f'got {obs.objectives}')
            if self.num_constraints > 0 and obs.trial_state == SUCCESS:
                if obs.constraints is None:
                    raise ValueError(f'constraints is None in a SUCCESS trial!')
                if self._has_invalid_value(obs.constraints):
                    raise ValueError(f'invalid values (inf, nan) are not allowed in constraints in a SUCCESS trial, '
                                     f'got {obs.constraints}')
            if obs.constraints is not None and len(obs.constraints) != self.num_constraints:
                raise ValueError(f'num constraints must be {self.num_constraints}, got {len(obs.constraints)}')
            if not isinstance(obs.extra_info, dict):
                raise ValueError(f'extra_info must be a dict, got {type(obs.extra_info)}')
        except Exception:
            logger.exception(f'Invalid observation: {obs}')
            if raise_error:
                raise
            return False
        return True

    def update_observation(self, observation: Observation) -> None:
        """Update the observation to the history."""
        self.is_valid_observation(observation, raise_error=True)
        if observation.config in self.configurations:
            logger.warning('Duplicate configuration detected!')
        self.observations.append(observation)
        logger.debug(f'Observation updated in history: {observation}')

    def update_observations(self, observations: List[Observation]) -> None:
        """Update a list of observations to the history."""
        for observation in observations:
            self.is_valid_observation(observation, raise_error=True)
        for observation in observations:
            self.update_observation(observation)
        logger.info(f'{len(observations)} observations updated in history.')

    def get_config_space(self) -> Optional[ConfigurationSpace]:
        if self.config_space is not None:
            return self.config_space
        elif len(self) > 0:
            config_space = self.configurations[0].configuration_space
            return config_space
        else:
            logger.warning('Failed to get config_space because it is not set in History '
                           'and no observation is recorded. Return None.')
            return None

    def get_config_array(self, transform: str = 'scale') -> np.ndarray:
        if transform == 'scale':
            return convert_configurations_to_array(self.configurations)
        elif transform == 'numerical':
            return np.array([get_config_numerical_values(config) for config in self.configurations])
        else:
            raise ValueError(f'Unknown transform method: {transform}')

    def get_config_dicts(self) -> List[dict]:
        return [copy.deepcopy(config.get_dictionary()) for config in self.configurations]

    @staticmethod
    def _get_min_max_values(X: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        X[np.isinf(X)] = np.nan
        min_X = np.nanmin(X, axis=axis)
        max_X = np.nanmax(X, axis=axis)
        min_X[np.isnan(min_X)] = 0.0
        max_X[np.isnan(max_X)] = 0.0
        return min_X, max_X

    def _get_transformed_values(self, attr: str, transform: str, warn_invalid_value: bool = True) -> np.ndarray:
        if attr == 'objectives':
            values = self.objectives
        elif attr == 'constraints':
            values = self.constraints
        else:
            raise ValueError(f'Unknown attribute: {attr}. Must be "objectives" or "constraints".')

        assert isinstance(transform, str)
        transform = transform.lower()

        values = np.asarray(values, dtype=np.float64)
        if transform == 'none':
            if warn_invalid_value and self._has_invalid_value(values):
                logger.warning(f'{attr} contains invalid values (nan or inf) and is returned as is.')
            return values

        transform = set(map(str.strip, transform.split(',')))
        if '' in transform:
            transform.remove('')

        if 'failed' in transform:
            transform.remove('failed')
        success_mask = self.get_success_mask()
        values[~success_mask] = np.full(values.shape[1], np.nan)
        min_values, max_values = self._get_min_max_values(values, axis=0)
        values[~success_mask] = max_values

        if 'infeasible' in transform:
            transform.remove('infeasible')
            if attr == 'constraints':
                raise ValueError('Cannot use "infeasible" transform for constraints!')
            feasible_mask = self.get_feasible_mask()
            values[~feasible_mask] = max_values

        for tf in transform:
            values = get_transform_function(tf)(values)

        values = np.asarray(values, dtype=np.float64)
        return values

    def get_objectives(self, transform: str = 'infeasible', warn_invalid_value: bool = True) -> np.ndarray:
        objectives = self._get_transformed_values(
            attr='objectives', transform=transform, warn_invalid_value=warn_invalid_value)
        return objectives

    def get_constraints(self, transform: str = 'bilog', warn_invalid_value: bool = True) -> Optional[np.ndarray]:
        if self.num_constraints == 0:
            return None
        constraints = self._get_transformed_values(
            attr='constraints', transform=transform, warn_invalid_value=warn_invalid_value)
        return constraints

    def get_success_mask(self) -> np.ndarray:
        success_mask = np.asarray([trial_state == SUCCESS for trial_state in self.trial_states], dtype=bool)
        return success_mask

    def get_success_count(self) -> int:
        cnt = np.sum(self.get_success_mask())
        return int(cnt.item())

    def get_feasible_mask(self, exclude_failed: bool = True) -> np.ndarray:
        if self.num_constraints == 0:
            feasible_mask = np.ones(len(self), dtype=bool)
        else:
            constraints = self.get_constraints(transform='none', warn_invalid_value=False)
            feasible_mask = np.all(constraints <= 0, axis=-1)
        if exclude_failed:
            feasible_mask &= self.get_success_mask()
        return feasible_mask

    def get_feasible_count(self, exclude_failed: bool = True) -> int:
        cnt = np.sum(self.get_feasible_mask(exclude_failed=exclude_failed))
        return int(cnt.item())

    def get_incumbents(self) -> List[Observation]:
        if self.num_objectives > 1:
            raise ValueError('get_incumbents() is used for single-objective optimization! Use get_pareto() instead.')

        feasible_mask = self.get_feasible_mask(exclude_failed=True)
        if not np.any(feasible_mask):
            logger.warning('No feasible incumbent observations returned!')
            return []

        objectives = self.get_objectives(transform='none', warn_invalid_value=False)
        incumbent_value = np.min(objectives[feasible_mask])
        incumbent_mask = (objectives == incumbent_value).reshape(-1) & feasible_mask

        incumbents = [self.observations[i] for i in np.where(incumbent_mask)[0]]
        return incumbents

    def get_incumbent_value(self) -> float:
        if self.num_objectives > 1:
            raise ValueError('get_incumbent_value() is used for single-objective optimization! '
                             'Use get_pareto_front() instead.')

        feasible_mask = self.get_feasible_mask(exclude_failed=True)
        if not np.any(feasible_mask):
            logger.warning('No feasible observations! Return np.inf as incumbent value.')
            return np.inf

        objectives = self.get_objectives(transform='none', warn_invalid_value=False)
        incumbent_value = np.min(objectives[feasible_mask])
        return incumbent_value

    def get_incumbent_configs(self) -> List[Configuration]:
        if self.num_objectives > 1:
            raise ValueError('get_incumbent_configs() is used for single-objective optimization! '
                             'Use get_pareto_set() instead.')
        incumbents = self.get_incumbents()
        incumbent_configs = [obs.config for obs in incumbents]
        return incumbent_configs

    def get_mo_incumbent_values(self) -> np.ndarray:
        assert self.num_objectives > 1

        feasible_mask = self.get_feasible_mask(exclude_failed=True)
        if not np.any(feasible_mask):
            logger.warning('No feasible observations! Return np.inf(s) as incumbent values.')
            return np.full(self.num_objectives, np.inf)

        objectives = self.get_objectives(transform='none', warn_invalid_value=False)
        mo_incumbent_values = np.min(objectives[feasible_mask], axis=0)
        return mo_incumbent_values

    def get_pareto(self) -> List[Observation]:
        assert self.num_objectives > 1

        feasible_mask = self.get_feasible_mask(exclude_failed=True)
        if not np.any(feasible_mask):
            logger.warning('No feasible observations! Return empty pareto.')
            return []

        objectives = self.get_objectives(transform='none', warn_invalid_value=False)
        pareto_idx = get_pareto_front(objectives[feasible_mask], return_index=True)
        pareto = [self.observations[i] for i in np.where(feasible_mask)[0][pareto_idx]]
        return pareto

    def get_pareto_front(self, lexsort: bool = True) -> np.ndarray:
        assert self.num_objectives > 1

        feasible_mask = self.get_feasible_mask(exclude_failed=True)
        if not np.any(feasible_mask):
            logger.warning('No feasible observations! Return empty pareto front.')
            return np.empty((0, self.num_objectives), dtype=np.float64)

        objectives = self.get_objectives(transform='none', warn_invalid_value=False)
        pareto_front = get_pareto_front(objectives[feasible_mask], lexsort=lexsort)
        return pareto_front

    def get_pareto_set(self) -> List[Configuration]:
        assert self.num_objectives > 1

        pareto = self.get_pareto()
        pareto_set = [obs.config for obs in pareto]
        return pareto_set

    def compute_hypervolume(
            self,
            ref_point: Optional[List[float]] = None,
            data_range: str = 'last',
    ) -> Union[float, List[float]]:
        assert self.num_objectives > 1
        ref_point = self.check_ref_point(ref_point)
        ref_point = ref_point if ref_point is not None else self.ref_point
        assert ref_point is not None, 'ref_point must be provided!'

        if data_range == 'last':
            pareto_front = self.get_pareto_front(lexsort=False)
            hv = Hypervolume(ref_point=ref_point).compute(pareto_front)
            return hv
        elif data_range == 'all':
            logger.info('Computing all hypervolumes...')
            feasible_mask = self.get_feasible_mask(exclude_failed=True)
            objectives = self.get_objectives(transform='none', warn_invalid_value=False)
            HV = Hypervolume(ref_point=ref_point)
            hv_list = []
            for i in range(len(self)):
                mask = feasible_mask[:i + 1]
                objs = objectives[:i + 1]
                pareto_front = get_pareto_front(objs[mask], lexsort=False)
                hv = HV.compute(pareto_front)
                hv_list.append(hv)
            if len(self) == 0:
                logger.warning('No observations! Return empty hypervolume list.')
            return hv_list
        else:
            raise ValueError(f'Invalid data_range: {data_range}')

    # ========================================================================
    # JSON Serialization — compact by default
    # ========================================================================

    def save_json(self, filename: str, compact: bool = True):
        """Save history to JSON file.

        Parameters
        ----------
        filename : str
            Output file path.
        compact : bool, default=True
            If True (default), omit fields that are None, empty, or take
            their default values.  The resulting file is smaller and easier
            to read.  ``load_json`` handles both compact and full formats
            transparently.

            Specifically when ``compact=True``:
            - ``num_objectives`` is omitted when it equals 1 (default).
            - ``num_constraints`` is omitted when it equals 0 (default).
            - ``ref_point`` is omitted when it is None.
            - ``meta_info`` is omitted when it is empty ``{}``.
            - ``global_start_time`` is omitted.
            - Each observation omits ``trial_state`` when SUCCESS,
              ``constraints``/``elapsed_time``/``create_time``/``extra_info``
              when None or empty.
        """
        dirname = os.path.dirname(filename)
        if dirname != '' and not os.path.exists(dirname):
            logger.info(f'Creating directory to save history: {dirname}')
            os.makedirs(dirname, exist_ok=True)

        # ----- required fields -----
        data = {
            'task_id': self.task_id,
        }

        # ----- conditionally included fields -----
        if not compact or self.num_objectives != 1:
            data['num_objectives'] = self.num_objectives

        if not compact or self.num_constraints != 0:
            data['num_constraints'] = self.num_constraints

        if not compact or self.ref_point is not None:
            data['ref_point'] = self.ref_point

        if not compact or self.meta_info:
            data['meta_info'] = self._serialize_meta_info()

        if not compact:
            data['global_start_time'] = self.global_start_time.isoformat()

        # ----- observations -----
        data['observations'] = [
            obs.to_dict(compact=compact) for obs in self.observations
        ]

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f'Saved history (len={len(self)}) to {filename}')

    def _serialize_meta_info(self) -> dict:
        """Serialize meta_info, converting numpy arrays to lists."""
        result = {}
        for key, value in self.meta_info.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, dict):
                result[key] = self._serialize_dict_recursive(value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _serialize_dict_recursive(d: dict) -> dict:
        result = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, dict):
                result[k] = History._serialize_dict_recursive(v)
            elif isinstance(v, (np.integer,)):
                result[k] = int(v)
            elif isinstance(v, (np.floating,)):
                result[k] = float(v)
            else:
                result[k] = v
        return result

    @classmethod
    def load_json(cls, filename: str, config_space: ConfigurationSpace) -> 'History':
        """Load history from JSON file.

        Handles both compact and full (legacy) formats transparently.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f'File not found: {filename}')
        with open(filename, 'r') as f:
            data = json.load(f)

        global_start_time = data.pop('global_start_time', None)
        if global_start_time is not None:
            global_start_time = datetime.fromisoformat(global_start_time)

        observations_data = data.pop('observations', [])
        observations = [Observation.from_dict(obs, config_space) for obs in observations_data]

        # Apply defaults for compact format
        data.setdefault('num_objectives', 1)
        data.setdefault('num_constraints', 0)
        data.setdefault('meta_info', {})
        data.setdefault('ref_point', None)

        history = cls(**data)
        if global_start_time is not None:
            history.global_start_time = global_start_time
        history.update_observations(observations)

        logger.info(f'Loaded history (len={len(observations)}) from {filename}')
        return history

    # ========================================================================
    # Display & visualization (unchanged)
    # ========================================================================

    def get_str(self, max_candidates: int = 5) -> str:
        from prettytable import PrettyTable

        if self.empty():
            return 'No observation in History. Please run optimization.'

        candidates = self.get_incumbents() if self.num_objectives == 1 else self.get_pareto()
        n_candidates = len(candidates)
        if len(candidates) > max_candidates:
            hint = 'incumbents in history' if self.num_objectives == 1 else 'points on Pareto front'
            logger.info(f'Too many {hint}. Only show {max_candidates}/{n_candidates} of them.')
            candidates = candidates[:max_candidates]

        parameters = self.get_config_space().get_hyperparameter_names()
        if len(candidates) == 1:
            field_names = ["Parameters"] + ["Optimal Value"]
        else:
            field_names = ["Parameters"] + ["Optimal Value %d" % i for i in range(1, len(candidates) + 1)]
        table = PrettyTable(field_names=field_names, float_format=".6", align="l")
        for param in parameters:
            row = [param] + [obs.config.get_dictionary().get(param) for obs in candidates]
            table.add_row(row)
        if self.num_objectives == 1:
            table.add_row(["Optimal Objective Value"] + [obs.objectives[0] for obs in candidates])
        else:
            for i in range(self.num_objectives):
                table.add_row([f"Objective {i+1}"] + [obs.objectives[i] for obs in candidates])
        if self.num_constraints > 0:
            for i in range(self.num_constraints):
                table.add_row([f"Constraint {i+1}"] + [obs.constraints[i] for obs in candidates])
        row = ["Num Trials", len(self)]
        if n_candidates >= 3 and max_candidates >= 3:
            row += ["Num Best" if self.num_objectives == 1 else "Num Pareto", n_candidates] + [""] * (len(candidates)-3)
        else:
            row += [""] * (len(candidates) - 1)
        table.add_row(row)

        n_last_rows = 1
        raw_table = str(table)
        lines = raw_table.splitlines()
        hline = lines[2]
        lines.insert(3 + len(parameters), hline)
        if self.num_constraints > 0:
            lines.insert(4 + len(parameters) + self.num_objectives, hline)
        for i in range(n_last_rows):
            lines.insert(-(i + 1) * 2, hline)
        render_table = "\n".join(lines)
        return render_table

    def __str__(self):
        return self.get_str()

    __repr__ = __str__

    def get_importance(self, method='fanova', return_dict=False):
        from prettytable import PrettyTable
        from openbox.utils.feature_importance import get_fanova_importance, get_shap_importance

        if len(self) == 0:
            logger.error('No observations in history! Please run optimization process.')
            return dict() if return_dict else None

        config_space = self.get_config_space()
        parameters = list(config_space.get_hyperparameter_names())

        if method == 'fanova':
            importance_func = partial(get_fanova_importance, config_space=config_space)
        elif method == 'shap':
            importance_func = get_shap_importance
            if any([isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter))
                    for hp in config_space.get_hyperparameters()]):
                logger.warning("SHAP can not support categorical/ordinal hyperparameters well. "
                               "To analyze a space with categorical/ordinal hyperparameters, "
                               "we recommend setting the method to fanova.")
        else:
            raise ValueError("Invalid method for feature importance: %s" % method)

        X = self.get_config_array(transform='numerical')
        Y = self.get_objectives(transform='failed')
        cY = self.get_constraints(transform='failed,bilog')

        importance_dict = {
            'objective_importance': {param: [] for param in parameters},
            'constraint_importance': {param: [] for param in parameters},
        }
        if method == 'shap':
            importance_dict['objective_shap_values'] = []
            importance_dict['constraint_shap_values'] = []

        for i in range(self.num_objectives):
            feature_importance = importance_func(X, Y[:, i])
            if method == 'shap':
                feature_importance, shap_values = feature_importance
                importance_dict['objective_shap_values'].append(shap_values)
            for param, importance in zip(parameters, feature_importance):
                importance_dict['objective_importance'][param].append(importance)

        for i in range(self.num_constraints):
            feature_importance = importance_func(X, cY[:, i])
            if method == 'shap':
                feature_importance, shap_values = feature_importance
                importance_dict['constraint_shap_values'].append(shap_values)
            for param, importance in zip(parameters, feature_importance):
                importance_dict['constraint_importance'][param].append(importance)

        if return_dict:
            return importance_dict

        rows = []
        for param in parameters:
            row = [param, *importance_dict['objective_importance'][param],
                   *importance_dict['constraint_importance'][param]]
            rows.append(row)
        if self.num_objectives == 1 and self.num_constraints == 0:
            field_names = ["Parameter", "Importance"]
            rows.sort(key=lambda x: x[1], reverse=True)
        else:
            field_names = ["Parameter"] + ["Obj%d Importance" % i for i in range(1, self.num_objectives + 1)] + \
                          ["Cons%d Importance" % i for i in range(1, self.num_constraints + 1)]
        importance_table = PrettyTable(field_names=field_names, float_format=".6", align="l")
        importance_table.add_rows(rows)
        return importance_table

    def plot_convergence(self, true_minimum=None, name=None, clip_y=True,
                         title="Convergence plot", xlabel="Iteration", ylabel="Min objective value",
                         ax=None, alpha=0.3, yscale=None, color='C0', infeasible_color='C1', **kwargs):
        from openbox.visualization import plot_convergence
        if self.num_objectives > 1:
            raise ValueError('plot_convergence only supports single-objective optimization. '
                             'Please use plot_pareto_front or plot_hypervolumes instead.')

        y = self.get_objectives(transform='failed').reshape(-1)
        cy = self.get_constraints(transform='none', warn_invalid_value=False)
        ax = plot_convergence(y, cy, true_minimum, name, clip_y, title, xlabel, ylabel, ax, alpha, yscale,
                              color, infeasible_color, **kwargs)
        return ax

    def plot_pareto_front(self, title="Pareto Front", ax=None, alpha=0.3, color='C0', infeasible_color='C1', **kwargs):
        from openbox.visualization import plot_pareto_front
        assert self.num_objectives > 1
        if self.num_objectives not in [2, 3]:
            raise ValueError('plot_pareto_front only supports 2 or 3 objectives!')

        y = self.get_objectives(transform='failed')
        cy = self.get_constraints(transform='none', warn_invalid_value=False)
        ax = plot_pareto_front(y, cy, title, ax, alpha, color, infeasible_color, **kwargs)
        return ax

    def plot_hypervolumes(self, optimal_hypervolume=None, ref_point=None, logy=False, ax=None, **kwargs):
        from openbox.visualization import plot_curve

        assert self.num_objectives > 1
        ref_point = self.check_ref_point(ref_point)
        ref_point = ref_point if ref_point is not None else self.ref_point
        assert ref_point is not None, 'ref_point must be provided!'

        x = np.arange(len(self)) + 1
        y = self.compute_hypervolume(ref_point=ref_point, data_range='all')
        y = np.asarray(y, dtype=np.float64)
        if optimal_hypervolume is not None:
            y = optimal_hypervolume - y
            ylabel = 'Hypervolume Difference'
        else:
            ylabel = 'Hypervolume'
        if logy:
            ylabel = 'Log ' + ylabel
            y = np.log10(y)
        xlabel = 'Iteration'
        ax = plot_curve(x=x, y=y, xlabel=xlabel, ylabel=ylabel, ax=ax, **kwargs)
        return ax

    def visualize_html(self, logging_dir='logs/', open_html=True, show_importance=False, verify_surrogate=False,
                       task_info=None, optimizer=None, advisor=None, **kwargs):
        from openbox.visualization import build_visualizer, HTMLVisualizer

        option = 'advanced' if (show_importance or verify_surrogate) else 'basic'
        visualizer = build_visualizer(
            option=option, history=self, logging_dir=logging_dir,
            task_info=task_info, optimizer=optimizer, advisor=advisor, **kwargs)
        if visualizer.history is not self:
            visualizer.history = self
            visualizer.meta_data['task_id'] = self.task_id
        visualizer.visualize(open_html=open_html, show_importance=show_importance, verify_surrogate=verify_surrogate)
        return visualizer

    def visualize_hiplot(self, html_file: Optional[str] = None, **kwargs):
        from openbox.visualization import visualize_hiplot
        configs = self.configurations
        y = self.get_objectives(transform='none', warn_invalid_value=True)
        cy = self.get_constraints(transform='none', warn_invalid_value=True)
        exp = visualize_hiplot(configs=configs, y=y, cy=cy, html_file=html_file, **kwargs)
        return exp


class MultiStartHistory(History):
    """
    History for multi-start algorithms.
    """
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_observations = []  # type: List[List[Observation]]

    def restart(self):
        self.stored_observations.append(self.observations)
        self.observations = []

    def get_observations_for_all_restarts(self):
        return [obs for obs_list in self.stored_observations for obs in obs_list] + self.observations