import numpy as np
from openbox import logger
from openbox.core.base_advisor import BaseAdvisor
from openbox.core.base import build_surrogate, build_acq_func
from openbox.acq_optimizer import build_acq_optimizer


class MiniAdvisor(BaseAdvisor):
  def __init__(
    self,
    config_space,
    init_num=3,
    optimization_strategy='bo',
    surrogate_type='gp',
    acq_type='ei',
    acq_optimizer_type='local_random',
    init_strategy='random_explore_first'
  ):
    super().__init__(config_space=config_space)
    self.rand_prob = 0.1

    # fixed parameters
    self.optimization_strategy = optimization_strategy
    self.surrogate_type = surrogate_type
    self.acq_type = acq_type
    self.acq_optimizer_type = acq_optimizer_type
    self.init_strategy = init_strategy

    self.initial_configurations = self.create_initial_design(init_num)
    self.init_num = len(self.initial_configurations)

    logger.info(
      '[BO auto selection]' +
      f' surrogate_type: {self.surrogate_type}.' +
      f' acq_type: {self.acq_type}.' +
      f' acq_optimizer_type: {self.acq_optimizer_type}.' +
      f' init_strategy: {self.init_strategy}.'
    )
    self.surrogate_model = build_surrogate(config_space=self.config_space, rng=self.rng)  # default value of func_str is 'gp'
    self.acquisition_function = build_acq_func(model=self.surrogate_model)  # default value of func_str is 'ei'
    self.acq_optimizer = build_acq_optimizer(config_space=self.config_space, rng=self.rng)  # default value of func_str is 'local_random'


  def create_initial_design(self, init_num):
    default_config = self.config_space.get_default_configuration()
    num_random_config = init_num - 1
    candidate_configs = self.sample_random_configs(self.config_space, 100)
    initial_configs = self.max_min_distance(default_config, candidate_configs, num_random_config)
    valid_configs = []
    for config in initial_configs:
      try:
        config.is_valid_configuration()
      except ValueError:
        continue
      valid_configs.append(config)
    if len(valid_configs) != len(initial_configs):
      logger.warning('Only %d/%d valid configurations are generated for initial design strategy: %s. '
                      'Add more random configurations.'
                      % (len(valid_configs), len(initial_configs), self.init_strategy))
      num_random_config = init_num - len(valid_configs)
      valid_configs += self.sample_random_configs(self.config_space, num_random_config, excluded_configs=valid_configs)
    return valid_configs


  def max_min_distance(self, default_config, src_configs, num):
    min_dis = list()
    initial_configs = list()
    initial_configs.append(default_config)

    for config in src_configs:
      dis = np.linalg.norm(config.get_array() - default_config.get_array())
      min_dis.append(dis)
    min_dis = np.array(min_dis)

    for _ in range(num):
      furthest_config = src_configs[np.argmax(min_dis)]
      initial_configs.append(furthest_config)
      min_dis[np.argmax(min_dis)] = -1

      for j in range(len(src_configs)):
        if src_configs[j] in initial_configs:
          continue
        updated_dis = np.linalg.norm(src_configs[j].get_array() - furthest_config.get_array())
        min_dis[j] = min(updated_dis, min_dis[j])
    return initial_configs


  def get_suggestion(self):
    history = self.history
    num_config_evaluated = len(history)
    num_config_successful = history.get_success_count()
    if num_config_evaluated < self.init_num:
      res = self.initial_configurations[num_config_evaluated]
      return res
    if self.rng.random() < self.rand_prob:
      logger.info('Sample random config. rand_prob=%f.' % self.rand_prob)
      res = self.sample_random_configs(self.config_space, 1, excluded_configs=history.configurations)[0]
      return res
    X = history.get_config_array(transform='scale')
    Y = history.get_objectives(transform='infeasible')

    if num_config_successful < max(self.init_num, 1):
      logger.warning('No enough successful initial trials! Sample random configuration.')
      res = self.sample_random_configs(self.config_space, 1, excluded_configs=history.configurations)[0]
      return res

    self.surrogate_model.train(X, Y[:, 0])
    incumbent_value = history.get_incumbent_value()
    self.acquisition_function.update(model=self.surrogate_model, eta=incumbent_value, num_data=num_config_evaluated)
    challengers = self.acq_optimizer.maximize(
      acquisition_function=self.acquisition_function,
      history=history,
      num_points=5000,
    )
    for config in challengers:
      if config not in history.configurations:
        return config

    logger.warning('Cannot get non duplicate configuration from BO candidates (len=%d). '
                    'Sample random config.' % (len(challengers), ))
    return self.sample_random_configs(self.config_space, 1, excluded_configs=history.configurations)[0]