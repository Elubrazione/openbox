import sys
import numpy as np
from pathlib import Path
cur_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(cur_path.as_posix())

from openbox import space as sp
from openbox import logger
from openbox.core.generic_advisor import Advisor
from openbox.core.mini_advisor import MiniAdvisor
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import run_obj_func
from openbox.utils.util_funcs import parse_result
from openbox.utils.history import Observation
from openbox.optimizer.base import BOBase


class MiniSMBO(BOBase):
  def __init__(
    self,
    objective_function,
    config_space,
    advisor,
    max_runs=100,
  ):
    self.objective_function = objective_function
    self.config_space = config_space
    self.max_runs = max_runs
    self.config_advisor = advisor
    self.iteration_id = 0
    self.FAILED_PERF = [np.inf]


  def iterate(self, timeout=None) -> Observation:

    config = self.config_advisor.get_suggestion()
    if config in self.config_advisor.history.configurations:
      logger.warning('Evaluating duplicated configuration: %s' % config)

    obj_args, obj_kwargs = (config,), dict()
    result = run_obj_func(self.objective_function, obj_args, obj_kwargs, timeout)

    ret, timeout_status, traceback_msg, elapsed_time = (
      result['result'], result['timeout'], result['traceback'], result['elapsed_time']
    )
    trial_state = TIMEOUT if timeout_status else FAILED if traceback_msg is not None else SUCCESS
    if trial_state == SUCCESS:
      objectives, constraints, extra_info = parse_result(ret)
    else:
      objectives, constraints, extra_info = self.FAILED_PERF.copy()

    observation = Observation(
      config=config, objectives=objectives, constraints=constraints,
      trial_state=trial_state, elapsed_time=elapsed_time, extra_info=extra_info,
    )
    self.config_advisor.update_observation(observation)
    self.iteration_id += 1
    logger.info('Iter %d, objectives: %s. constraints: %s.' % (self.iteration_id, objectives, constraints))
    return observation


def mishra(config: sp.Configuration):
  X = np.array([config['x%d' % i] for i in range(2)])
  x, y = X[0], X[1]
  t1 = np.sin(y) * np.exp((1 - np.cos(x))**2)
  t2 = np.cos(x) * np.exp((1 - np.sin(y))**2)
  t3 = (x - y)**2
  result = dict()
  result['objectives'] = [t1 + t2 + t3, ]
  return result


if __name__ == "__main__":
  params = {
    'float': {
      'x0': (-10, 0, -5),
      'x1': (-6.5, 0, -3.25)
    }
  }
  space = sp.Space()
  space.add_variables([
    sp.Real(name, *para) for name, para in params['float'].items()
  ])
  max_runs = 10
  advisor = MiniAdvisor(space)

  optimizer = MiniSMBO(objective_function=mishra, config_space=space, max_runs=max_runs, advisor=advisor)
  for _ in range(max_runs):
    observation = optimizer.iterate()
    print(observation)