from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ParamType(Enum):
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    PATH = "path"
    CHOICE = "choice"

# ============================================================================
# Each entry:
#   "type":      ParamType
#   "default":   default value
#   "choices":   list of valid values (only for CHOICE type)
#   "range":     (min, max) tuple for numeric, None means unbounded
#   "required":  bool
#   "desc":      description string
#   "config_path": list[str] — where it lives in the nested yaml config dict
#                  None means it's a top-level / cli-only param

PARAM_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ===== Core =====
    "config": {
        "type": ParamType.PATH,
        "default": "configs/base.yaml",
        "desc": "Path to config YAML file",
    },
    "opt": {
        "type": ParamType.CHOICE,
        "default": "MFES_SMAC",
        "choices": [
            "RS", "BO", "SMAC", "TPE",
            "MFES", "MFES_SMAC", "MFES_HB",
            "BOHB", "HYPERBAND",
            "REM`BO", "HESBO", "LLAMATUNE",
            "RGPE", "TAF",
        ],
        "desc": "Optimizer type",
    },
    "task": {
        "type": ParamType.STRING,
        "default": "test_ws",
        "desc": "Task name",
    },
    "log_level": {
        "type": ParamType.CHOICE,
        "default": "info",
        "choices": ["debug", "info", "warning", "error"],
        "desc": "Logging level",
    },
    "iter_num": {
        "type": ParamType.INT,
        "default": 40,
        "range": (1, 100000),
        "desc": "Number of iterations",
    },
    "seed": {
        "type": ParamType.INT,
        "default": 42,
        "range": (0, None),
        "desc": "Random seed",
    },

    # ===== Multi-fidelity =====
    "R": {
        "type": ParamType.INT,
        "default": None,
        "range": (1, 10000),
        "config_path": ["method_args", "R"],
        "desc": "Max fidelity budget",
    },
    "eta": {
        "type": ParamType.INT,
        "default": 3,
        "range": (2, 10),
        "config_path": ["method_args", "eta"],
        "desc": "Successive halving reduction factor",
    },
    "use_flatten_scheduler": {
        "type": ParamType.BOOL,
        "default": False,
        "desc": "Use flatten scheduler variant",
    },

    # ===== Paths =====
    "history_dir": {
        "type": ParamType.PATH,
        "default": None,
        "config_path": ["paths", "history_dir"],
        "desc": "History directory",
    },
    "data_dir": {
        "type": ParamType.PATH,
        "default": None,
        "config_path": ["paths", "data_dir"],
        "desc": "Data directory",
    },
    "save_dir": {
        "type": ParamType.PATH,
        "default": None,
        "config_path": ["paths", "save_dir"],
        "desc": "Save directory",
    },
    "target": {
        "type": ParamType.STRING,
        "default": None,
        "config_path": ["paths", "target"],
        "desc": "Target workload identifier",
    },
    "database": {
        "type": ParamType.STRING,
        "default": None,
        "config_path": ["database"],
        "desc": "Database name",
    },

    # ===== Compression =====
    "compress": {
        "type": ParamType.CHOICE,
        "default": "none",
        "choices": ["none", "shap", "expert", "llamatune"],
        "config_path": ["method_args", "cp_args", "strategy"],
        "desc": "Compression strategy",
    },
    "cp_topk": {
        "type": ParamType.INT,
        "default": None,
        "range": (1, 1000),
        "config_path": ["method_args", "cp_args", "topk"],
        "desc": "Compression top-k",
    },

    # ===== Warm start =====
    "warm_start": {
        "type": ParamType.CHOICE,
        "default": "none",
        "choices": ["none", "best_rover", "best_all", "rgpe"],
        "desc": "Warm start strategy",
    },
    "ws_init_num": {
        "type": ParamType.INT,
        "default": None,
        "range": (1, 100),
        "config_path": ["method_args", "ws_args", "init_num"],
        "desc": "Warm start initial config count",
    },
    "ws_topk": {
        "type": ParamType.INT,
        "default": None,
        "range": (1, 100),
        "config_path": ["method_args", "ws_args", "topk"],
        "desc": "Warm start top-k sources",
    },
    "ws_inner_surrogate_model": {
        "type": ParamType.STRING,
        "default": None,
        "config_path": ["method_args", "ws_args", "inner_surrogate_model"],
        "desc": "Surrogate model type for warm start",
    },

    # ===== Transfer learning =====
    "transfer": {
        "type": ParamType.CHOICE,
        "default": "none",
        "choices": ["none", "rgpe", "taf", "mapping"],
        "desc": "Transfer learning strategy",
    },
    "tl_topk": {
        "type": ParamType.INT,
        "default": None,
        "range": (1, 100),
        "config_path": ["method_args", "tl_args", "topk"],
        "desc": "Transfer learning top-k sources",
    },

    # ===== Exploration =====
    "rand_prob": {
        "type": ParamType.FLOAT,
        "default": 0.15,
        "range": (0.0, 1.0),
        "config_path": ["method_args", "rand_prob"],
        "desc": "Random exploration probability",
    },
    "rand_mode": {
        "type": ParamType.CHOICE,
        "default": "ran",
        "choices": ["ran", "rs"],
        "desc": "Random exploration mode",
    },

    # ===== System =====
    "target_system": {
        "type": ParamType.CHOICE,
        "default": "spark",
        "choices": ["spark", "flink", "hive", "generic"],
        "desc": "Target system to optimize",
    },

    # ===== Debug / flags =====
    "test_mode": {
        "type": ParamType.BOOL,
        "default": False,
        "desc": "Test mode (no cluster)",
    },
    "debug": {
        "type": ParamType.BOOL,
        "default": False,
        "desc": "Debug mode",
    },
    "resume": {
        "type": ParamType.PATH,
        "default": None,
        "desc": "Resume from path",
    },
    "backup_flag": {
        "type": ParamType.BOOL,
        "default": False,
        "desc": "Enable backup",
    },
    "use_cached_model": {
        "type": ParamType.BOOL,
        "default": False,
        "desc": "Use cached surrogate model",
    },
}


# Parameters that should NOT be written from CLI into the yaml config dict
SKIP_CLI_TO_CONFIG = frozenset({
    "config", "opt", "task", "log_level", "iter_num",
    "warm_start", "transfer", "backup_flag",
    "test_mode", "debug", "resume", "use_cached_model",
})