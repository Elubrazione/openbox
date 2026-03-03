import os
import sys
import argparse
import yaml
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .constants import PARAM_REGISTRY, SKIP_CLI_TO_CONFIG, ParamType

def validate_param(name: str, value: Any) -> Tuple[bool, Optional[str]]:
    """
    (is_valid, error_msg).  Unknown params pass silently.
    """
    spec = PARAM_REGISTRY.get(name)
    if spec is None:
        # Unknown param — not our business
        return True, None

    if value is None:
        if spec.get("required", False):
            return False, f"'{name}' is required"
        return True, None

    ptype = spec["type"]

    # --- type check ---
    type_ok = {
        ParamType.STRING:  lambda v: isinstance(v, str),
        ParamType.INT:     lambda v: isinstance(v, int) and not isinstance(v, bool),
        ParamType.FLOAT:   lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        ParamType.BOOL:    lambda v: isinstance(v, bool),
        ParamType.PATH:    lambda v: isinstance(v, str),
        ParamType.CHOICE:  lambda v: True,
    }
    if not type_ok.get(ptype, lambda v: True)(value):
        return False, f"'{name}' expects {ptype.value}, got {type(value).__name__}"

    # --- choices check ---
    choices = spec.get("choices")
    if choices is not None and value not in choices:
        return False, f"'{name}' must be one of {choices}, got '{value}'"

    # --- range check ---
    vrange = spec.get("range")
    if vrange is not None and ptype in (ParamType.INT, ParamType.FLOAT):
        lo, hi = vrange
        if lo is not None and value < lo:
            return False, f"'{name}' must be >= {lo}, got {value}"
        if hi is not None and value > hi:
            return False, f"'{name}' must be <= {hi}, got {value}"

    return True, None


def validate_params(params: Dict[str, Any],
                    strict: bool = False) -> Tuple[bool, List[str]]:
    errors = []
    for name, value in params.items():
        if strict and name not in PARAM_REGISTRY:
            errors.append(f"Unknown parameter: '{name}'")
            continue
        ok, msg = validate_param(name, value)
        if not ok:
            errors.append(msg)
    # required check
    for name, spec in PARAM_REGISTRY.items():
        if spec.get("required", False) and name not in params:
            errors.append(f"Required parameter '{name}' is missing")
    return len(errors) == 0, errors


def is_supported(name: str) -> bool:
    """Check if a parameter name is known in the registry."""
    return name in PARAM_REGISTRY


def get_default(name: str) -> Any:
    """Get default value from registry.  Returns None for unknown params."""
    spec = PARAM_REGISTRY.get(name)
    return spec["default"] if spec else None


def get_choices(name: str) -> Optional[List[Any]]:
    """Get valid choices list, or None."""
    spec = PARAM_REGISTRY.get(name)
    if spec:
        return spec.get("choices")
    return None


def get_config_path(name: str) -> Optional[List[str]]:
    spec = PARAM_REGISTRY.get(name)
    if spec:
        return spec.get("config_path")
    return None


# CLI Parsing
def build_parser(description: str = "OpenBox") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    for name, spec in PARAM_REGISTRY.items():
        _add_argument(parser, name, spec)
    return parser


def _add_argument(parser: argparse.ArgumentParser,
                  name: str,
                  spec: Dict[str, Any]) -> None:
    ptype = spec["type"]
    kwargs: Dict[str, Any] = {
        "default": spec.get("default"),
        "help": spec.get("desc", ""),
    }

    if ptype == ParamType.BOOL:
        # flag
        if not spec.get("default", False):
            kwargs.pop("default", None)
            parser.add_argument(f"--{name}", action="store_true", default=False,
                                help=spec.get("desc", ""))
        else:
            parser.add_argument(f"--no_{name}", action="store_false", dest=name,
                                default=True, help=spec.get("desc", ""))
        return

    if ptype == ParamType.CHOICE:
        kwargs["choices"] = spec.get("choices")
        kwargs["type"] = str
    elif ptype == ParamType.INT:
        kwargs["type"] = int
    elif ptype == ParamType.FLOAT:
        kwargs["type"] = float
    else:
        kwargs["type"] = str

    parser.add_argument(f"--{name}", **kwargs)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    ok, errors = validate_params(vars(args))
    if not ok:
        parser.error("\n".join(errors))

    return args

# YAML loading & dict merging
def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_yaml_with_includes(path: str) -> Dict[str, Any]:
    config = load_yaml(path)
    if "includes" not in config:
        return config

    includes = config.pop("includes")
    base_dir = os.path.dirname(path)
    merged = {}
    for inc in includes:
        inc_path = os.path.join(base_dir, inc) if not os.path.isabs(inc) else inc
        if os.path.exists(inc_path):
            merged = deep_merge(merged, load_yaml(inc_path))
    return deep_merge(merged, config)


def deep_merge(base: Dict[str, Any],
               override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result

# Apply CLI args → config dict
def apply_args_to_config(config: Dict[str, Any],
                         args: argparse.Namespace) -> Dict[str, Any]:
    config = deepcopy(config)
    for name, value in vars(args).items():
        if name in SKIP_CLI_TO_CONFIG:
            continue
        if not _should_override(value):
            continue
        cpath = get_config_path(name)
        if cpath is None:
            cpath = _search_key(config, name)
        if cpath is not None:
            _set_nested(config, cpath, value)
    return config


def _should_override(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (list, str)) and len(value) == 0:
        return False
    return True


def _set_nested(d: Dict[str, Any], path: List[str], value: Any) -> None:
    cur = d
    for k in path[:-1]:
        if k not in cur:
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


def _search_key(config: Dict[str, Any],
                key: str,
                prefix: Optional[List[str]] = None) -> Optional[List[str]]:
    prefix = prefix or []
    if key in config:
        return prefix + [key]
    for k, v in config.items():
        if isinstance(v, dict):
            found = _search_key(v, key, prefix + [k])
            if found:
                return found
    return None

def build_config(args: Optional[argparse.Namespace] = None,
                 config_file: Optional[str] = None,
                 config_dict: Optional[Dict[str, Any]] = None,
                 overrides: Optional[Dict[str, Any]] = None,
                 root_dir: Optional[str] = None) -> Dict[str, Any]:
    root = root_dir or os.path.dirname(os.path.dirname(__file__))

    if config_dict is not None:
        cfg = deepcopy(config_dict)
    elif config_file is not None:
        path = config_file if os.path.isabs(config_file) else os.path.join(root, config_file)
        cfg = load_yaml_with_includes(path)
    else:
        cfg = {}
    
    if overrides:
        cfg = deep_merge(cfg, overrides)

    if args is not None:
        cfg = apply_args_to_config(cfg, args)

    return cfg

def get_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur = config
    for k in key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def set_value(config: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    _set_nested(config, parts, value)