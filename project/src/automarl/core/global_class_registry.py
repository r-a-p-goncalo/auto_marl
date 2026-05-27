import sys
import types
from typing import Type
import inspect

from automarl.component import Schema
from automarl.components.loggers.global_logger import globalWriteLine

from dataclasses import dataclass
from typing import Callable

CUSTOM_MODULE_NAME = "__custom_classes.custom_classes"


@dataclass
class ClassSpec:
    name : str
    bases: tuple
    namespace: dict

_CUSTOM_CLASS_SPECS: dict[str, ClassSpec] = {}

def register_custom_class(*, name: str, bases: tuple[type, ...], namespace: dict | None = None):    

    if namespace is None:
        namespace = {}

    module = get_or_create_custom_module()

    # Ensure fresh namespace like real class statement (__prepare__ semantics)
    prepared = Schema.__prepare__(name, bases)
    prepared.update(namespace)

    cls = Schema(name, bases, prepared)
    cls.__module__ = CUSTOM_MODULE_NAME

    setattr(module, name, cls)

    spec = ClassSpec(name=name, bases=bases, namespace=namespace)

    key = f"{CUSTOM_MODULE_NAME}.{name}"
    _CUSTOM_CLASS_SPECS[key] = spec

    globalWriteLine(f"Registered custom class: {key}", file="global_classes.txt")

    return cls


def get_registered_classes_generators():
    return dict(_CUSTOM_CLASS_SPECS)

def has_registered_classes_generators():
    return len(_CUSTOM_CLASS_SPECS) > 0

def clear_registry():
    _CUSTOM_CLASS_SPECS.clear()


def get_or_create_custom_module():

    if CUSTOM_MODULE_NAME in sys.modules:
        return sys.modules[CUSTOM_MODULE_NAME]

    pkg_name = "__custom_classes"

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = []  # mark as package
        sys.modules[pkg_name] = pkg

    module = types.ModuleType(CUSTOM_MODULE_NAME)
    sys.modules[CUSTOM_MODULE_NAME] = module
    return module


def clone_class_into_module(cls, target_module_name: str):
    
    namespace = dict(cls.__dict__)

    # Remove runtime-only attributes
    namespace.pop("__dict__", None)
    namespace.pop("__weakref__", None)

    new_cls = Schema(
        cls.__name__,
        cls.__bases__,
        dict(cls.__dict__)
    )

    new_cls.__module__ = target_module_name

    if hasattr(cls, "__source__"):
        new_cls.__source__ = cls.__source__

    return new_cls


def _serialize_bases(bases: tuple[type, ...]) -> str:
    return ", ".join(f"get_class_from(\"{str(base)}\")" for base in bases) or "object"




def _serialize_namespace(namespace: dict) -> list[str]:
    lines: list[str] = []

    for key, value in namespace.items():

        if callable(value):
            try:
                src = inspect.getsource(value)
                lines.append(src)
                continue
            except (OSError, TypeError):
                pass

        lines.append(f" {key} = {repr(value)}")

    if not lines:
        lines.append(" pass")

    return lines


def serialize_registered_classes() -> str:
    """
    Serialize registered classes into real Python class statements
    """


    lines = [
    "# Auto-generated custom classes",
    "from automarl.component import Component",
    "from automarl.schema import Schema",
    "from automarl.utils.class_util import get_class_from",
    "from automarl.core.global_class_registry import register_custom_class",
    "",
    ]


    for spec in _CUSTOM_CLASS_SPECS.values():
        bases_str = _serialize_bases(spec.bases)

        lines.append(f"{spec.name} = register_custom_class(name=\"{spec.name}\", bases=({bases_str}))")

        lines.append("")

    return "\n".join(lines)


def load_custom_classes(file):

    globalWriteLine(f"Loading custom classes...", file="global_classes.txt")

    module_name = f"_automarl_custom_classes"

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    globalWriteLine(f"Finished loading custom classes", file="global_classes.txt")
