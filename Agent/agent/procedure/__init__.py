from .procedure import Procedure
from .variants.summary_based.dynamic import DynamicSummary

PROCEDURE_REGISTRY = {
    "default": DynamicSummary
}

def get_procedure(name: str, **kwargs) -> Procedure:
    if name in PROCEDURE_REGISTRY:
        return PROCEDURE_REGISTRY[name](**kwargs)
    else:
        raise ValueError(f"'{name}' is not a valid procedure variant. "
                         f"Please use one of the following: {list(PROCEDURE_REGISTRY.keys())}.")