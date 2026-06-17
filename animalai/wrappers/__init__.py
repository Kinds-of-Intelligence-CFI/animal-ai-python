__all__ = [
    "UnityToGymnasiumWrapper",
    "UnityGymnasiumException",
    "ActionFlattener",
    "AnimalAIGymnasiumWrapper",
]


def __getattr__(name):
    if name in ("UnityToGymnasiumWrapper", "UnityGymnasiumException", "ActionFlattener"):
        from animalai.wrappers import unity_gymnasium

        return getattr(unity_gymnasium, name)
    if name == "AnimalAIGymnasiumWrapper":
        from animalai.wrappers.animalai_gymnasium import AnimalAIGymnasiumWrapper

        return AnimalAIGymnasiumWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
