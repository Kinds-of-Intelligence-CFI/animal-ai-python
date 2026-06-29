__all__ = [
    "UnityToGymnasiumWrapper",
    "UnityGymnasiumException",
    "ActionFlattener",
    "AnimalAIGymnasiumWrapper",
    "make_animalai_env_fns",
    "make_animalai_vec_env",
]


def __getattr__(name):
    if name in ("UnityToGymnasiumWrapper", "UnityGymnasiumException", "ActionFlattener"):
        from animalai.wrappers import unity_gymnasium

        return getattr(unity_gymnasium, name)
    if name == "AnimalAIGymnasiumWrapper":
        from animalai.wrappers.animalai_gymnasium import AnimalAIGymnasiumWrapper

        return AnimalAIGymnasiumWrapper
    if name in ("make_animalai_env_fns", "make_animalai_vec_env"):
        from animalai.wrappers import vector

        return getattr(vector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
