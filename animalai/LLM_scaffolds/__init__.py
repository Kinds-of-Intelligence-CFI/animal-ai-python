from animalai.LLM_scaffolds.environment_scaffolds import EnvironmentScaffold, FrameByFrameScaffold

__all__ = ["EnvironmentScaffold", "FrameByFrameScaffold", "KaggleWrapper"]

def __getattr__(name):
    if name == "KaggleWrapper":
        from animalai.LLM_scaffolds.kaggle_wrapper import KaggleWrapper
        return KaggleWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
