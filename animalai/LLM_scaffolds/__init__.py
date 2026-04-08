from animalai.LLM_scaffolds.environment_scaffolds import EnvironmentScaffold, FrameByFrameScaffold

__all__ = ["EnvironmentScaffold", "FrameByFrameScaffold", "KaggleWrapper", "act", "total_reward_scorer"]

def __getattr__(name):
    if name == "KaggleWrapper":
        from animalai.LLM_scaffolds.kaggle_wrapper import KaggleWrapper
        return KaggleWrapper
    if name == "act":
        from animalai.LLM_scaffolds.inspect_wrapper import act
        return act
    if name == "total_reward_scorer":
        from animalai.LLM_scaffolds.inspect_wrapper import total_reward_scorer
        return total_reward_scorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
