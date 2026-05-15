import warnings

from animalai.LLM_scaffolds.environment_scaffolds import EnvironmentScaffold, FrameByFrameScaffold

__all__ = ["EnvironmentScaffold", "FrameByFrameScaffold", "KaggleWrapper", "act", "total_reward_scorer", "add_act_tool", "close_environment"]

warnings.warn(
    "The 'animalai.LLM_scaffolds' module is still in early development and may receive breaking changes. Use with caution and pin your version of 'animalai' to avoid unexpected issues.",
    FutureWarning,
    stacklevel=2,
)

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
    if name == "add_act_tool":
        from animalai.LLM_scaffolds.inspect_wrapper import add_act_tool
        return add_act_tool
    if name == "close_environment":
        from animalai.LLM_scaffolds.inspect_wrapper import close_environment
        return close_environment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
