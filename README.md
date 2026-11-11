![steampunkFOURcrop](https://github.com/Kinds-of-Intelligence-CFI/animal-ai/assets/65875290/df798f4a-cb2c-416f-a150-093b9382a621)

[![Python Versions](https://img.shields.io/pypi/pyversions/animalai)](https://pypi.org/project/animalai/) [![PyPI](https://img.shields.io/pypi/v/animalai)](https://pypi.org/project/animalai/) [![PyPI Downloads](https://img.shields.io/pypi/dm/animalai)](https://pypi.org/project/animalai/)

# Animal-AI

This repository manages the Python interface for the Animal-AI environment.

The main project repository is located [here](https://github.com/Kinds-of-Intelligence-CFI/animal-ai).

* **Website:** [here](https://sites.google.com/csah.cam.ac.uk/animalai/)
* **Unity Source code:** [here](https://github.com/Kinds-of-Intelligence-CFI/animal-ai-unity-project)
* **Python Source code:** [here](https://github.com/Kinds-of-Intelligence-CFI/animal-ai-python)
* **Bug reports:** [here](https://github.com/Kinds-of-Intelligence-CFI/animal-ai/issues)

For more information about the ways you can contribute to Animal-AI, visit our website. If you’re unsure where to start or how your skills fit in, reach out! You can ask on GitHub, by opening a new issue or leaving a comment on a relevant issue that is already open.

If you are new to contributing to open source, [this](https://github.com/Kinds-of-Intelligence-CFI/animal-ai/blob/main/CONTRIBUTING.md) guide helps explain why, what, and how to successfully get involved.

## Multiple agents

An arena can contain several `Agent` items (one per agent, each with its
own `positions`, `rotations`, `skins`, `frozenAgentDelays` and optional `teams: [N]`), plus an optional
arena-level `episodeEnd: any | all`. All agents on a team share one behavior (`AnimalAI?team=0` by default)
and arrive together in its `DecisionSteps`, one row per agent:

```python
from animalai.actions import AAIActions, stack_actions

behavior = list(env.behavior_specs.keys())[0]
actions = AAIActions()
decision_steps, terminal_steps = env.get_steps(behavior)
for agent_id, obs in zip(decision_steps.agent_id, env.get_obs_dicts(decision_steps.obs)):
    print(agent_id, obs["health"], obs["position"])
# one action per agent, in the order of decision_steps.agent_id
env.set_actions(behavior, stack_actions([actions.FORWARDS for _ in decision_steps.agent_id]))
```

Agents in an arena finish their episodes together. Play mode (`play=True`) and the LLM scaffolds support
one agent only. See `tests/multiAgent.py` for a complete example of this direct interface.

### Wrappers for several agents

`AnimalAIParallelEnv` is a [PettingZoo](https://pettingzoo.farama.org/api/parallel/) `ParallelEnv` designed for when you have multiple agents in the environment but only 1 environment running at a time. The following is an example of how to use this wrapper:

```python
from animalai import AnimalAIEnvironment
from animalai.wrappers import AnimalAIParallelEnv

env = AnimalAIParallelEnv(AnimalAIEnvironment(arenas_configurations="multiAgentArena.yml", useRayCasts=True))
obs, infos = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

`make_animalai_multi_agent_vec_env` is used for when you want to use multiple agent in multiple concurrent environments, particular useful when trying to speed up training. Rather than addressing agents through a dict this wrapper stacks the agents and environments into `(M,N,.....)` where `M` is the number of environments and `N` is the number of agents. The following is an example of how to use the wrapper:

```python
from animalai.wrappers import make_animalai_multi_agent_vec_env

venv = make_animalai_multi_agent_vec_env(
    4, arenas_configurations="multiAgentArena.yml", env_kwargs={"useRayCasts": True}
)
obs, infos = venv.reset()  # obs["position"].shape == (4, number_of_agents, 3)
obs, rewards, terminations, truncations, infos = venv.step(venv.action_space.sample())
# rewards.shape == (4,): the sum of the agent rewards of each instance
# infos["agent_rewards"].shape == (4, number_of_agents): the reward of each agent
venv.close()
```

These rules apply to the two wrappers:

- The agents are `agent_0`, `agent_1`, ... in the order of the `Agent` items of the arena.
- Each agent has the same observations and actions as with `AnimalAIGymnasiumWrapper`. The `include_*` and
  `uint8_visual` arguments are the same.
- All agents must be on one team. An arena with more than one team raises an error. (worth noting that this means a team in the sense that they have the same action and observation spaces not that they are working collaboratively on the task.)
- The number of agents must not change. If a later arena has a different number of `Agent` items, the
  wrapper raises an error.
- Unity builds the next arena when an episode ends. `reset()` after an episode end does not reset Unity, so
  no arena is skipped.
- `reset(options={"arena_config": path})` loads a new configuration file.
- `reset(seed=...)` seeds the action space only. The seed of Unity is set when you create the
  `AnimalAIEnvironment`.
- The vector env resets an instance in the step after its episode ends. That step ignores the actions for
  that instance and gives it rewards of zero.


Rays report 18 object types, including other agents (`RayCastObjects.AGENT`). `RayCastParser` raises if
asked for an object the environment's rays don't report.

## Version History
* Unreleased
  + Multi-agent support: `AnimalAIEnvironment.get_obs_dict(obs, agent_index)` and `get_obs_dicts(obs)`,
    `stack_actions`, `RayCastObjects.AGENT`
  + Fixed `AAIActions(no_agents=N)` building a single discrete row for N > 1
  + `RayCastParser` raises for objects the environment's rays don't report (previously returned zeros)
  + Gymnasium wrappers and `FrameByFrameScaffold` give clear errors for multi-agent arenas
  + Added `AnimalAIParallelEnv`, a PettingZoo `ParallelEnv` for arenas with several agents
  + Added `make_animalai_multi_agent_vec_env` for several Unity instances with several agents each
  + `make_animalai_env_fns` accepts a `wrapper_cls` argument
  + The `gymnasium` extra now installs `pettingzoo`
* v6.1.3
  + Added UnityToGymnasium and AnimalAIGymnasiumWrapper wrappers  
  + Added make_animalai_vec_env / make_animalai_env_fns helpers for vectorised (parallel) environments  
  + Added an optional timeout argument to AnimalAIEnvironment
* v6.1.2
  + Fixed the version string for the executable autodownload
  + Misc fixes to the inspect wrapper
* v6.1.1
  + Updated the type hints in the scaffold to be supported by more python versions
* v6.1.0
  + Added automatic binary downloading
  + Added prototype scaffold for LLM integration
* v6.0.0
**Note: Version 6.0.0 is not backward compatible with previous versions of Animal-AI due to breaking changes.**
  + Minimum Python version is now `3.14.0` (breaking change).
  + Switch the mlagents-envs to our fork `mlagents-envs-aai 1.0.0`.
* v5.0.1
  + Added new contributor to list of PyPI contributors list.
* v5.0.0
**Note: Version 5.0.0 is not backward compatible with previous versions of Animal-AI due to breaking changes.**
  + Minimum Python version is now `3.10.0 (<3.10.13)` (breaking change).
  + Upgraded to ml-agents-ml-env `1.0.0`.
  + Adds no graphics monitor support.
* v4.1.0
  + Updated `RaycastParser` to accept new object:
    - `HollowBox`.
  + Added a new low-level random agent implemented on Braitenberg model.
  + Bug fixes and performance improvements, specifically on improving the reliability of the Braitenberg model.
  + Added built-in functionality to run yaml configuration files directly via Python.
* v4.0.1
  + Updated RaycastParser to accept two new objects:
    - `DecoyGoal` and `DecoyGoalBounce`.
* v4.0.0
**Note: Version 4.0.0 is not backward compatible with previous versions of Animal-AI due to breaking changes.**
  + Only implements _`mlagents 0.30.0`_ package to avoid dependency issues; also reduces package size considerably.
  + Cleaned up package and removed redundant files.
  + Implemented tests for the package.
  + Added documentation for the package.
* v3.0.5
  + Removed redundant packages in `setup.py`.
  + Added download stats.
* v3.0.4
  + Added current and past contributors.
  + Added project description and metadata.
* v3.0.3
  + Asserted python version to be 3.6.1 or higher, but less than 3.10.0 (exclusive).
  + Added package description to `setup.py` and package metadata.
* v3.0.2
  + Fixed major package dependency issues, related to `mlagents 0.30.0`, `protobuf`, and `shimmy`.
  + Updated project `setup.py` to accommodate the latest version of Animal-AI package dependencies.
    - Users can now use `pip install animalai` to install latest version of Animal-AI from PyPI effortlessly.
