import unittest
from unittest.mock import MagicMock
import numpy as np

from animalai.LLM_scaffolds.environment_scaffolds import (
    ACTION_MAP,
    AVAILABLE_ACTIONS,
    DEFAULT_DONE,
    DEFAULT_LAST_FRAME,
    DEFAULT_TOTAL_REWARD,
    DEFAULT_TOTAL_STEPS,
    FrameByFrameScaffold,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_obs(value: float = 0.5, shape: tuple = (3, 4, 4)) -> np.ndarray:
    """Float32 array simulating a (C, H, W) Unity camera observation."""
    return np.full(shape, value, dtype=np.float32)


def _expected_frame(obs: np.ndarray) -> np.ndarray:
    """What _process_obs produces for a (C, H, W) float32 array."""
    return (obs.transpose(1, 2, 0) * 255).astype(np.uint8)


def _make_decision_steps(obs_array: np.ndarray, reward: float = 0.0) -> MagicMock:
    ds = MagicMock()
    ds.obs = [[obs_array]]
    ds.reward = [reward]
    ds.__len__.return_value = 1
    return ds


def _make_empty_terminal_steps() -> MagicMock:
    ts = MagicMock()
    ts.__len__.return_value = 0
    return ts


def _make_terminal_steps(obs_array: np.ndarray, reward: float = 0.0) -> MagicMock:
    ts = MagicMock()
    ts.obs = [[obs_array]]
    ts.reward = [reward]
    ts.__len__.return_value = 1
    return ts


def _make_env(initial_obs: np.ndarray | None = None) -> MagicMock:
    """Mock env whose get_steps always returns the same decision step."""
    if initial_obs is None:
        initial_obs = _make_obs()
    env = MagicMock()
    env.behavior_specs = {"Brain?team=0": MagicMock()}
    env.get_steps.return_value = (
        _make_decision_steps(initial_obs),
        _make_empty_terminal_steps(),
    )
    return env


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

class TestGetDefaultSystemPrompt(unittest.TestCase):
    def test_returns_string(self):
        self.assertIsInstance(FrameByFrameScaffold.get_default_system_prompt(), str)

    def test_contains_all_available_actions(self):
        prompt = FrameByFrameScaffold.get_default_system_prompt()
        for action in AVAILABLE_ACTIONS:
            self.assertIn(action, prompt)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInitialisation(unittest.TestCase):
    def setUp(self):
        self.obs = _make_obs()
        self.env = _make_env(self.obs)
        self.scaffold = FrameByFrameScaffold(self.env, skipframe=3)

    def test_env(self):
        self.assertIs(self.scaffold.env, self.env)

    def test_skipframe(self):
        self.assertEqual(self.scaffold.skipframe, 3)

    def test_behavior_name(self):
        self.assertEqual(self.scaffold.behavior_name, "Brain?team=0")

    def test_total_reward_property(self):
        self.assertEqual(self.scaffold.total_reward, DEFAULT_TOTAL_REWARD)

    def test_total_steps(self):
        self.assertEqual(self.scaffold._total_steps, DEFAULT_TOTAL_STEPS)

    def test_done(self):
        self.assertEqual(self.scaffold._done, DEFAULT_DONE)

    def test_last_frame_populated_by_collect_obs(self):
        # _collect_obs() runs at the end of __init__, so last_frame holds the
        # processed initial observation rather than DEFAULT_LAST_FRAME.
        np.testing.assert_array_equal(
            self.scaffold.last_frame, _expected_frame(self.obs)
        )

    def test_is_finished_false(self):
        self.assertFalse(self.scaffold.is_finished())


# ---------------------------------------------------------------------------
# Available actions
# ---------------------------------------------------------------------------

class TestAvailableActions(unittest.TestCase):
    def test_matches_action_map_keys(self):
        scaffold = FrameByFrameScaffold(_make_env())
        self.assertEqual(scaffold.available_actions, list(ACTION_MAP.keys()))
        self.assertEqual(scaffold.available_actions, AVAILABLE_ACTIONS)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset(unittest.TestCase):
    def setUp(self):
        self.obs = _make_obs()
        self.env = _make_env(self.obs)
        self.scaffold = FrameByFrameScaffold(self.env, skipframe=2)
        # Dirty every resettable field before calling reset.
        self.scaffold._total_steps = 5
        self.scaffold._total_reward = 3.14
        self.scaffold._done = True
        self.scaffold.last_frame = _make_obs(0.9)
        self.scaffold.reset()

    def test_total_reward_reset(self):
        self.assertEqual(self.scaffold.total_reward, DEFAULT_TOTAL_REWARD)

    def test_total_steps_reset(self):
        self.assertEqual(self.scaffold._total_steps, DEFAULT_TOTAL_STEPS)

    def test_done_reset(self):
        self.assertEqual(self.scaffold._done, DEFAULT_DONE)

    def test_last_frame_repopulated_after_reset(self):
        # reset() clears last_frame to DEFAULT_LAST_FRAME then calls _collect_obs(),
        # so the final value should be the processed observation from the env.
        np.testing.assert_array_equal(
            self.scaffold.last_frame, _expected_frame(self.obs)
        )

    def test_env_reset_called(self):
        self.env.reset.assert_called()

    def test_is_finished_false_after_reset(self):
        self.assertFalse(self.scaffold.is_finished())


# ---------------------------------------------------------------------------
# Step — action mapping
# ---------------------------------------------------------------------------

class TestStepActionMapping(unittest.TestCase):
    def _discrete_sent_for(self, action_name: str) -> np.ndarray:
        env = _make_env()
        scaffold = FrameByFrameScaffold(env, skipframe=1)
        scaffold.step(action_name)
        action_tuple = env.set_actions.call_args[0][1]
        return action_tuple.discrete

    def test_all_actions_produce_correct_discrete_tuple(self):
        for action_name, (b0, b1) in ACTION_MAP.items():
            with self.subTest(action=action_name):
                np.testing.assert_array_equal(
                    self._discrete_sent_for(action_name), [[b0, b1]]
                )


# ---------------------------------------------------------------------------
# Step — skipframe
# ---------------------------------------------------------------------------

class TestSkipframe(unittest.TestCase):
    def _env_always_deciding(self, obs: np.ndarray | None = None) -> MagicMock:
        """Env that never terminates, suitable for counting env.step() calls."""
        env = MagicMock()
        env.behavior_specs = {"b": MagicMock()}
        env.get_steps.return_value = (
            _make_decision_steps(obs or _make_obs()),
            _make_empty_terminal_steps(),
        )
        return env

    def _step_count(self, skipframe: int, scaffold_steps: int = 1) -> int:
        env = self._env_always_deciding()
        scaffold = FrameByFrameScaffold(env, skipframe=skipframe)
        env.step.reset_mock()
        for _ in range(scaffold_steps):
            scaffold.step("NOOP")
        return env.step.call_count

    def test_skipframe_1(self):
        self.assertEqual(self._step_count(skipframe=1), 1)

    def test_skipframe_3(self):
        self.assertEqual(self._step_count(skipframe=3), 3)

    def test_skipframe_5(self):
        self.assertEqual(self._step_count(skipframe=5), 5)

    def test_skipframe_accumulates_over_multiple_scaffold_steps(self):
        self.assertEqual(self._step_count(skipframe=2, scaffold_steps=4), 8)

    def test_stops_early_on_terminal(self):
        # With skipframe=5 but a terminal on the first inner step, env.step()
        # should only be called once.
        obs = _make_obs()
        env = MagicMock()
        env.behavior_specs = {"b": MagicMock()}
        env.get_steps.side_effect = [
            (_make_decision_steps(obs), _make_empty_terminal_steps()),   # init
            (_make_decision_steps(obs), _make_empty_terminal_steps()),   # loop iter 1 pre-action
            (_make_decision_steps(obs), _make_terminal_steps(obs, 1.0)), # loop iter 1 post-action → terminal
        ]
        scaffold = FrameByFrameScaffold(env, skipframe=5)
        scaffold.step("NOOP")
        self.assertEqual(env.step.call_count, 1)


# ---------------------------------------------------------------------------
# Step — sequencing
# ---------------------------------------------------------------------------

class TestStepSequencing(unittest.TestCase):
    def test_get_steps_set_actions_step_get_steps_order(self):
        obs = _make_obs()
        env = MagicMock()
        env.behavior_specs = {"b": MagicMock()}
        env.get_steps.return_value = (
            _make_decision_steps(obs), _make_empty_terminal_steps()
        )
        scaffold = FrameByFrameScaffold(env, skipframe=1)

        call_log = []

        def log_get_steps(name):
            call_log.append("get_steps")
            return (_make_decision_steps(obs), _make_empty_terminal_steps())

        def log_set_actions(*args):
            call_log.append("set_actions")

        def log_env_step():
            call_log.append("env_step")

        env.get_steps.side_effect = log_get_steps
        env.set_actions.side_effect = log_set_actions
        env.step.side_effect = log_env_step

        scaffold.step("FORWARD")

        self.assertEqual(call_log, ["get_steps", "set_actions", "env_step", "get_steps"])


# ---------------------------------------------------------------------------
# Step — reward accumulation (decision steps)
# ---------------------------------------------------------------------------

class TestRewardAccumulationDecisionSteps(unittest.TestCase):
    def setUp(self):
        obs = _make_obs()
        env = MagicMock()
        env.behavior_specs = {"b": MagicMock()}
        empty_ts = _make_empty_terminal_steps()
        # Reward is only read from the post-action get_steps call; the
        # pre-action call's reward value is intentionally different to confirm
        # it is ignored.
        env.get_steps.side_effect = [
            (_make_decision_steps(obs, reward=99.0), empty_ts),  # init
            (_make_decision_steps(obs, reward=99.0), empty_ts),  # step 1 pre  (ignored)
            (_make_decision_steps(obs, reward=1.0),  empty_ts),  # step 1 post
            (_make_decision_steps(obs, reward=99.0), empty_ts),  # step 2 pre  (ignored)
            (_make_decision_steps(obs, reward=2.5),  empty_ts),  # step 2 post
        ]
        self.scaffold = FrameByFrameScaffold(env, skipframe=1)
        _, self.r1, _, _ = self.scaffold.step("NOOP")
        _, self.r2, _, _ = self.scaffold.step("NOOP")

    def test_step_reward_1(self):
        self.assertAlmostEqual(self.r1, 1.0)

    def test_step_reward_2(self):
        self.assertAlmostEqual(self.r2, 2.5)

    def test_total_reward_accumulates(self):
        self.assertAlmostEqual(self.scaffold.total_reward, 3.5)


# ---------------------------------------------------------------------------
# Step — reward accumulation (terminal step)
# ---------------------------------------------------------------------------

class TestRewardAccumulationTerminalStep(unittest.TestCase):
    def test_terminal_reward_returned_and_accumulated(self):
        obs = _make_obs()
        env = MagicMock()
        env.behavior_specs = {"b": MagicMock()}
        empty_ts = _make_empty_terminal_steps()
        env.get_steps.side_effect = [
            (_make_decision_steps(obs),              empty_ts),                      # init
            (_make_decision_steps(obs),              empty_ts),                      # step pre
            (_make_decision_steps(obs), _make_terminal_steps(obs, reward=7.0)),      # step post → terminal
        ]
        scaffold = FrameByFrameScaffold(env, skipframe=1)
        _, reward, done, _ = scaffold.step("FORWARD")
        self.assertTrue(done)
        self.assertAlmostEqual(reward, 7.0)
        self.assertAlmostEqual(scaffold.total_reward, 7.0)


# ---------------------------------------------------------------------------
# Step — done flag
# ---------------------------------------------------------------------------

class TestDone(unittest.TestCase):
    def test_done_set_on_terminal_visible_via_is_finished_and_step_return(self):
        obs = _make_obs()
        env = MagicMock()
        env.behavior_specs = {"b": MagicMock()}
        empty_ts = _make_empty_terminal_steps()
        env.get_steps.side_effect = [
            (_make_decision_steps(obs), empty_ts),
            (_make_decision_steps(obs), empty_ts),
            (_make_decision_steps(obs), _make_terminal_steps(obs, 1.0)),
        ]
        scaffold = FrameByFrameScaffold(env, skipframe=1)
        self.assertFalse(scaffold.is_finished())
        _, _, done, _ = scaffold.step("NOOP")
        self.assertTrue(done)
        self.assertTrue(scaffold.is_finished())

    def test_done_set_when_no_decision_agents(self):
        obs = _make_obs()
        env = MagicMock()
        env.behavior_specs = {"b": MagicMock()}
        empty_ds = MagicMock()
        empty_ds.__len__.return_value = 0
        empty_ts = _make_empty_terminal_steps()
        env.get_steps.side_effect = [
            (_make_decision_steps(obs), empty_ts),  # init
            (empty_ds,                 empty_ts),   # step pre: 0 agents → done
        ]
        scaffold = FrameByFrameScaffold(env, skipframe=1)
        _, _, done, _ = scaffold.step("NOOP")
        self.assertTrue(done)
        self.assertTrue(scaffold.is_finished())


# ---------------------------------------------------------------------------
# last_frame updates
# ---------------------------------------------------------------------------

class TestLastFrame(unittest.TestCase):
    def test_set_during_init(self):
        obs = _make_obs(0.5)
        scaffold = FrameByFrameScaffold(_make_env(obs))
        np.testing.assert_array_equal(scaffold.last_frame, _expected_frame(obs))

    def test_updated_on_decision_step(self):
        obs_init = _make_obs(0.2)
        obs_step = _make_obs(0.8)
        env = MagicMock()
        env.behavior_specs = {"b": MagicMock()}
        empty_ts = _make_empty_terminal_steps()
        env.get_steps.side_effect = [
            (_make_decision_steps(obs_init), empty_ts),
            (_make_decision_steps(obs_init), empty_ts),  # step pre
            (_make_decision_steps(obs_step), empty_ts),  # step post
        ]
        scaffold = FrameByFrameScaffold(env, skipframe=1)
        scaffold.step("NOOP")
        np.testing.assert_array_equal(scaffold.last_frame, _expected_frame(obs_step))

    def test_updated_on_terminal_step(self):
        obs_init = _make_obs(0.2)
        obs_term = _make_obs(0.9)
        env = MagicMock()
        env.behavior_specs = {"b": MagicMock()}
        empty_ts = _make_empty_terminal_steps()
        terminal = _make_terminal_steps(obs_term, reward=1.0)
        env.get_steps.side_effect = [
            (_make_decision_steps(obs_init), empty_ts),
            (_make_decision_steps(obs_init), empty_ts),   # step pre
            (_make_decision_steps(obs_init), terminal),   # step post → terminal frame
        ]
        scaffold = FrameByFrameScaffold(env, skipframe=1)
        scaffold.step("NOOP")
        np.testing.assert_array_equal(scaffold.last_frame, _expected_frame(obs_term))


if __name__ == "__main__":
    unittest.main()
