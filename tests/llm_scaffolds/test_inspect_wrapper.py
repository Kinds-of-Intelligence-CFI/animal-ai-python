import base64
import io
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from animalai.LLM_scaffolds.inspect_wrapper import (
    ContentImage,
    TERMINATION_ERROR,
    TERMINATION_INSPECT_LIMIT,
    encode_camera_obs,
    parse_action,
    act,
    add_act_tool,
    total_reward_scorer,
    close_environment,
)
from animalai.LLM_scaffolds.environment_scaffolds import (
    DONE_REASON_TERMINAL,
    DONE_REASON_TIMEOUT,
    EnvironmentScaffold,
    FrameByFrameScaffold,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_ACTIONS = [
    "FORWARD", "BACKWARD", "NOOP",
    "TURN_LEFT", "TURN_RIGHT", "FORWARD_LEFT", "FORWARD_RIGHT",
]


def _make_state(metadata=None):
    state = MagicMock()
    state.metadata = metadata if metadata is not None else {"arenas_configurations": "arena.yml"}
    state.tools = []
    state.completed = False
    return state


def _make_aai_state(scaffold=None):
    """
    Plain MagicMock standing in for AAIStateModel.

    AAIStateModel is a StoreModel (inspect_ai) with class-level backing storage,
    so instantiating it directly in tests shares state across all instances in the
    same process. Using a plain MagicMock avoids that entirely.
    """
    aai_state = MagicMock()
    aai_state.AAI = scaffold   # None triggers env creation; a scaffold bypasses it
    aai_state.rewards = []
    aai_state.termination_reason = None
    return aai_state


def _make_scaffold(obs=None, reward: float = 1.0, done: bool = False, done_reason=None):
    """Mock EnvironmentScaffold. obs should be HWC uint8, matching _process_obs output."""
    scaffold = MagicMock(spec=EnvironmentScaffold)
    scaffold.available_actions = _ALL_ACTIONS
    if obs is None:
        obs = np.zeros((4, 4, 3), dtype=np.uint8)
    # The scaffold only fills `info` with a done_reason when the episode ends.
    info = {"done_reason": done_reason} if done else {}
    scaffold.step.return_value = (obs, reward, done, info)
    return scaffold


# ---------------------------------------------------------------------------
# encode_camera_obs
# ---------------------------------------------------------------------------

class TestEncodeCameraObs(unittest.TestCase):
    def _decode(self, b64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(b64)))

    def test_returns_string(self):
        result = encode_camera_obs(np.zeros((4, 4, 3), dtype=np.uint8))
        self.assertIsInstance(result, str)

    def test_valid_base64(self):
        result = encode_camera_obs(np.zeros((4, 4, 3), dtype=np.uint8))
        base64.b64decode(result)  # must not raise

    def test_encodes_as_png(self):
        result = encode_camera_obs(np.full((4, 4, 3), 128, dtype=np.uint8))
        self.assertEqual(self._decode(result).format, "PNG")

    def test_rgb_image_dimensions(self):
        # Input shape (H=8, W=6, C=3) → PIL reports (width=6, height=8)
        result = encode_camera_obs(np.zeros((8, 6, 3), dtype=np.uint8))
        self.assertEqual(self._decode(result).size, (6, 8))

    def test_grayscale_2d_array_uses_mode_L(self):
        # _process_obs squeezes (H, W, 1) → (H, W) before returning, so
        # encode_camera_obs receives a 2D uint8 array for grayscale frames.
        result = encode_camera_obs(np.full((4, 4), 128, dtype=np.uint8))
        self.assertEqual(self._decode(result).mode, "L")

    def test_pixel_values_preserved(self):
        # uint8 values are passed through unchanged — no scaling or clipping.
        obs = np.full((2, 2, 3), 127, dtype=np.uint8)
        arr = np.array(self._decode(encode_camera_obs(obs)))
        np.testing.assert_array_equal(arr, obs)


# ---------------------------------------------------------------------------
# parse_action
# ---------------------------------------------------------------------------

class TestParseAction(unittest.TestCase):
    def setUp(self):
        self.wrapper = MagicMock(spec=EnvironmentScaffold)
        self.wrapper.available_actions = _ALL_ACTIONS

    def test_action_at_end_of_response(self):
        self.assertEqual(parse_action(self.wrapper, "I will go FORWARD"), "FORWARD")

    def test_finds_last_matching_word(self):
        # Scans words in reverse, so returns the last match in the string.
        self.assertEqual(parse_action(self.wrapper, "NOOP or FORWARD"), "FORWARD")

    def test_strips_trailing_punctuation(self):
        for punct in '.,!?;:\'"()[]{}':
            with self.subTest(punct=repr(punct)):
                self.assertEqual(
                    parse_action(self.wrapper, f"do FORWARD{punct}"), "FORWARD"
                )

    def test_strips_leading_punctuation(self):
        for punct in '.,!?;:\'"()[]{}':
            with self.subTest(punct=repr(punct)):
                self.assertEqual(
                    parse_action(self.wrapper, f"do {punct}FORWARD"), "FORWARD"
                )

    def test_single_word_action(self):
        self.assertEqual(parse_action(self.wrapper, "NOOP"), "NOOP")

    def test_case_sensitive_no_match_raises(self):
        with self.assertRaises(ValueError):
            parse_action(self.wrapper, "forward")

    def test_raises_value_error_when_no_action_found(self):
        with self.assertRaises(ValueError):
            parse_action(self.wrapper, "I have no idea what to do here")


# ---------------------------------------------------------------------------
# act
# ---------------------------------------------------------------------------

class TestAct(unittest.IsolatedAsyncioTestCase):
    def _invoke(self, scaffold_cls, state, instance=None, action="FORWARD"):
        """Call act(...) to obtain the Tool, then return the awaitable for execute."""
        tool_obj = act(scaffold_type=scaffold_cls, state=state, instance=instance)
        return tool_obj(action=action)

    # ------------------------------------------------------------------
    # Environment creation
    # ------------------------------------------------------------------

    async def test_creates_aai_env_when_none_in_store(self):
        mock_scaffold = _make_scaffold()
        mock_scaffold_cls = MagicMock(return_value=mock_scaffold)
        aai_state = _make_aai_state()  # AAI=None → triggers creation

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state), \
             patch("animalai.LLM_scaffolds.inspect_wrapper.AnimalAIEnvironment") as mock_env_cls:
            await self._invoke(mock_scaffold_cls, _make_state())
            mock_env_cls.assert_called_once()
            mock_scaffold_cls.assert_called_once_with(mock_env_cls.return_value)

    async def test_does_not_create_env_when_one_exists_in_store(self):
        mock_scaffold = _make_scaffold()
        aai_state = _make_aai_state(scaffold=mock_scaffold)  # AAI already set

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state), \
             patch("animalai.LLM_scaffolds.inspect_wrapper.AnimalAIEnvironment") as mock_env_cls:
            await self._invoke(MagicMock(), _make_state())
            mock_env_cls.assert_not_called()

    async def test_env_created_with_wrapper_defaults_when_metadata_omits_them(self):
        # Verifies the wrapper's intentional defaults, which differ from
        # AnimalAIEnvironment's own constructor defaults in several places.
        # Rewards are tracked in AAIStateModel.rewards (a per-step list) rather than
        # via EnvironmentScaffold.total_reward. The base class has no per-step reward
        # list, and the scorer needs the full list to report results. The two
        # accumulations are therefore parallel and independent.
        mock_scaffold = _make_scaffold()
        mock_scaffold_cls = MagicMock(return_value=mock_scaffold)
        state = _make_state(metadata={"arenas_configurations": "arena.yml"})
        aai_state = _make_aai_state()  # AAI=None → triggers creation

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state), \
             patch("animalai.LLM_scaffolds.inspect_wrapper.AnimalAIEnvironment") as mock_env_cls:
            await self._invoke(mock_scaffold_cls, state)
            mock_env_cls.assert_called_once_with(
                file_name=None,
                arenas_configurations="arena.yml",
                useCamera=True,        # same as AAI default
                no_graphics=False,      # AAI default: False
                resolution=84,         # AAI default: 150
                base_port=5005,        # same as AAI default
                worker_id=0,           # same as AAI default
                seed=None,             # AAI default: 0
                timescale=5,           # AAI default: 1
                targetFrameRate=-1,    # AAI default: 60
            )

    # ------------------------------------------------------------------
    # Step failure
    # ------------------------------------------------------------------

    async def test_step_failure_closes_env_and_reraises(self):
        mock_scaffold = _make_scaffold()
        mock_scaffold.step.side_effect = RuntimeError("env crashed")
        aai_state = _make_aai_state(scaffold=mock_scaffold)

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            with self.assertRaises(RuntimeError):
                await self._invoke(MagicMock(), _make_state())
            mock_scaffold.close.assert_called_once()

    async def test_step_failure_records_error_termination_reason(self):
        mock_scaffold = _make_scaffold()
        mock_scaffold.step.side_effect = RuntimeError("env crashed")
        aai_state = _make_aai_state(scaffold=mock_scaffold)

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            with self.assertRaises(RuntimeError):
                await self._invoke(MagicMock(), _make_state())
        self.assertEqual(aai_state.termination_reason, TERMINATION_ERROR)

    # ------------------------------------------------------------------
    # Reward accumulation
    # ------------------------------------------------------------------

    async def test_reward_appended_to_aai_state_rewards(self):
        mock_scaffold = _make_scaffold(reward=3.5, done=False)
        aai_state = _make_aai_state(scaffold=mock_scaffold)

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            await self._invoke(MagicMock(), _make_state())
        self.assertEqual(aai_state.rewards, [3.5])

    # ------------------------------------------------------------------
    # Done — consequences
    # ------------------------------------------------------------------

    async def test_done_records_termination_reason_from_info(self):
        mock_scaffold = _make_scaffold(done=True, done_reason=DONE_REASON_TERMINAL)
        aai_state = _make_aai_state(scaffold=mock_scaffold)

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            await self._invoke(MagicMock(), _make_state())
        self.assertEqual(aai_state.termination_reason, DONE_REASON_TERMINAL)

    async def test_done_sets_state_completed(self):
        mock_scaffold = _make_scaffold(done=True)
        state = _make_state()
        aai_state = _make_aai_state(scaffold=mock_scaffold)

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            await self._invoke(MagicMock(), state)
        self.assertTrue(state.completed)

    async def test_done_closes_environment(self):
        mock_scaffold = _make_scaffold(done=True)
        aai_state = _make_aai_state(scaffold=mock_scaffold)

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            await self._invoke(MagicMock(), _make_state())
        mock_scaffold.close.assert_called_once()

    # ------------------------------------------------------------------
    # Observation return types
    # ------------------------------------------------------------------

    async def test_obs_string_returned_as_is(self):
        mock_scaffold = _make_scaffold()
        mock_scaffold.step.return_value = ("text observation", 0.0, False, {})
        aai_state = _make_aai_state(scaffold=mock_scaffold)

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            result = await self._invoke(MagicMock(), _make_state())
        self.assertEqual(result, "text observation")

    async def test_obs_ndarray_returned_as_content_image_with_data_url(self):
        obs = np.zeros((4, 4, 3), dtype=np.uint8)
        mock_scaffold = _make_scaffold(obs=obs)
        aai_state = _make_aai_state(scaffold=mock_scaffold)

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            result = await self._invoke(MagicMock(), _make_state())
        self.assertIsInstance(result, ContentImage)
        self.assertTrue(result.image.startswith("data:image/png;base64,"))

    async def test_obs_unexpected_type_raises_value_error(self):
        mock_scaffold = _make_scaffold()
        mock_scaffold.step.return_value = (42, 0.0, False, {})
        aai_state = _make_aai_state(scaffold=mock_scaffold)

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            with self.assertRaises(ValueError):
                await self._invoke(MagicMock(), _make_state())


# ---------------------------------------------------------------------------
# add_act_tool
# ---------------------------------------------------------------------------

class TestAddActTool(unittest.IsolatedAsyncioTestCase):
    async def test_appends_one_tool_to_state_tools(self):
        solver_fn = add_act_tool(scaffold_type=MagicMock())
        state = _make_state()
        await solver_fn(state=state, generate=MagicMock())
        self.assertEqual(len(state.tools), 1)


# ---------------------------------------------------------------------------
# total_reward_scorer
# ---------------------------------------------------------------------------

class TestTotalRewardScorer(unittest.IsolatedAsyncioTestCase):
    async def test_sums_rewards_from_store(self):
        aai_state = _make_aai_state()
        aai_state.rewards = [1.0, 2.0, 3.5]
        scorer_fn = total_reward_scorer()
        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            score = await scorer_fn(state=_make_state(), target=MagicMock())
        self.assertAlmostEqual(score.value, 6.5)

    async def test_returns_zero_when_no_rewards_in_store(self):
        aai_state = _make_aai_state()  # rewards == []
        scorer_fn = total_reward_scorer()
        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            score = await scorer_fn(state=_make_state(), target=MagicMock())
        self.assertAlmostEqual(score.value, 0.0)

    async def test_scores_rewards_even_when_episode_not_done(self):
        # Regression: an episode stopped by an Inspect limit never hits the
        # `done` branch, but its accumulated reward must still be scored, and
        # the missing env reason falls back to the harness limit.
        aai_state = _make_aai_state()
        aai_state.rewards = [-0.1, -0.1, -0.1]
        aai_state.termination_reason = None  # env never reported done
        scorer_fn = total_reward_scorer()
        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            score = await scorer_fn(state=_make_state(), target=MagicMock())
        self.assertAlmostEqual(score.value, -0.3)
        self.assertEqual(score.metadata["termination_reason"], TERMINATION_INSPECT_LIMIT)

    async def test_includes_termination_reason_and_step_count_in_metadata(self):
        aai_state = _make_aai_state()
        aai_state.rewards = [1.0, 2.0]
        aai_state.termination_reason = DONE_REASON_TERMINAL
        scorer_fn = total_reward_scorer()
        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            score = await scorer_fn(state=_make_state(), target=MagicMock())
        self.assertEqual(score.metadata["termination_reason"], DONE_REASON_TERMINAL)
        self.assertEqual(score.metadata["num_steps"], 2)


# ---------------------------------------------------------------------------
# close_environment
# ---------------------------------------------------------------------------

class TestCloseEnvironment(unittest.IsolatedAsyncioTestCase):
    async def test_closes_scaffold_when_env_is_present(self):
        mock_scaffold = MagicMock()
        aai_state = _make_aai_state(scaffold=mock_scaffold)

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state), \
             patch("animalai.LLM_scaffolds.inspect_wrapper.transcript"):
            await close_environment(_make_state(), instance="test")
        mock_scaffold.close.assert_called_once()

    async def test_does_nothing_when_no_env_in_store(self):
        aai_state = _make_aai_state()  # AAI=None
        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state), \
             patch("animalai.LLM_scaffolds.inspect_wrapper.transcript") as mock_transcript:
            await close_environment(_make_state(), instance="test")  # must not raise
        mock_transcript.assert_not_called()

    async def test_records_env_termination_reason_in_metadata(self):
        aai_state = _make_aai_state(scaffold=MagicMock())
        aai_state.rewards = [1.0, 2.0]
        aai_state.termination_reason = DONE_REASON_TERMINAL
        state = _make_state(metadata={})

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state), \
             patch("animalai.LLM_scaffolds.inspect_wrapper.transcript"):
            await close_environment(state, instance="test")

        outcome = state.metadata["episode_outcome"]
        self.assertEqual(outcome["termination_reason"], DONE_REASON_TERMINAL)
        self.assertAlmostEqual(outcome["total_reward"], 3.0)
        self.assertEqual(outcome["num_steps"], 2)

    async def test_defaults_to_inspect_limit_when_no_env_reason(self):
        # Env never reported done (e.g. message limit) -> generic harness reason.
        aai_state = _make_aai_state(scaffold=MagicMock())
        aai_state.rewards = [-0.1, -0.1]
        aai_state.termination_reason = None
        state = _make_state(metadata={})

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state), \
             patch("animalai.LLM_scaffolds.inspect_wrapper.transcript"):
            await close_environment(state, instance="test")

        self.assertEqual(
            state.metadata["episode_outcome"]["termination_reason"],
            TERMINATION_INSPECT_LIMIT,
        )

    async def test_emits_transcript_info_event(self):
        aai_state = _make_aai_state(scaffold=MagicMock())
        aai_state.rewards = [1.0]
        aai_state.termination_reason = DONE_REASON_TIMEOUT
        state = _make_state(metadata={})

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state), \
             patch("animalai.LLM_scaffolds.inspect_wrapper.transcript") as mock_transcript:
            await close_environment(state, instance="test")

        mock_transcript.return_value.info.assert_called_once()
        _, kwargs = mock_transcript.return_value.info.call_args
        self.assertEqual(kwargs.get("source"), "animalai")


# ---------------------------------------------------------------------------
# FrameByFrameScaffold._classify_terminal
# ---------------------------------------------------------------------------

class _FakeTerminalSteps:
    """Minimal stand-in for ml-agents TerminalSteps (only the fields we read)."""
    def __init__(self, interrupted: bool, reward: float):
        self.interrupted = np.array([interrupted])
        self.reward = np.array([reward], dtype=np.float32)


class TestClassifyTerminal(unittest.TestCase):
    def test_interrupted_is_env_timeout(self):
        ts = _FakeTerminalSteps(interrupted=True, reward=1.0)
        self.assertEqual(FrameByFrameScaffold._classify_terminal(ts), DONE_REASON_TIMEOUT)

    def test_non_interrupted_is_terminal_regardless_of_reward(self):
        # Reward sign is left for the caller to interpret, not classified here.
        for reward in (2.5, 0.0, -1.0):
            with self.subTest(reward=reward):
                ts = _FakeTerminalSteps(interrupted=False, reward=reward)
                self.assertEqual(FrameByFrameScaffold._classify_terminal(ts), DONE_REASON_TERMINAL)


if __name__ == "__main__":
    unittest.main()
