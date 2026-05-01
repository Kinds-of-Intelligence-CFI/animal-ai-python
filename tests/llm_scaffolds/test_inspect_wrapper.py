import base64
import io
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from animalai.LLM_scaffolds.inspect_wrapper import (
    ContentImage,
    encode_camera_obs,
    parse_action,
    act,
    add_act_tool,
    total_reward_scorer,
    close_environment,
)
from animalai.LLM_scaffolds.environment_scaffolds import EnvironmentScaffold


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
    return aai_state


def _make_scaffold(obs=None, reward: float = 1.0, done: bool = False):
    """Mock EnvironmentScaffold. obs should be HWC float32 to match encode_camera_obs."""
    scaffold = MagicMock(spec=EnvironmentScaffold)
    scaffold.available_actions = _ALL_ACTIONS
    if obs is None:
        obs = np.zeros((4, 4, 3), dtype=np.float32)
    scaffold.step.return_value = (obs, reward, done, {})
    return scaffold


# ---------------------------------------------------------------------------
# encode_camera_obs
# ---------------------------------------------------------------------------
#
# encode_camera_obs expects HWC float32 in [0, 1] — the raw ML-Agents camera
# observation format — and returns a base64-encoded PNG string.
#
# Compare with _process_obs (environment_scaffolds.py):
#   _process_obs — handles CHW → HWC transposition and float32 → uint8; returns
#                  a numpy array for internal scaffold use.
#   encode_camera_obs — expects HWC already (no transposition); clips before
#                       converting (more defensive); encodes to base64 PNG for
#                       consumption by a vision LLM.

class TestEncodeCameraObs(unittest.TestCase):
    def _decode(self, b64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(b64)))

    def test_returns_string(self):
        result = encode_camera_obs(np.zeros((4, 4, 3), dtype=np.float32))
        self.assertIsInstance(result, str)

    def test_valid_base64(self):
        result = encode_camera_obs(np.zeros((4, 4, 3), dtype=np.float32))
        base64.b64decode(result)  # must not raise

    def test_encodes_as_png(self):
        result = encode_camera_obs(np.full((4, 4, 3), 0.5, dtype=np.float32))
        self.assertEqual(self._decode(result).format, "PNG")

    def test_rgb_image_dimensions(self):
        # Input shape (H=8, W=6, C=3) → PIL reports (width=6, height=8)
        result = encode_camera_obs(np.zeros((8, 6, 3), dtype=np.float32))
        self.assertEqual(self._decode(result).size, (6, 8))

    def test_grayscale_channel_squeezed_to_mode_L(self):
        # (H, W, 1) should be squeezed so PIL uses mode "L"
        result = encode_camera_obs(np.full((4, 4, 1), 0.5, dtype=np.float32))
        self.assertEqual(self._decode(result).mode, "L")

    def test_pixel_values_scaled_correctly(self):
        # 0.5 * 255 = 127.5 → truncated to 127 by astype(uint8)
        obs = np.full((2, 2, 3), 0.5, dtype=np.float32)
        arr = np.array(self._decode(encode_camera_obs(obs)))
        np.testing.assert_array_equal(arr, np.full((2, 2, 3), 127, dtype=np.uint8))

    def test_values_above_1_clipped_to_255(self):
        # clip(0, 255) prevents overflow before astype(uint8)
        obs = np.full((2, 2, 3), 2.0, dtype=np.float32)
        arr = np.array(self._decode(encode_camera_obs(obs)))
        np.testing.assert_array_equal(arr, np.full((2, 2, 3), 255, dtype=np.uint8))


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
                arenas_configurations="arena.yml",
                useCamera=True,        # same as AAI default
                no_graphics=True,      # AAI default: False
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
    # Done — three consequences
    # ------------------------------------------------------------------

    async def test_done_copies_rewards_to_state_metadata(self):
        mock_scaffold = _make_scaffold(reward=2.0, done=True)
        state = _make_state()
        aai_state = _make_aai_state(scaffold=mock_scaffold)
        aai_state.rewards = [1.0]

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            await self._invoke(MagicMock(), state)
        self.assertEqual(state.metadata["rewards"], [1.0, 2.0])

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
        obs = np.zeros((4, 4, 3), dtype=np.float32)
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
    async def test_sums_rewards_from_metadata(self):
        scorer_fn = total_reward_scorer()
        state = _make_state(metadata={"rewards": [1.0, 2.0, 3.5]})
        score = await scorer_fn(state=state, target=MagicMock())
        self.assertAlmostEqual(score.value, 6.5)

    async def test_returns_zero_when_no_rewards_key_in_metadata(self):
        scorer_fn = total_reward_scorer()
        state = _make_state(metadata={})
        score = await scorer_fn(state=state, target=MagicMock())
        self.assertAlmostEqual(score.value, 0.0)


# ---------------------------------------------------------------------------
# close_environment
# ---------------------------------------------------------------------------

class TestCloseEnvironment(unittest.IsolatedAsyncioTestCase):
    async def test_closes_scaffold_when_env_is_present(self):
        mock_scaffold = MagicMock()
        aai_state = _make_aai_state(scaffold=mock_scaffold)

        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            await close_environment(MagicMock(), instance="test")
        mock_scaffold.close.assert_called_once()

    async def test_does_nothing_when_no_env_in_store(self):
        aai_state = _make_aai_state()  # AAI=None
        with patch("animalai.LLM_scaffolds.inspect_wrapper.store_as", return_value=aai_state):
            await close_environment(MagicMock(), instance="test")  # must not raise


if __name__ == "__main__":
    unittest.main()
