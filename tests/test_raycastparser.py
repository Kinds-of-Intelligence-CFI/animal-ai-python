import unittest
import contextlib
import io

from numpy.testing import assert_array_equal
from animalai.raycastparser import RayCastParser, RayCastObjects

""" This file contains tests for the raycastparser class. Please keep it this way. """


class TestRayCastParser(unittest.TestCase):
    def test_GOODGOAL_IMMOVABLE(self):
        """
        Check if the parser correctly identifies GOODGOAL and IMMOVABLE objects
        and places them correctly in the parsed raycast array.
        """
        parser = RayCastParser(
            [RayCastObjects.GOODGOAL, RayCastObjects.IMMOVABLE], 5)
        # fmt: off
        test_raycast = [1, 1, 1, 1, 1, 1, 0, 0.1,
                        1, 1, 1, 1, 1, 1, 0, 0.2,
                        1, 1, 1, 1, 1, 1, 1, 0.3,
                        1, 1, 1, 1, 1, 1, 1, 0.4,
                        1, 1, 1, 1, 1, 1, 1, 0.5]
        parsed_raycast = parser.parse(test_raycast)
        assert_array_equal(parsed_raycast, [
            [0.3, 0.5, 0.1, 0.4, 0.2],
            [0.3, 0.5, 0.1, 0.4, 0.2]
        ])

        # Test prettyprint
        pp_out = io.StringIO()
        with contextlib.redirect_stdout(pp_out):
            parser.prettyPrint(test_raycast)

        self.assertEqual(pp_out.getvalue(),
            "GOODGOAL : [0.3 0.5 0.1 0.4 0.2]\n" +
            "IMMOVABLE : [0.3 0.5 0.1 0.4 0.2]\n",
        )
        # fmt: on

    def test_NO_OBJECTS(self):
        """
        Checks if the parser correctly identifies when no objects are detected.
        """
        parser = RayCastParser(
            [RayCastObjects.GOODGOAL, RayCastObjects.BADGOAL], 3)
        # fmt: off
        test_raycast = [0, 0, 0, 0, 0, 0, 0, 0.1,
                        0, 0, 0, 0, 0, 0, 0, 0.2,
                        0, 0, 0, 0, 0, 0, 0, 0.3]
        parsed_raycast = parser.parse(test_raycast)
        assert_array_equal(parsed_raycast, [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        # Test prettyprint
        pp_out = io.StringIO()
        with contextlib.redirect_stdout(pp_out):
            parser.prettyPrint(test_raycast)

        self.assertEqual(pp_out.getvalue(),
            "GOODGOAL : [0. 0. 0.]\n" + 
            "BADGOAL : [0. 0. 0.]\n")
        # fmt: on

    def test_MIXED_OBJECTS(self):
        """
        Checks if the parser correctly identifies some objects while ignoring others.
        """
        parser = RayCastParser(
            [
                RayCastObjects.ARENA,
                RayCastObjects.MOVABLE,
                RayCastObjects.GOODGOALMULTI,
            ],
            7,
        )
        # fmt: off
        test_raycast = [1, 0, 0, 0, 0, 0, 0, 0.1,
                        0, 1, 0, 0, 0, 0, 0, 0.2,
                        0, 0, 1, 0, 0, 0, 1, 0.3,
                        0, 0, 0, 1, 0, 0, 1, 0.4,
                        0, 0, 0, 0, 1, 0, 1, 0.5,
                        0, 0, 0, 0, 0, 1, 1, 0.6,
                        0, 0, 0, 0, 0, 0, 0, 0]
        parsed_raycast = parser.parse(test_raycast)
        assert_array_equal(parsed_raycast, [
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0 ]
            ])
        # Test prettyprint
        pp_out = io.StringIO()
        with contextlib.redirect_stdout(pp_out):
            parser.prettyPrint(test_raycast)

        self.assertEqual(pp_out.getvalue(),
            "ARENA : [0.  0.  0.  0.1 0.  0.  0. ]\n" + 
            "MOVABLE : [0.3 0.  0.  0.  0.  0.  0. ]\n" + 
            "GOODGOALMULTI : [0.  0.5 0.  0.  0.  0.  0. ]\n")
        # fmt: on

    def test_AGENT_in_full_tag_layout(self):
        """
        Check the parser reads the AGENT tag (the last of the 18 tags) and other
        objects from a full-size ray layout.
        """
        self.assertEqual(RayCastObjects.AGENT.value, 17)
        self.assertEqual(len(RayCastObjects), 18)

        parser = RayCastParser([RayCastObjects.AGENT, RayCastObjects.GOODGOAL], 3)
        number_of_tags = 18

        def ray(obj, distance):
            """One ray: a one-hot over the tags, a "hit nothing" flag, then the distance."""
            one_hot = [0] * number_of_tags
            if obj is not None:
                one_hot[obj.value] = 1
            hit_nothing = 1 if obj is None else 0
            return one_hot + [hit_nothing, distance]

        agent_distance, goal_distance = 0.4, 0.7
        # Rays arrive middle first, then right, then left; parse() returns each object's
        # distances left to right
        test_raycast = (
            ray(RayCastObjects.AGENT, agent_distance)
            + ray(None, 1.0)
            + ray(RayCastObjects.GOODGOAL, goal_distance)
        )
        self.assertEqual(len(test_raycast), 3 * (number_of_tags + 2))

        parsed_raycast = parser.parse(test_raycast)
        assert_array_equal(parsed_raycast, [
            [0.0, agent_distance, 0.0],  # AGENT: seen by the middle ray
            [goal_distance, 0.0, 0.0],   # GOODGOAL: seen by the left ray
        ])

    def test_objects_beyond_the_reported_tags_raise(self):
        """
        Asking for an object whose tag index is beyond the tags the rays report (so it could
        never be detected) raises instead of silently returning zeros.
        """
        twelve_tag_ray = [0] * 12 + [1, 1.0]
        for obj in (RayCastObjects.AGENT, RayCastObjects.DECOYGOAL):
            parser = RayCastParser([RayCastObjects.GOODGOAL, obj], 3)
            with self.assertRaises(ValueError) as context:
                parser.parse(twelve_tag_ray * 3)
            self.assertIn(obj.name, str(context.exception))

    def test_raycast_length_not_multiple_of_rays_raises(self):
        parser = RayCastParser([RayCastObjects.GOODGOAL], 3)
        with self.assertRaises(ValueError):
            parser.parse([0] * 41)

    def test_MIX_SPAWNERBUTTON(self):
        """
        Check if the parser correctly identifies some objects including SPAWNERBUTTON while ignoring others.
        """
        parser = RayCastParser(
            [RayCastObjects.ARENA, RayCastObjects.SPAWNERBUTTON, RayCastObjects.MOVABLE],
            7,
        )
        # fmt: off
        test_raycast = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.1,
                        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.2,
                        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.3,
                        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0.4,
                        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0.5,
                        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.6,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        parsed_raycast = parser.parse(test_raycast)
        assert_array_equal(parsed_raycast, [
            [0.0,  0.0,  0.0,  0.1, 0.0,  0.0,  0.0],
            [0.3, 0.5, 0.0, 0.1, 0.6, 0.4, 0.2],
            [0.3, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
            ])
        # Test prettyprint
        pp_out = io.StringIO()
        with contextlib.redirect_stdout(pp_out):
            parser.prettyPrint(test_raycast)

        self.assertEqual(pp_out.getvalue(),
            "ARENA : [0.  0.  0.  0.1 0.  0.  0. ]\n" +
            "SPAWNERBUTTON : [0.3 0.5 0.  0.1 0.6 0.4 0.2]\n" +
            "MOVABLE : [0.3 0.  0.  0.  0.  0.  0. ]\n")
        # fmt: on
