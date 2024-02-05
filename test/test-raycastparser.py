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
        parser = RayCastParser([RayCastObjects.GOODGOAL, RayCastObjects.IMMOVABLE], 5)
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

        # Test pprint
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
        parser = RayCastParser([RayCastObjects.GOODGOAL, RayCastObjects.BADGOAL], 3)
        # fmt: off
        test_raycast = [0, 0, 0, 0, 0, 0, 0, 0.1,
                        0, 0, 0, 0, 0, 0, 0, 0.2,
                        0, 0, 0, 0, 0, 0, 0, 0.3]
        parsed_raycast = parser.parse(test_raycast)
        assert_array_equal(parsed_raycast, [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        # Test pprint
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
        # Test pprint
        pp_out = io.StringIO()
        with contextlib.redirect_stdout(pp_out):
            parser.prettyPrint(test_raycast)

        self.assertEqual(pp_out.getvalue(),
            "ARENA : [0.  0.  0.  0.1 0.  0.  0. ]\n" + 
            "MOVABLE : [0.3 0.  0.  0.  0.  0.  0. ]\n" + 
            "GOODGOALMULTI : [0.  0.5 0.  0.  0.  0.  0. ]\n")
        # fmt: on

    def test_MIX_PILLARBUTTON(self):
        """
        Check if the parser correctly identifies some objects including PILLARBUTTON while ignoring others.
        """
        parser = RayCastParser(
            [RayCastObjects.ARENA, RayCastObjects.PILLARBUTTON, RayCastObjects.MOVABLE],
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
        # Test pprint
        pp_out = io.StringIO()
        with contextlib.redirect_stdout(pp_out):
            parser.prettyPrint(test_raycast)

        self.assertEqual(pp_out.getvalue(),
            "ARENA : [0.  0.  0.  0.1 0.  0.  0. ]\n" +
            "PILLARBUTTON : [0.3 0.5 0.  0.1 0.6 0.4 0.2]\n" + 
            "MOVABLE : [0.3 0.  0.  0.  0.  0.  0. ]\n")
        # fmt: on
