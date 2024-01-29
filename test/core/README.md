# Core Test Folder - what is this?

This folder contains the core tests for the project. **These tests are run on every pull request to the repository and must pass**, where they are used to ensure that the core functionality of the project is working as expected. 

Note that if you require configs for your tests, look into the `test/test-Configs` folder. You can modify the configs manually specific for your tests (i.e. changing the object to spawn in the arena and/or it's parameters) to better suit your testing needs.

## Running Tests

To run the tests, simply run the following command in the root folder:

```bash
python -m unittest test/core/core-test-actions.py
```
The above will run a specific test file, where you cycle manually via specifiying the test file name. You would then repeat the above command for each test file you wish to run in any order you like.

Additionally, if you wish to run all tests in one batch, run the following command in the root folder:

```bash
python -m unittest discover -s test/core
```

This command will find and run every test in the core test folder, useful for running all tests at once (batch). Note that the order of the tests is not guaranteed, and may change between runs.

## Writing Custom Tests

To write a test, simply create a new file in the core test folder, and name it something like `your-custom-test.py`. Then, add the following code to the file:

```python
import unittest

class TestCore<name>(unittest.TestCase):
    def test_<name>(self):
        # Test code here
        pass
```

Then, add the following to the bottom of the file:

```python
if __name__ == '__main__':
    unittest.main()
```

This will allow the test to be run as a standalone file, as well as part of the test suite. Please let us know if you used any additional test libraries, so we can evaluate them for use in the project if they are useful.

# Test Configs - what are they?

This folder contains the configs for the tests in the `test/core` folder. These configs are used to specify the parameters of the tests, such as the object to spawn in the arena, the parameters of the object, and the parameters of the arena itself.

For a full list of objects and their parameters, see the [Arena Object Definitions](https://github.com/Kinds-of-Intelligence-CFI/animal-ai/blob/main/docs/Arena-Object-Definitions.md) guide on what objects you can spawn as well as their accepted parameters as part of your custom testing.

## When to use a Test Config

You should use a test config when you want to test a specific aspect of the agent's/object's behaviour, such as how it interacts with a specific object, or how it behaves in a specific arena. You can also specifify custom arena parameters such as the number of objects to spawn, object parameters such as position and size, and agent parameters such as the agent's position and rotation, to fully test the changes you made which may affect the environment or the agent.

## Important Notes

- The test configs are not meant to be comprehensive, but rather used to test specific aspects which may be affected by your changes. For example, if you changed the agent's movement speed, you may want to test how it interacts with a specific object, or how it behaves in a specific arena. You do not need to test every single object or arena.
  
- These configs are meant to be run manually by using specific operating systems, such as Windows or Linux. Please note in your pull request which operating systems you have tested your changes on, and which test configs you have used to test your changes.