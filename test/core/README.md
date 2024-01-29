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

To write a test, simply create a new file in the core test folder, and name it something like `core-test-<name>.py`. Then, add the following code to the file:

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