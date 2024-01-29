# Test Configs - what are they?

This folder contains the configs for the tests in the `test/core` folder. These configs are used to specify the parameters of the tests, such as the object to spawn in the arena, the parameters of the object, and the parameters of the arena itself.

For a full list of objects and their parameters, see the [Arena Object Definitions](https://github.com/Kinds-of-Intelligence-CFI/animal-ai/blob/main/docs/Arena-Object-Definitions.md) guide on what objects you can spawn as well as their accepted parameters as part of your custom testing.

## When to use a Test Config

You should use a test config when you want to test a specific aspect of the agent's/object's behaviour, such as how it interacts with a specific object, or how it behaves in a specific arena. You can also specifify custom arena parameters such as the number of objects to spawn, object parameters such as position and size, and agent parameters such as the agent's position and rotation, to fully test the changes you made which may affect the environment or the agent.

## Important Notes

- The test configs are not meant to be comprehensive, but rather used to test specific aspects which may be affected by your changes. For example, if you changed the agent's movement speed, you may want to test how it interacts with a specific object, or how it behaves in a specific arena. You do not need to test every single object or arena.
  
- These configs are meant to be run manually by using specific operating systems, such as Windows or Linux. Please note in your pull request which operating systems you have tested your changes on, and which test configs you have used to test your changes.