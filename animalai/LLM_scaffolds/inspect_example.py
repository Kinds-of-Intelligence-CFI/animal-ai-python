from inspect_ai import Task, eval, task
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    basic_agent,
    solver,
    system_message,
    use_tools,
)
from inspect_ai.dataset import MemoryDataset, Sample


from animalai.LLM_scaffolds.environment_scaffolds import FrameByFrameScaffold
from animalai.LLM_scaffolds.inspect_wrapper import act, add_act_tool, total_reward_scorer


@task
def AAI__eval(agent_solver: Solver | None = None) -> Task:
    dataset = MemoryDataset(samples=[
        Sample(
            input="Please path to the goal area marked by the green object.",
            metadata={
                "arenas_configurations": "animalai/arenas/GoodGoal_Fixed.yml",
            },
            )
    ])

    solver_chain = [
        system_message(FrameByFrameScaffold.get_default_system_prompt()),
        add_act_tool(scaffold_type=FrameByFrameScaffold),
        agent_solver or basic_agent(),
    ]

    return Task(
        dataset=dataset,
        solver=solver_chain,
        scorer=total_reward_scorer(),
        message_limit=30,
    )