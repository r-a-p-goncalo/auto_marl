# Reinforcement Learning

AutoMARL provides a complete RL abstraction stack.


## RL training configuration


### Environment

Environments provide observations, rewards, and episode transitions.

Supported integrations include:

* Gymnasium
* PettingZoo
* Custom environments

Examples found in the repository include:

* CartPole
* MountainCar
* Connect Four
* Multiwalker
* Cooperative Pong


### Agents

Agents encapsulate decision-making behavior in an environment episode. This means they may use the whole trajectory to make decisions instead of only the current observation.

### Policies

Policies convert observations into actions.


### Learners

Learners update policy parameters using received training data.


### Agent Trainers

Agent trainers handle the interaction between agents, memories of training and learners. This means making the decision of when to learn with what data.



### RL Trainers

RL trainers coordinate full training sessions. Syncronizing the environment and the multiple agents and trainers.


### RL Pipeline

The RL Pipeline is the highest-level training component.

It typically contains:

```text
RLPipeline
├── Environment
├── RL Trainer
├── Agents
└── Evaluator
```

Running the pipeline executes the complete experiment.


## Example RL Configuration

An example implementation of an RL algorithm is PPO for CartPole.

The component system can be resumed to:

![Architecture Diagram](/media/diagrams/rl_component_systems/PPO_CartPole.svg)

Conceptually:

```python
{
    "__type__": RLPipelineComponent,
    "input": {
        "environment": ...,
        "agents_input": ...,
        "trainer": ...
    }
}
```

Example configurations can be found under:

```text
base_configurations/
```

for environments such as:

* CartPole
* MountainCar
* Connect Four
* Multiwalker