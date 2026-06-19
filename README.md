# AutoMARL

AutoMARL is a modular framework for defining, executing, evaluating, visualizing, and optimizing Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL) pipelines.

The objective of this project is to serve as a proof of concept as to how outer loop learning algorithms can be further automatized in the RL and MARL fields, focusing on the specific of Hyperparameter Optimization. The project's structure is prepared to be extended to allow for more of this algorithms to be implemented, such as Meta learning. This means being able to describe and serialize MARL solutions in a way that allows for them to be dynamically changed, created and evaluatedl.

The framework is built around a **component system** that allows complex experiments to be represented as hierarchical, serializable configurations. Components can be composed, saved, loaded, visualized, optimized, and executed without changing application code.


# Installation

```bash
pip install automarl
```

or install from source:

```bash
git clone <repository-url>
cd automarl
pip install -e .
```


# Core Concepts

Here we talk about the most basic functionalities offered by the framework, the building blocks in which the rest is built on top.

## Components

Every configurable element in AutoMARL inherits from:

```python
Component(metaclass=Schema)
```



This hierarchy allows entire experiments to be represented as a single component tree.


## Schemas

Schemas define the structure of a component.

Internally, the framework uses the `Schema` metaclass add aditional logic to the parameters of the components, namely:

* Merge inherited parameters
* Validate configuration inputs
* Handle default values
* Define exposed values
* Generate serialization metadata

Example:

```python
class MyComponent(Component):

    parameters_signature = {
        "learning_rate": ParameterSignature(),
        "batch_size": ParameterSignature()
    }
```

The schema automatically validates inputs and provides a consistent configuration interface.


## Input System

Component parameters are described through `ParameterSignature`, with the arguments:

| Parameter | Description |
|------------|-------------|
| `default_value` | Default value used when no value is provided. |
| `mandatory` | If `True`, the parameter must be supplied during component creation. |
| `possible_types` | List of accepted types for the parameter. |
| `validator` | Custom validation function executed on the supplied value. |
| `ignore_at_serialization` | Whether the parameter should be included when serializing the component configuration. |
| `priority` | Makes the parameter visible to parent components and optimization pipelines. |
| `on_pass` | Functions applied to the component after a value is passed to the input. |
| `custom_dict` | Extra custom values that allow for extra logic. |
| `description` | Human-readable description of the parameter. |

Example:

```python
parameters_signature = {
    "learning_rate": ParameterSignature(
        default_value=0.001,
        mandatory=False,
        possible_types=float,
        validity_verificator=lambda x: x > 0,
        description="Optimizer learning rate"
    )
}
```

### Advanced Input System

There are extensions of ParameterSignature...


## Component Systems

Components can be defined as parent or child of eachother, allowing for Component Trees and Component Forests to be made.

An example of a component tree in the case of Reinforcement Learning is:

```text
RL Pipeline
├── Environment
├── Trainer
│   ├── Agent Trainer
│   └── Learner
├── Agent
│   ├── Policy
│   └── Model
└── Evaluator
```

## Localizations

Describing where in a Component System a component is...



## Serialization

Every component can be converted into a JSON-compatible representation.

Example:

```python
component.save_configuration()
```

The resulting configuration contains:

```json
{
  "__type__": "<class 'automarl.component.Component'>",
  "name": "NameOfComponent",
  "input": {
    ...
  },
  "child_components" : [...]
}
```

Configurations can later be reconstructed:

```python
component = gen_component_from_dict(config)
```


### Custom serialization logic

## Graph Visualization

Since every experiment is represented as a component tree, configurations can be visualized as graphs

## Creating Custom Schemas

New functionality is added by creating custom schemas and registering it as a custom type.

Example:

```python
class MyNetwork(Component):

    parameters_signature = {
        "hidden_size": ParameterSignature()
    }
```

Because schemas are inherited, custom types integrate seamlessly with the framework.


# Basic Schemas

There are a number of basic schemas to represent common functionality components may need.

## Runnable Components

Runnable components implement executable behavior.

Typical examples:

* RL pipelines
* Training jobs
* Evaluation jobs
* Hyperparameter searches

Runnable components are derived from:

```python
ExecComponent
```

and expose methods such as:

```python
run()
```

## Artifact and Stateful Components

Artifact components own a directory on disk where they store generated artifacts, such as their configuration or basic text files.

```python
ArtifactComponent
```

Stateful components can save and reload their internal state, such as Neural Networks saving their model parameters.

Stateful components inherit from:

```python
StatefulComponent
```

and implement:

```python
_save_state_internal()
_load_state_internal()
```

## Seeded Components

Seeded components provide deterministic experiment execution.

They centralize random seed management across:

* Python
* NumPy
* PyTorch
* Environment seeds

This improves reproducibility.


## Logging Components

Logging components automatically generate:

* Text logs
* Result tables
* Metrics

Results can be aggregated across executions and stored alongside artifacts.





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


# Hyperparameter Optimization

AutoMARL includes an Optuna-based optimization framework.

Main components:

```text
HyperparameterOptimizationPipeline
├── Sampler
├── Pruner
├── Suggestions
└── Target Configuration
```


## Hyperparameter Suggestions

Suggestions define searchable parameters.

Examples:

```text
learning_rate
gamma
batch_size
epsilon
```

The optimization pipeline modifies the serialized configuration before component instantiation, allowing any parameter in the component tree to be optimized.


## Samplers

Sampling strategies determine how candidate values are generated.

Examples:

* Random Sampling
* TPE
* Other Optuna samplers


## Pruners

Pruners stop unpromising trials early.

Examples:

* Median Pruner
* Successive Halving
* Other Optuna-compatible pruners


## Optimization Workflow

1. Load base configuration.
2. Generate hyperparameter suggestions.
3. Apply suggestions to configuration.
4. Instantiate experiment.
5. Execute training.
6. Evaluate results.
7. Report objective value.
8. Repeat.


# Project Structure

```text
automarl/
├── components/
│   ├── basic_components/
│   ├── hp_opt/
│   ├── ml/
│   ├── rl/
│   └── loggers/
├── core/
├── utils/
├── cli/
└── base_configurations/
```