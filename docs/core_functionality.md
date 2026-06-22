
# Core Concepts

Here we talk about the most basic functionalities offered by the framework, the building blocks in which the rest is built on top.

## Components

Every configurable element in AutoMARL inherits from:

```python
Component(metaclass=Schema)
```



This hierarchy allows entire experiments to be represented as a single component tree.


## Schemas and Components

Simply put, `schema` is what we call the class of a `component`

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

The schema automatically validates inputs and provides a consistent configuration interface. This is done to be easier to catch problems in complex component systems, where the stacktrace will be unintelligible if an input fails to verify certain conditions and is only noted when it is used.


## Input System

Component parameters are described through `ParameterSignature`, with the arguments:

| Parameter | Description |
|------------|-------------|
| `default_value` | Default value used when no value is provided. |
| `mandatory` | If `True`, the parameter must be supplied during component creation. |
| `possible_types` | List of accepted types for the parameter. |
| `validator` | Custom validation function executed on the supplied value. |
| `ignore_at_serialization` | Whether the parameter should be included when serializing the component configuration. |
| `priority` | An integer that defines the order in which inputs should be processed. |
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


