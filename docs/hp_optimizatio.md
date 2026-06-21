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