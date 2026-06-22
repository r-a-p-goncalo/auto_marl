# AutoMARL

AutoMARL is a modular framework for defining, executing, evaluating, visualizing, and optimizing Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL) pipelines.

The objective of this project is to serve as a proof of concept as to how outer loop learning algorithms can be further automatized in the RL and MARL fields, focusing on the specific of Hyperparameter Optimization. The project's structure is prepared to be extended to allow for more of this algorithms to be implemented, such as Meta learning. This means being able to describe and serialize MARL solutions in a way that allows for them to be dynamically changed, created and evaluatedl.

The framework is built around a **component system** that allows complex experiments to be represented as hierarchical, serializable configurations. Components can be composed, saved, loaded, visualized, optimized, and executed without changing application code.



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

# Why AutoMARL?

AutoMARL focuses on experiment automation.

Unlike traditional RL libraries that require manually written training scripts,
AutoMARL represents complete RL pipelines as serializable component trees.

This enables:

- Dynamic pipeline generation
- Hyperparameter optimization
- Experiment reproducibility
- Pipeline visualization
- Future support for meta-learning and AutoRL






