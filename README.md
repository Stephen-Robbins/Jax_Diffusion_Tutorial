# JAX Diffusion Tutorial

Score-based diffusion models implemented in JAX with Equinox.

## Overview

Tutorial implementation of diffusion models using JAX's functional programming paradigm and automatic differentiation. Includes both standard diffusion and diffusion bridge implementations.

## Why JAX?

- **Functional purity**: Immutable model parameters work naturally with diffusion SDEs
- **JIT compilation**: Significant speedup for score function evaluation
- **vmap**: Efficient batched operations for sampling
- **Automatic differentiation**: Clean score function implementation via `grad`

## Components

- `Score_nets.py`: Neural network architectures using Equinox
- `sde.py`: SDE solvers with JAX's functional approach
- `data.py`: Dataset utilities compatible with JAX arrays
- `train.py`: Training loop using Optax optimizers
- `plotting_functions.py`: Visualization helpers

## Key Features

- **Diffusion Bridge**: Implementation of conditional diffusion between two distributions
- **Checkpointing**: Model serialization with Equinox
- **Multiple SDEs**: VP-SDE and VE-SDE implementations

## Notebooks

- `Examples.ipynb`: Step-by-step tutorial with visualizations

## Usage

```python
import jax
import equinox as eqx
from train import train_model
from data import get_2d_data

# Generate data
key = jax.random.PRNGKey(0)
data = get_2d_data(key, n_samples=1000)

# Train model
model = train_model(data, n_epochs=500)

# Model weights are immutable - functional updates
params, static = eqx.partition(model, eqx.is_array)
```

## Checkpoints

Bridge models saved in `checkpoints/Bridge/` with different beta configurations.