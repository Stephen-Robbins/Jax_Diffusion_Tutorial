# JAX Diffusion Tutorial

Score-based diffusion models implemented with JAX, Equinox, and Optax,
including diffusion and bridge-conditioned examples.

## Installation

```bash
pip install jax equinox optax matplotlib numpy
```

## Components

- `Score_nets.py`: Equinox network architectures.
- `sde.py`: Diffusion and bridge SDE implementations.
- `data.py`: Toy dataset generators.
- `train.py`: Training and checkpoint utilities.
- `plotting_functions.py`: Visualization helpers.
- `Examples.ipynb`: Interactive examples and experiment configurations.

The modules live in the repository root. For example:

```python
import jax.random as jr
from data import generate_happy_face

points = generate_happy_face(10000, key=jr.PRNGKey(0))
```

See `Examples.ipynb` for model and SDE configurations. Checkpoints are linked
to the canonical local Data directory and are not included in Git.

## References and license

The implementation follows score-based generative modeling
([Song et al., 2021](https://arxiv.org/abs/2011.13456)).
See [LICENSE](LICENSE) for the MIT license.
