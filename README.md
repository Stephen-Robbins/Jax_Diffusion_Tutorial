# JAX Diffusion Tutorial

This repository demonstrates a simple score-based diffusion model implemented with [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox). The code trains a neural network to approximate the score of a variance-preserving SDE and optionally a bridge-conditioned variant.

## Key Idea
The model learns the gradient of the data log-density (the *score*) by sampling from a diffusion process described by a stochastic differential equation. Once trained, the network can generate new samples by integrating the reverse-time SDE starting from random noise.

## Installation
```bash
pip install jax equinox optax matplotlib numpy
```

## Usage
```python
import jax.random as jr
from src import train, sde, score_nets, data

key = jr.PRNGKey(0)
points = data.generate_happy_face(10000, key=key)
model = score_nets.Bridge_Diffusion_Net(jr.PRNGKey(1), input_dim=2, output_dim=2)
bridge = sde.BridgeVPSDE()
train.main(model, points, bridge, num_steps=10000)
```

## Related Work
The approach follows the score-based generative modeling framework introduced in [Song et al., 2021](https://arxiv.org/abs/2011.13456).

## Folder Guide
| Path | Purpose |
|------|---------|
| `src/` | Library code and training script |
| `notebooks/` | Tutorial notebook |
| `checkpoints/` | Saved model checkpoints |

## License
[MIT](LICENSE)

---
*Maintainer: Stephen Robbins (<sw2tennis@gmail.com>)*
