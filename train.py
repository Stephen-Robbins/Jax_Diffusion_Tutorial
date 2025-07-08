
"""Training utilities for score-based diffusion models."""

import functools as ft
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax  # https://github.com/deepmind/optax
from plotting_functions import plot_points
import os
import datetime


def batch_loss_fn(
    model,
    sde,
    data: jnp.ndarray,
    data_y: jnp.ndarray,
    t1: float,
    key: jr.KeyArray,
) -> jnp.ndarray:
    """Compute the training loss for a batch."""

    batch_size = data.shape[0]
    
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    # Low-discrepancy sampling over t to reduce variance
    t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)
   
    loss_fn = ft.partial(sde.score_loss, model)
    loss_fn = jax.vmap(loss_fn)
    
    return jnp.mean(loss_fn(data, data_y, t, losskey))


@eqx.filter_jit
def make_step(
    model,
    sde,
    data: jnp.ndarray,
    data_y: jnp.ndarray,
    t1: float,
    key: jr.KeyArray,
    opt_state,
    opt_update,
) -> tuple[jnp.ndarray, eqx.Module, jr.KeyArray, object]:
    
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, sde, data, data_y, t1, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state

def dataloader(data: jnp.ndarray, batch_size: int, *, key: jr.KeyArray):
    """Simple data generator yielding batches."""
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    
    while True:
        key, subkey = jr.split(key, 2)
        perm = jr.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield data[batch_perm]
            start = end
            end = start + batch_size

def main(
    model,
    data,
    sde,
    data_y=None,
    t1=1.0,
    # Optimisation hyperparameters
    num_steps=100_000,
    lr=1e-5,
    batch_size=10_000,
    print_every=1000,
    sample_every=10_000,
    sample_size=1000,
    # Seed
    seed=22094,
    checkpoint_every=1000,
    checkpoint_dir="checkpoints/Diffusion",
    filename=None,
) -> None:
    """Run a training loop for the supplied model and SDE."""

    
    key = jr.PRNGKey(seed)
    train_key, loader_key,loader_key2, sample_key = jr.split(key, 4)
    data_shape = data.shape[1:]
    
    if data_y is not None:
        checkpoint_dir="checkpoints/Bridge"
        
    if filename is None:
        filename= f'Model_{seed}_{lr}'

    opt = optax.adabelief(lr)
    # Optax will update the floating-point JAX arrays in the model.
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    total_value = 0
    total_size = 0
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Generate a unique filename for this checkpoint
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_file = os.path.join(checkpoint_dir, f"{filename}.eqx")

    
    for step, data, data_y in zip(
    range(num_steps), 
    dataloader(data, batch_size, key=loader_key), 
    dataloader(data_y, batch_size, key=loader_key2) if data_y is not None else [None]*num_steps
):
        
        value, model, train_key, opt_state = make_step(
            model, sde, data, data_y, t1, train_key, opt_state, opt.update
        )
        total_value += value.item()
        total_size += 1
        
        if ((step % print_every) == 0) or step == num_steps - 1:
            print(f"Step={step} Loss={total_value / total_size}")
            total_value = 0
            total_size = 0
            
        if ((step % sample_every) == 0) or step == num_steps - 1:

            vmap_key = jr.split(sample_key, sample_size)
            sample_fn = ft.partial(sde.backward_sample, model, data_shape)
            sample = jax.vmap(sample_fn)(vmap_key, y=data_y[:sample_size])
            sample=jnp.array(sample)
            sample= sample[-1, :, :]
            sample_key = jr.split(sample_key, 1)[0]
            
            plot_points( sample)

        if step % checkpoint_every == 0:
            eqx.tree_serialise_leaves(checkpoint_file, model)