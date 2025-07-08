"""Stochastic differential equation helpers for score-based models."""

from abc import ABC, abstractmethod
import jax.random as jr
import jax.numpy as jnp

class SDE(ABC):

    @abstractmethod
    def score_loss(self, model, data, data_y, t, y=None):
        """
        Calculates the loss of a neural network approximation of a score function

        """
        raise NotImplementedError
    
  
    @abstractmethod
    def Drift(self, x, t, y=None):
        """Drift coefficient of the forward SDE."""
        raise NotImplementedError
    
    @abstractmethod
    def Diffusion(self, x, t):
        """Diffusion coefficient of the SDE."""
        raise NotImplementedError
    
    def Score(
        self,
        score,
        x: jnp.ndarray,
        t: float,
        y: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Evaluate the score model.

        Args:
            score: Callable neural network.
            x: Current sample.
            t: Time parameter.
            y: Optional conditioning data.

        Returns:
            Model output evaluated at ``(x, t)``.
        """

        return score(x, t)

    def Prior(
        self, key: jr.KeyArray, data_shape: tuple[int, ...], y: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Sample from the prior distribution."""

        return jr.normal(key, data_shape)
    
    def Backward_Drift(
        self, score, x: jnp.ndarray, t: float, y: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Drift term for the reverse-time SDE."""

        score = self.Score(score, x, t, y)

        return self.Drift(x, t, y) - (self.Diffusion(x, t) ** 2 * score)
  
    #@eqx.filter_jit  
    def backward_sample(
        self,
        score,
        data_shape: tuple[int, ...],
        key: jr.KeyArray,
        t0: float = 0.0,
        t1: float = 1.0,
        dt: float = 0.001,
        y: jnp.ndarray | None = None,
    ) -> list[jnp.ndarray]:
        """Simulate the reverse-time SDE path."""

        num_steps = int((t1 - t0) / dt)
        times = jnp.linspace(t1-0.0001, t0+0.0011, num_steps)  # Create an array of times
        key, subkey = jr.split(key)

        current_x = self.Prior(subkey, data_shape, y)  
        
        path = []  # Initialize an empty list to store the path
        for time in times:
            backward_drift = self.Backward_Drift(score, current_x, time, y)
            diffusion = self.Diffusion(current_x, time)
            noise = jr.normal(key, current_x.shape) * jnp.sqrt(dt)

            # Euler-Maruyama method for SDE integration, with negative time step for backward process
            current_x += backward_drift * (-dt) + diffusion * noise
            key, _ = jr.split(key)

            path.append(current_x.copy())  # Store the current state in the path

        return path
    
        

  
class VPSDE(SDE):
    def __init__(self,  bmin, bmax,  T=1.0):
        self.bmin = bmin
        self.bmax = bmax
        self.T=T
       
    def score_loss(self, model, data, data_y, t, key):
        mean, std=self.p(data, t)
        weight=1/self.B(t)
        noise = jr.normal(key, data.shape)
        x = mean + std * noise
        pred = model(x, t)
        return weight * jnp.mean((std*pred + noise) ** 2)

    def B(self, t):
        """Variance schedule."""

        return self.bmin + t * (self.bmax - self.bmin)

    def alpha(self, t):
        """Compute the mean decay."""

        x = self.bmin * t + ((self.bmax - self.bmin) * t**2) / 2
        a = jnp.exp(-x / 2)
        return a

    def p(self, x, t):
        a = self.alpha(t)
        mu = x * a
        std = (1-a**2)**0.5
        return mu, std
    
    def Drift(self, x,  t, y=None):
        """
        SDE drift

        """
        return (-self.B(t)/2)*x

    def Diffusion(self, x, t, y=None):
        """
        SDE Diffusion

        """
        return jnp.full(x.shape, jnp.sqrt(self.B(t)))
    
class BridgeVPSDE(SDE):
    def __init__(self,  bmin=0.1, bmax=1,  T=1.0):
        self.bmin = bmin
        self.bmax = bmax
        self.T=T

    def score_loss(self, model, data, data_y, t, key):
        mean, std=self.p(data, data_y, t)
        weight=1/(t+0.1)
        noise = jr.normal(key, data.shape)
        x = mean + std * noise
        x_and_y = jnp.concatenate([x, data_y], axis=0)
        pred = model(x_and_y, t)
        return weight * jnp.mean((std*pred + noise ) ** 2)

    def B(self, t):
        b = self.bmin + ((self.bmax - self.bmin) / 2) * (1 - jnp.cos(2 * jnp.pi * t))
        return b

    def alpha(self, t):
        integral = self.bmin * t + ((self.bmax - self.bmin) / 2) * (t - jnp.sin(2 * jnp.pi * t) / (2 * jnp.pi))
        a = jnp.exp(-integral / 2)
        return a
    
    def sigma(self, t):
        std = ((1-self.alpha(t)**2)**0.5)
        return std

    def SNR(self, t):
        return (self.alpha(t)**2/self.sigma(t)**2)
    
    def p(self, x, y, t, T=1):
        mu = y * (self.SNR(T) / self.SNR(t)) * (self.alpha(t) / self.alpha(T)) + self.alpha(t) * x * (1 - self.SNR(T) / self.SNR(t))
        std = self.sigma(t) * jnp.sqrt(1. - (self.SNR(T) / self.SNR(t)))
        return mu, std
    
    def h(self, x, y, t, T=1):
        
        score = ((self.alpha(t)/self.alpha(T))*y-x) / (self.sigma(t)**2*(self.SNR(t)/self.SNR(T)-1))
        return score
    
    def Drift(self, x, t, y):
        """
        SDE drift

        """
        
        return (-self.B(t)/2)*x+self.B(t)*self.h(x, y, t)

    def Diffusion(self, x, t):
        """
        SDE Diffusion

        """
        return jnp.sqrt(self.B(t))
    
    def Score(self, score, x, t, y=None):
        """
        Samples from the backward sde

        """
        x_and_y = jnp.concatenate([x, y], axis=0)
        score=score(x_and_y, t)
        return score
    
    def Prior(self, key, data_shape, y=None):
        """
        Samples from the backward sde

        """
        return y


    


    
