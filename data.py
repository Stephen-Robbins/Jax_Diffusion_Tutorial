import jax.numpy as jnp
import numpy as np
from jax import random

def generate_happy_face(num_samples, spread=0.01, key=None, face_prob=0.7, eye_prob=0.1):
    if key is None:
        key = random.PRNGKey(0)

    data_points = []

    # Probabilistically assign samples to each feature
    face_samples = int(num_samples * face_prob)
    eye_samples = int(num_samples * eye_prob) // 2  # Divide by 2 for two eyes
    smile_samples = num_samples - face_samples - 2 * eye_samples

    # Face outline - circle
    angles = np.linspace(0, 2 * np.pi, face_samples)
    data_points.extend(zip(np.cos(angles), np.sin(angles)))

    # Eyes - smaller circles
    eye_radius = 0.1
    angles = np.linspace(0, 2 * np.pi, eye_samples)
    data_points.extend(zip(-0.35 + eye_radius * np.cos(angles), 0.45 + eye_radius * np.sin(angles)))  # Left eye
    data_points.extend(zip(0.35 + eye_radius * np.cos(angles), 0.45 + eye_radius * np.sin(angles)))   # Right eye

    # Smile - semi-circle
    angles = np.linspace(0.75 * np.pi, 0.25 * np.pi, smile_samples)
    data_points.extend(zip(-0.85 * np.cos(angles), -0.85 * np.sin(angles) + 0.35))

    # Convert data_points to numpy array and shuffle
    data = np.array(data_points)
    np.random.shuffle(data)

    # Add Gaussian noise
    key, subkey = random.split(key)
    data += random.normal(subkey, data.shape) * spread

    return jnp.array(data * 6, dtype=jnp.float32)

def generate_mixture_gaussians(num_samples, centers=6, spread=.5, radius=5.0, center_probs=None, key=None):
    if key is None:
        key = random.PRNGKey(0)

    if center_probs is None:
        # Uniform distribution if no specific probabilities provided
        center_probs = np.ones(centers) / centers

    data = []
    total_samples = 0

    angles = np.linspace(0, 2 * np.pi, centers, endpoint=False)

    for i, (angle, prob) in enumerate(zip(angles, center_probs)):
        samples_per_center = int(num_samples * prob)
        total_samples += samples_per_center

        # Adjust for last center to cover all samples due to rounding
        if i == len(angles) - 1:
            samples_per_center += (num_samples - total_samples)

        center_x = radius * np.cos(angle)
        center_y = radius * np.sin(angle)

        key, subkey = random.split(key)
        samples = random.normal(subkey, (samples_per_center, 2)) * spread + np.array([center_x, center_y])
        data.append(samples)

    data = np.vstack(data)
    np.random.shuffle(data)

    return jnp.array(data, dtype=jnp.float32)