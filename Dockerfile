FROM nvcr.io/nvidia/jax:24.04-py3
WORKDIR /workspace

# Install packages more explicitly
RUN pip install --no-cache-dir numpy>=1.22 mujoco>=3.0.0 mujoco-mjx

# Verify installation
RUN python -c "import mujoco; print('MuJoCo installed:', mujoco.__version__)"