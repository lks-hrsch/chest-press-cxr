FROM docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# uv (and uvx) binaries from astral's image
COPY --from=ghcr.io/astral-sh/uv:python3.12-trixie-slim /usr/local/bin/uv /usr/local/bin/uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    build-essential \
    git \
    git-lfs \
    openssh-client \
    sudo && \
    rm -rf /var/lib/apt/lists/*

# passwordless sudo for the ubuntu user (uid 1000, created by the cuda base image)
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user && chmod 0440 /etc/sudoers.d/user

# Pre-create the HF + uv cache mount-points so they stay ubuntu-owned after the
# bind/named-volume mounts attach (a named volume inherits the target dir's ownership
# on first creation — without this, uid 1000 can't write it under --userns=keep-id).
RUN mkdir -p /home/ubuntu/.cache/huggingface /home/ubuntu/.cache/uv && chown -R ubuntu:ubuntu /home/ubuntu/.cache

USER ubuntu

# Put user-local tool bins (e.g. `graphify`, `ruff` from `uv tool install`) on PATH
ENV PATH="/home/ubuntu/.local/bin:/home/ubuntu/bin:${PATH}"

# Cache mount keeps tool deps off the network when earlier layers change.
# uid/gid required so the `ubuntu` user can write the cache under --userns=keep-id.
RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    uv tool install graphifyy[all] && \
    uv tool install ruff