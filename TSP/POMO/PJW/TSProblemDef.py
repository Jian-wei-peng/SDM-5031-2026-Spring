import math
import random

import torch


SUPPORTED_DISTRIBUTIONS = (
    "uniform",
    "clustered",
    "anisotropic",
    "grid_jitter",
    "ring",
    "line_biased",
    "mixed_density",
    "integer",
)


def get_random_problems(batch_size, problem_size, distribution="uniform"):
    """Generate synthetic TSP instances for distribution-aware fine-tuning."""
    if distribution == "uniform":
        problems = _uniform(batch_size, problem_size)
    elif distribution == "clustered":
        problems = _clustered(batch_size, problem_size)
    elif distribution == "anisotropic":
        problems = _anisotropic(batch_size, problem_size)
    elif distribution == "grid_jitter":
        problems = _grid_jitter(batch_size, problem_size)
    elif distribution == "ring":
        problems = _ring(batch_size, problem_size)
    elif distribution == "line_biased":
        problems = _line_biased(batch_size, problem_size)
    elif distribution == "mixed_density":
        problems = _mixed_density(batch_size, problem_size)
    elif distribution == "integer":
        problems = _integer_like(batch_size, problem_size)
    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Supported: {', '.join(SUPPORTED_DISTRIBUTIONS)}"
        )
    return problems


def _uniform(batch_size, problem_size):
    return torch.rand(size=(batch_size, problem_size, 2))


def _clustered(batch_size, problem_size):
    centers_per_instance = random.randint(3, 8)
    centers = torch.rand(size=(batch_size, centers_per_instance, 2))
    assignments = torch.randint(
        centers_per_instance,
        size=(batch_size, problem_size),
        device=centers.device,
    )
    batch_idx = torch.arange(batch_size, device=centers.device)[:, None].expand(batch_size, problem_size)
    selected_centers = centers[batch_idx, assignments]
    std = 0.035 + 0.065 * torch.rand(size=(batch_size, 1, 1))
    problems = selected_centers + std * torch.randn(size=(batch_size, problem_size, 2))
    return problems.clamp(0.0, 1.0)


def _anisotropic(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2))
    stretch = 0.25 + 0.45 * torch.rand(size=(batch_size, 1, 1))
    problems[:, :, [1]] = problems[:, :, [1]] * stretch

    theta = 2.0 * math.pi * torch.rand(size=(batch_size, 1, 1))
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    centered = problems - 0.5
    x = centered[:, :, [0]]
    y = centered[:, :, [1]]
    rotated = torch.cat((cos_t * x - sin_t * y, sin_t * x + cos_t * y), dim=2)
    return _normalize_to_unit_square(rotated)


def _grid_jitter(batch_size, problem_size):
    grid_size = int(math.ceil(math.sqrt(problem_size)))
    base = torch.floor(torch.rand(size=(batch_size, problem_size, 2)) * grid_size)
    jitter = 0.08 * torch.randn(size=(batch_size, problem_size, 2))
    problems = (base + 0.5 + jitter) / grid_size
    return problems.clamp(0.0, 1.0)


def _ring(batch_size, problem_size):
    angle = 2.0 * math.pi * torch.rand(size=(batch_size, problem_size, 1))
    radius = 0.30 + 0.18 * torch.rand(size=(batch_size, problem_size, 1))
    radius = radius + 0.035 * torch.randn(size=(batch_size, problem_size, 1))
    x = 0.5 + radius * torch.cos(angle)
    y = 0.5 + radius * torch.sin(angle)
    problems = torch.cat((x, y), dim=2)
    return problems.clamp(0.0, 1.0)


def _line_biased(batch_size, problem_size):
    t = torch.rand(size=(batch_size, problem_size, 1))
    slope = -1.5 + 3.0 * torch.rand(size=(batch_size, 1, 1))
    intercept = -0.25 + 0.5 * torch.rand(size=(batch_size, 1, 1))
    noise = 0.045 * torch.randn(size=(batch_size, problem_size, 2))
    x = t
    y = slope * (t - 0.5) + 0.5 + intercept
    problems = torch.cat((x, y), dim=2) + noise
    return _normalize_to_unit_square(problems)


def _mixed_density(batch_size, problem_size):
    uniform_count = max(1, problem_size // 2)
    clustered_count = problem_size - uniform_count
    uniform_part = _uniform(batch_size, uniform_count)
    clustered_part = _clustered(batch_size, clustered_count)
    problems = torch.cat((uniform_part, clustered_part), dim=1)
    perm = torch.rand(size=(batch_size, problem_size)).argsort(dim=1)
    gather_idx = perm[:, :, None].expand(batch_size, problem_size, 2)
    return problems.gather(dim=1, index=gather_idx)


def _integer_like(batch_size, problem_size):
    coords = torch.randint(
        0,
        10000,
        size=(batch_size, problem_size, 2),
        device=torch.empty(0).device,
    ).float()
    return _normalize_to_unit_square(coords)


def _normalize_to_unit_square(problems):
    xy_max = torch.max(problems, dim=1, keepdim=True).values
    xy_min = torch.min(problems, dim=1, keepdim=True).values
    span = torch.max(xy_max - xy_min, dim=2, keepdim=True).values
    span[span == 0] = 1
    return (problems - xy_min) / span


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems
