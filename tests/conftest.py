from __future__ import annotations

import pytest

from chronoise import GeneratorConfig


@pytest.fixture
def small_cfg() -> GeneratorConfig:
    """Small config so the full 6 x 3 x 2 x n_seeds grid stays cheap."""
    return GeneratorConfig(T=512, n_seeds=2)
