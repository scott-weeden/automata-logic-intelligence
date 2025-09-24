"""Pytest template for intelligent systems assignments."""

from __future__ import annotations

import pytest


@pytest.fixture
def setup_environment():  # pragma: no cover - template fixture
    """Provide shared resources for a group of tests."""
    # TODO: create domain objects used across tests.
    return {}


def test_placeholder(setup_environment):  # pragma: no cover - template
    """Replace this with a real test that validates your solution."""
    # TODO: assert on meaningful behaviour.
    assert setup_environment == {}
