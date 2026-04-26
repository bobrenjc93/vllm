# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    ConfiguredAttentionBackend,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    MambaAttentionBackendEnum,
    register_backend,
)

pytestmark = pytest.mark.skip_global_cleanup


class CustomAttentionImpl(AttentionImpl):
    """Mock custom attention implementation for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Mock forward pass."""
        pass


class CustomAttentionBackend(AttentionBackend):
    """Mock custom attention backend for testing."""

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return CustomAttentionImpl

    @staticmethod
    def get_builder_cls():
        """Mock builder class."""
        return None

    @staticmethod
    def get_required_kv_cache_layout():
        """Mock KV cache layout."""
        return None


class CustomConfiguredMetadata:
    """Mock metadata class for configured backend testing."""


class CustomConfiguredBuilder:
    """Mock builder class for configured backend testing."""


class CustomConfiguredAttentionBackend(ConfiguredAttentionBackend):
    """Mock custom attention backend using class attribute configuration."""

    name = "CONFIGURED_CUSTOM"
    impl_cls = "CustomAttentionImpl"
    builder_cls = "CustomConfiguredBuilder"
    metadata_cls = "CustomConfiguredMetadata"
    supported_head_sizes = [64, 128]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)


class CustomMambaAttentionImpl(AttentionImpl):
    """Mock custom mamba attention implementation for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Mock forward pass."""
        pass


class CustomMambaAttentionBackend(AttentionBackend):
    """Mock custom mamba attention backend for testing."""

    @staticmethod
    def get_name():
        return "CUSTOM_MAMBA"

    @staticmethod
    def get_impl_cls():
        return CustomMambaAttentionImpl

    @staticmethod
    def get_builder_cls():
        """Mock builder class."""
        return None

    @staticmethod
    def get_required_kv_cache_layout():
        """Mock KV cache layout."""
        return None


def test_custom_is_not_alias_of_any_backend():
    # Get all members of AttentionBackendEnum
    all_backends = list(AttentionBackendEnum)

    # Find any aliases of CUSTOM
    aliases = []
    for backend in all_backends:
        if backend.name != "CUSTOM" and backend is AttentionBackendEnum.CUSTOM:
            aliases.append(backend.name)

    # CUSTOM should not be an alias of any other backend
    assert len(aliases) == 0, (
        f"BUG! CUSTOM is an alias of: {', '.join(aliases)}!\n"
        f"CUSTOM.value = {repr(AttentionBackendEnum.CUSTOM.value)}\n"
        f"This happens when CUSTOM has the same value as another backend.\n"
        f"When you register to CUSTOM, you're actually registering to {aliases[0]}!\n"
        f"All backend values:\n"
        + "\n".join(f"  {b.name}: {repr(b.value)}" for b in all_backends)
    )

    # Verify CUSTOM has its own unique identity
    assert AttentionBackendEnum.CUSTOM.name == "CUSTOM", (
        f"CUSTOM.name should be 'CUSTOM', but got '{AttentionBackendEnum.CUSTOM.name}'"
    )


def test_register_custom_backend_with_class_path():
    # Register with explicit class path
    register_backend(
        backend=AttentionBackendEnum.CUSTOM,
        class_path="tests.test_attention_backend_registry.CustomAttentionBackend",
        is_mamba=False,
    )

    # Check that CUSTOM backend is registered
    assert AttentionBackendEnum.CUSTOM.is_overridden(), (
        "CUSTOM should be overridden after registration"
    )

    # Get the registered class path
    class_path = AttentionBackendEnum.CUSTOM.get_path()
    assert class_path == "tests.test_attention_backend_registry.CustomAttentionBackend"

    # Get the backend class
    backend_cls = AttentionBackendEnum.CUSTOM.get_class()
    assert backend_cls.get_name() == "CUSTOM"
    assert backend_cls.get_impl_cls() == CustomAttentionImpl


def test_configured_attention_backend_class_attributes():
    assert CustomConfiguredAttentionBackend.get_name() == "CONFIGURED_CUSTOM"
    assert CustomConfiguredAttentionBackend.get_impl_cls() is CustomAttentionImpl
    assert CustomConfiguredAttentionBackend.get_builder_cls() is CustomConfiguredBuilder
    assert (
        CustomConfiguredAttentionBackend.get_metadata_cls()
        is CustomConfiguredMetadata
    )
    assert CustomConfiguredAttentionBackend.get_supported_head_sizes() == [64, 128]
    assert CustomConfiguredAttentionBackend.supports_head_size(64)
    assert not CustomConfiguredAttentionBackend.supports_head_size(96)


def test_mamba_custom_is_not_alias_of_any_backend():
    # Get all mamba backends
    all_backends = list(MambaAttentionBackendEnum)

    # Find any aliases of CUSTOM
    aliases = []
    for backend in all_backends:
        if backend.name != "CUSTOM" and backend is MambaAttentionBackendEnum.CUSTOM:
            aliases.append(backend.name)

    # CUSTOM should not be an alias of any other backend
    assert len(aliases) == 0, (
        f"BUG! MambaAttentionBackendEnum.CUSTOM is an alias of: {', '.join(aliases)}!\n"
        f"CUSTOM.value = {repr(MambaAttentionBackendEnum.CUSTOM.value)}\n"
        f"All mamba backend values:\n"
        + "\n".join(f"  {b.name}: {repr(b.value)}" for b in all_backends)
    )


def test_register_custom_mamba_backend_with_class_path():
    # Register with explicit class path
    register_backend(
        backend=MambaAttentionBackendEnum.CUSTOM,
        class_path="tests.test_attention_backend_registry.CustomMambaAttentionBackend",
        is_mamba=True,
    )

    # Check that the backend is registered
    assert MambaAttentionBackendEnum.CUSTOM.is_overridden()

    # Get the registered class path
    class_path = MambaAttentionBackendEnum.CUSTOM.get_path()
    assert (
        class_path
        == "tests.test_attention_backend_registry.CustomMambaAttentionBackend"
    )

    # Get the backend class
    backend_cls = MambaAttentionBackendEnum.CUSTOM.get_class()
    assert backend_cls.get_name() == "CUSTOM_MAMBA"
    assert backend_cls.get_impl_cls() == CustomMambaAttentionImpl
