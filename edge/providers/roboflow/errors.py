from __future__ import annotations


class ProviderError(RuntimeError):
    """Base provider error."""


class CapabilityUnavailableError(ProviderError):
    """Raised when a provider capability is unavailable."""


class InvalidProviderResponseError(ProviderError):
    """Raised when a provider returns invalid internal data."""


class RetryableProviderError(ProviderError):
    """Raised for retryable provider failures."""
