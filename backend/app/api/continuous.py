"""Compatibility import surface for the continuous post-training API.

The implementation lives in :mod:`continuous_post_training` so the router and
its contracts remain isolated from the existing coordinator bootstrap.
"""

from .continuous_post_training import *  # noqa: F403
