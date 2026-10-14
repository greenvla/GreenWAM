# SPDX-License-Identifier: OpenMDW-1.1
"""Copies of the preprocessing the evaluation entry points need, without the training data pipeline.

The policy servers and the world-model rollout originally compiled their samples through the training data pipeline,
which is not part of this release. The functions they call are copied here with their bodies unchanged; branches that
are dead on the serving path are removed and fail loudly if reached. Each serving entry point was checked bit for bit
against the original pipeline.
"""
