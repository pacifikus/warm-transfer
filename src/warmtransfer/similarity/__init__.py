"""Content similarity cold->warm (optional; may be delegated to the user).

Populated in Phase 1+ (cosine, kNN graph). This is the assembly point.
"""

from __future__ import annotations

from warmtransfer.similarity.content import content_similarity

__all__ = ["content_similarity"]
