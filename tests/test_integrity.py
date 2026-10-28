import hashlib

import pytest

from movie_recs.common import verify


def test_corrupted_dataset_is_rejected():
    digest = hashlib.sha256(b"trusted").hexdigest()
    verify(b"trusted", digest)
    with pytest.raises(ValueError, match="SHA-256"):
        verify(b"changed", digest)
