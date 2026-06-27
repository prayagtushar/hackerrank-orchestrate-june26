"""Tests for the versioned cache key."""

import hashlib
import main

ROW = {
    "user_id": "u1",
    "image_paths": "a.jpg",
    "user_claim": "Customer: hi",
    "claim_object": "car",
}


def test_stable_for_same_inputs():
    assert main.cache_key(ROW, "m") == main.cache_key(dict(ROW), "m")


def test_model_sensitive():
    assert main.cache_key(ROW, "m1") != main.cache_key(ROW, "m2")


def test_input_sensitive():
    other = dict(ROW, user_claim="Customer: bye")
    assert main.cache_key(ROW, "m") != main.cache_key(other, "m")


def test_includes_prompt_version_and_signature():
    legacy = hashlib.sha256(
        f"{ROW['user_id']}|{ROW['image_paths']}|{ROW['user_claim']}|"
        f"{ROW['claim_object']}|m".encode()
    ).hexdigest()
    assert main.cache_key(ROW, "m") != legacy
    assert main.PROMPT_VERSION and isinstance(main._PROMPT_SIG, str)
