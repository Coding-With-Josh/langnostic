"""Whisper language names and codes. Standard library only: safe for CI and tests."""

LANGUAGE_CODES = {
    "english": "en",
    "en": "en",
    "yoruba": "yo",
    "yo": "yo",
    "hausa": "ha",
    "ha": "ha",
    "swahili": "sw",
    "sw": "sw",
    "french": "fr",
    "fr": "fr",
    "german": "de",
    "de": "de",
    "igbo": "ig",
    "ig": "ig",
}


def resolve_language(value, supported=None):
    """Map a manifest language value to the code the checkpoint expects.

    ``supported`` is the checkpoint's own set of language codes (from
    ``generation_config.lang_to_id``). When it is supplied, a language that the
    checkpoint does not expose is an error: we never fall back to auto-detect,
    because that would change the experiment being measured.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("language must be a non-empty string")

    candidate = value.strip().lower()

    if supported is not None:
        if candidate in supported:
            return candidate
        code = LANGUAGE_CODES.get(candidate)
        if code in supported:
            return code
        raise ValueError(
            f"language {value!r} is not supported by this checkpoint; "
            f"supported codes: {sorted(supported)}"
        )

    code = LANGUAGE_CODES.get(candidate)
    if code is None:
        raise ValueError(
            f"unknown language {value!r}; add it to LANGUAGE_CODES "
            "or pass a whisper language code"
        )
    return code
