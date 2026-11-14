"""Text normalization. Fix these settings per language BEFORE running experiments and
write them in ANALYSIS_PLAN.md. Report results with and without diacritics when the
language uses them to carry meaning (for example tone marks)."""
import re
import unicodedata


def normalize(text, lowercase=True, strip_punct=True, strip_diacritics=False):
    text = unicodedata.normalize("NFC", text)
    if lowercase:
        text = text.lower()
    if strip_diacritics:
        decomposed = unicodedata.normalize("NFD", text)
        text = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
        text = unicodedata.normalize("NFC", text)
    if strip_punct:
        text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return re.sub(r"\s+", " ", text).strip()
