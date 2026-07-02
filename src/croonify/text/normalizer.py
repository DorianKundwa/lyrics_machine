"""Lyrics text normalization for the Croonify alignment pipeline.

Normalization converts raw lyrics text into clean, phone-level word sequences
that can be compared against ASR outputs.  Key operations:

1. Contraction expansion  (``don't`` → ``do not``)
2. Punctuation removal
3. Lower-casing
4. Line-structure preservation (empty strings mark line boundaries)

All normalizers are pure-Python and have no external dependencies beyond the
standard library, making them safe to import in any environment.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contraction dictionary — 60+ entries covering common English contractions
# ---------------------------------------------------------------------------

CONTRACTIONS: dict[str, str] = {
    # Negative contractions
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "cannot": "cannot",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "needn't": "need not",
    "daren't": "dare not",
    "mayn't": "may not",
    "oughtn't": "ought not",
    # To-be contractions
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "who's": "who is",
    "what's": "what is",
    "where's": "where is",
    "how's": "how is",
    # Have contractions
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "could've": "could have",
    "should've": "should have",
    "would've": "would have",
    "might've": "might have",
    "must've": "must have",
    # Will contractions
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "we'll": "we will",
    "they'll": "they will",
    "that'll": "that will",
    "there'll": "there will",
    # Would/Had contractions
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "it'd": "it would",
    "we'd": "we would",
    "they'd": "they would",
    # Going-to
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "hafta": "have to",
    "oughta": "ought to",
    "kinda": "kind of",
    "sorta": "sort of",
    "lotta": "lot of",
    # Informal
    "ain't": "is not",
    "y'all": "you all",
    "let's": "let us",
    "o'clock": "o clock",
    "'cause": "because",
    "c'mon": "come on",
    "ma'am": "madam",
    "o'er": "over",
    "ne'er": "never",
    "e'er": "ever",
    "nothin'": "nothing",
    "somethin'": "something",
    "everythin'": "everything",
    "'em": "them",
    "ol'": "old",
}

# Build a case-insensitive lookup by lower-casing all keys
_CONTRACTIONS_LOWER: dict[str, str] = {k.lower(): v for k, v in CONTRACTIONS.items()}

# Pre-compile a regex that matches any contraction (longest-match, with apostrophe variants)
_APOSTROPHE_PATTERN = re.compile(r"['\u2018\u2019\u02BC]")  # plain + typographic + modifier
_WORD_TOKENIZE_PATTERN = re.compile(r"\b[\w''\u2018\u2019\u02BC]+\b", re.UNICODE)
_PUNCTUATION_STRIP_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)


# ---------------------------------------------------------------------------
# LyricsNormalizer
# ---------------------------------------------------------------------------

class LyricsNormalizer:
    """Normalize raw lyrics text for forced alignment.

    Performs contraction expansion, punctuation removal, lower-casing, and
    line-structure-preserving tokenization.  The normalizer is stateless — all
    methods can be called without creating an instance.

    Examples
    --------
    >>> norm = LyricsNormalizer()
    >>> norm.normalize_word("can't")
    'cannot'
    >>> norm.normalize_lyrics("Don't stop\\nBelievin'")
    ['do', 'not', 'stop', '', 'believing']
    """

    def __init__(self) -> None:
        pass  # stateless

    # ------------------------------------------------------------------
    # Word-level
    # ------------------------------------------------------------------

    def normalize_word(
        self,
        word: str,
        expand_contractions: bool = True,
    ) -> str:
        """Normalize a single word token.

        Steps:
        1. Normalize unicode (NFC).
        2. Unify apostrophe variants to ASCII ``'``.
        3. Optionally expand known contractions.
        4. Strip leading/trailing punctuation.
        5. Lower-case.

        Parameters
        ----------
        word:
            Raw word token (may include surrounding punctuation).
        expand_contractions:
            If ``True``, expand known contractions (e.g. ``don't → do not``).
            Note: expanded forms may contain spaces.

        Returns
        -------
        str
            Normalized, lower-cased word (or expanded phrase).
        """
        # Unicode normalization
        word = unicodedata.normalize("NFC", word)
        # Unify apostrophe variants
        word = _APOSTROPHE_PATTERN.sub("'", word)
        # Strip leading/trailing punctuation (but keep interior apostrophes)
        word = word.strip(".,!?;:\"()[]{}—–-…")
        word_lower = word.lower()

        if expand_contractions:
            expanded = _CONTRACTIONS_LOWER.get(word_lower)
            if expanded is not None:
                return expanded  # may contain spaces, e.g. "do not"

        return word_lower

    # ------------------------------------------------------------------
    # Line-level
    # ------------------------------------------------------------------

    def normalize_line(
        self,
        line: str,
        expand_contractions: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = True,
    ) -> List[str]:
        """Normalize a single line of lyrics to a list of word strings.

        Parameters
        ----------
        line:
            A single line of raw lyrics text.
        expand_contractions:
            Expand contractions to their full forms.
        remove_punctuation:
            Remove punctuation characters.
        lowercase:
            Lower-case all output tokens.

        Returns
        -------
        list[str]
            List of clean word tokens (may include multiple tokens from a
            single expanded contraction).  Empty list for blank lines.
        """
        if not line.strip():
            return []

        # Tokenize: extract word-like tokens including apostrophes
        raw_tokens = _WORD_TOKENIZE_PATTERN.findall(line)
        result: List[str] = []

        for token in raw_tokens:
            normalized = self.normalize_word(token, expand_contractions=expand_contractions)
            if not normalized:
                continue
            # An expanded contraction may contain spaces — split and extend
            if " " in normalized:
                sub_words = normalized.split()
                if not lowercase:
                    sub_words = [w for w in sub_words]
                result.extend(sub_words)
            else:
                if not lowercase:
                    normalized = normalized  # already processed above
                result.append(normalized)

        if remove_punctuation:
            # Secondary pass: strip any residual non-word characters
            result = [re.sub(r"[^\w]", "", w) for w in result]
            result = [w for w in result if w]

        return result

    # ------------------------------------------------------------------
    # Full-document
    # ------------------------------------------------------------------

    def normalize_lyrics(
        self,
        text: str,
        expand_contractions: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = True,
    ) -> List[str]:
        """Normalize a full lyrics document to a flat token list.

        Line breaks are preserved as empty-string sentinel values (``''``)
        in the output list.  This allows downstream consumers to reconstruct
        the original line structure.

        Parameters
        ----------
        text:
            Full lyrics document (newline-delimited lines).
        expand_contractions:
            Expand contractions.
        remove_punctuation:
            Remove punctuation.
        lowercase:
            Lower-case tokens.

        Returns
        -------
        list[str]
            Flat list of word tokens with ``''`` marking line boundaries.
            Multiple consecutive blank lines are collapsed to a single ``''``.

        Examples
        --------
        >>> norm = LyricsNormalizer()
        >>> norm.normalize_lyrics("Hello world\\nGoodnight moon")
        ['hello', 'world', '', 'goodnight', 'moon']
        """
        lines = text.splitlines()
        result: List[str] = []
        last_was_boundary = False

        for raw_line in lines:
            words = self.normalize_line(
                raw_line,
                expand_contractions=expand_contractions,
                remove_punctuation=remove_punctuation,
                lowercase=lowercase,
            )
            if not words:
                # blank / structural line → emit sentinel (collapse consecutive)
                if not last_was_boundary and result:
                    result.append("")
                    last_was_boundary = True
            else:
                result.extend(words)
                last_was_boundary = False

        # Strip trailing sentinel
        while result and result[-1] == "":
            result.pop()

        return result

    # ------------------------------------------------------------------
    # Structured access
    # ------------------------------------------------------------------

    def get_line_structure(self, text: str) -> List[List[str]]:
        """Return lyrics as a list of lines, each being a list of words.

        Unlike :meth:`normalize_lyrics`, this method does **not** insert
        sentinel empty strings — instead it returns a nested list where each
        outer element corresponds to one non-blank line.

        Parameters
        ----------
        text:
            Full lyrics document.

        Returns
        -------
        list[list[str]]
            ``[[word, ...], ...]`` — one sub-list per non-empty line.

        Examples
        --------
        >>> norm = LyricsNormalizer()
        >>> norm.get_line_structure("Hello world\\nGoodnight moon")
        [['hello', 'world'], ['goodnight', 'moon']]
        """
        lines = text.splitlines()
        structured: List[List[str]] = []
        for raw_line in lines:
            words = self.normalize_line(raw_line)
            if words:
                structured.append(words)
        return structured

    # ------------------------------------------------------------------
    # Static factory — useful for inline use
    # ------------------------------------------------------------------

    @staticmethod
    def flat_words(text: str) -> List[str]:
        """Convenience shortcut: normalize *text* and return non-empty tokens only."""
        norm = LyricsNormalizer()
        tokens = norm.normalize_lyrics(text)
        return [t for t in tokens if t]  # filter sentinels
