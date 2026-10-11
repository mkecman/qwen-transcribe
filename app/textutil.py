import re

# Standalone hesitation sounds. Deliberately narrow: "hmm"/"mm" can carry meaning.
_FILLER = re.compile(r"(?<![\w'])(?:u+m+|u+h+|e+r+m+|a+h+m*)(?![\w'])[,…]*\s*", re.IGNORECASE)
# Stutters: the same word repeated back to back ("the the the" -> "the").
_STUTTER = re.compile(r"\b([A-Za-z']+)(?:,?\s+\1\b)+", re.IGNORECASE)


def remove_fillers(text: str) -> str:
    if not text:
        return text
    out = _FILLER.sub("", text)
    out = _STUTTER.sub(r"\1", out)
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)             # "And ." -> "And."
    out = re.sub(r"([,;:])+(?=[.!?])", "", out)            # "so,." -> "so."
    out = re.sub(r"(^|[.!?]\s+)([,.;:]\s*)+", r"\1", out)  # orphan punctuation at sentence start
    out = re.sub(r"(?<=[.!?])\s+([a-z])", lambda m: " " + m.group(1).upper(), out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    if out[:1].islower() and text[:1].isupper():  # removed a leading "Um, so" -> "So"
        out = out[0].upper() + out[1:]
    return "" if re.fullmatch(r"[\W_]*", out) else out


def clean(text: str, fillers: bool) -> str:
    text = (text or "").strip()
    return remove_fillers(text) if fillers else text


# At a seam between separately transcribed pieces the model ends the first piece as if the
# sentence ended there. These words almost never end a sentence...
_DANGLING = set("a an the to of in on at for with from by into about and or but so because that which "
                "who like my your our their his her its is are was were be been than as if when this these "
                "those very really actually just kind sort".split())
# ...and these are safe to lowercase when they turn out to continue the sentence.
_LOWERABLE = set("a an the and or but so then that this these those it we they you he she there here what "
                 "which who how when where why because if of to in on for with as at by from my our your "
                 "their his her its maybe just also all some like is are was were can could would should "
                 "will not no yes yeah okay ok".split())


def _seam(a: str, b: str) -> tuple[str, str]:
    first = re.match(r"([A-Z][a-z']*)\b", b)
    lowerable = bool(first) and first.group(1).lower() in _LOWERABLE
    dangling = re.search(r"(\w+)\.$", a)
    if dangling and dangling.group(1).lower() in _DANGLING and lowerable:
        return a[:-1], b[0].lower() + b[1:]      # "change to." + "The logic" -> "change to the logic"
    if re.search(r"[\w,]$", a) and lowerable:
        return a, b[0].lower() + b[1:]           # "work, and" + "Yeah, so" -> "work, and yeah, so"
    if re.search(r"[.!?][\"')\]]?$", a) and b[0].islower():
        return a, b[0].upper() + b[1:]           # keep sentence case after a real sentence end
    return a, b


def join(a: str, b: str, paragraph: bool = False) -> str:
    if not b:
        return a
    if not a:
        return b
    if paragraph:
        return a + "\n\n" + (b[0].upper() + b[1:])
    a, b = _seam(a, b)
    return a + " " + b


def auto_title(text: str, limit: int = 60) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    title = m.group(1) if m and len(m.group(1)) <= limit else text
    if len(title) > limit:
        title = title[:limit].rsplit(" ", 1)[0].rstrip(",;:-") + "…"
    return title


def stable_prefix(old: str, new: str) -> int:
    """How many leading chars of `new` agree with `old`, cut back to a word boundary."""
    n = 0
    for x, y in zip(old, new):
        if x != y:
            break
        n += 1
    if n == len(new):
        return n
    if new[n] == " " and (n == len(old) or not old[n].isalnum()):
        return n
    return max(new.rfind(" ", 0, n), 0)
