import re

# Standalone hesitation sounds. Deliberately narrow: "hmm"/"mm" can carry meaning.
_FILLER = re.compile(r"(?<![\w'])(?:u+m+|u+h+|e+r+m+|a+h+m*)(?![\w'])[,…]*\s*", re.IGNORECASE)
# Stutters: the same word repeated back to back ("the the the" -> "the").
_STUTTER = re.compile(r"\b([A-Za-z']+)(?:,?\s+\1\b)+", re.IGNORECASE)
_AMPM = re.compile(r"\b([ap])\.\s?m\.", re.IGNORECASE)  # the model sometimes writes "a. M."
_ABBREV = re.compile(r"\b(?:[ap]\.m|e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr)\.$", re.IGNORECASE)


def remove_fillers(text: str) -> str:
    if not text:
        return text
    out = _FILLER.sub("", text)
    out = _STUTTER.sub(r"\1", out)
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)             # "And ." -> "And."
    out = re.sub(r"([,;:])+(?=[.!?])", "", out)            # "so,." -> "so."
    out = re.sub(r"(^|[.!?]\s+)([,.;:]\s*)+", r"\1", out)  # orphan punctuation at sentence start
    out = re.sub(r"(?<=[.!?])\s+([a-z])",                  # sentence case, except after "a.m." etc.
                 lambda m: m.group(0) if _ABBREV.search(m.string, 0, m.start()) else " " + m.group(1).upper(), out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    if out[:1].islower() and text[:1].isupper():  # removed a leading "Um, so" -> "So"
        out = out[0].upper() + out[1:]
    return "" if re.fullmatch(r"[\W_]*", out) else out


# When a piece of audio ends, the model ends its text as if the speech ended: period, closing
# quote. Continuation decoding strips that and lets the model decide again with more audio.
_TRAILING = re.compile(r"[\s.!?…,;:\"'”’]+$")
_LEADING = re.compile(r"^[\s.!?…,;:\"'”’]*")


def split_tail(text: str) -> tuple[str, str]:
    """'…which is essentially.' -> ('…which is essentially', '.')"""
    m = _TRAILING.search(text)
    return (text[:m.start()], text[m.start():]) if m else (text, "")


def split_lead(text: str) -> tuple[str, str]:
    """'." And then' -> ('."', 'And then')"""
    lead = _LEADING.match(text).group(0)
    return lead.strip(), text[len(lead):]


def clean(text: str, fillers: bool) -> str:
    text = _AMPM.sub(lambda m: m.group(1).lower() + ".m.", (text or "").strip())
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return remove_fillers(text) if fillers else text


def repair_quotes(text: str, max_len: int = 600) -> str:
    """Speech has no audible quotation marks, so the model guesses where a quote ends, and
    sometimes never closes one, swallowing the narration that follows. Keep well-formed
    quotes; drop marks that are unmatched or enclose an implausibly long passage."""
    out = []
    for para in text.split("\n\n"):
        drop, open_at = set(), None
        for m in re.finditer('"', para):
            i = m.start()
            before = para[i - 1] if i else " "
            after = para[i + 1] if i + 1 < len(para) else " "
            if (before.isspace() or before in "([—") and not after.isspace():
                if open_at is not None:
                    drop.add(open_at)  # a new quote opens before the last one closed
                open_at = i
            elif not before.isspace() and (after.isspace() or after in ".,!?;:)—"):
                if open_at is None:
                    drop.add(i)
                elif i - open_at > max_len:
                    drop.update((open_at, i))
                open_at = None
            else:
                drop.add(i)  # a stray mark
        if open_at is not None:
            drop.add(open_at)
        para = "".join(c for k, c in enumerate(para) if k not in drop)
        out.append(re.sub(r" {2,}", " ", para).replace(" ,", ","))
    return "\n\n".join(out)


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
