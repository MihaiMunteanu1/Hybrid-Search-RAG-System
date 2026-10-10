"""The grounded prompt (numbered sources in) and the parsing of the model's reply.

Plain text handling with no model involved, so the rules an answer depends on
(language, refusal, valid citations) can be tested without a server.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.retrieval.search import Hit

# The model is told to reply with exactly this marker when the sources lack the answer.
# A fixed token is reliable to detect; a refusal phrased in prose is not.
NOT_FOUND = "NOT_FOUND"

LANGUAGE_NAMES = {"ro": "Romanian", "en": "English"}
NOT_FOUND_MESSAGES = {
    "ro": "Nu am găsit această informație în documente.",
    "en": "I could not find this information in the documents.",
}

# Rules 4 and 5 stop the model from writing refusals in prose with citations attached,
# which would be indistinguishable from an answer.
SYSTEM_PROMPT = """You answer questions about a collection of documents, using only the \
numbered sources you are given.

Rules:
1. Use only facts that the sources state. Never add outside knowledge.
2. After each claim, cite the source it comes from, like [2]. Cite only the sources that \
state that claim directly, not every source you read.
3. If the sources do not contain the answer, reply with exactly {not_found} and nothing else.
4. Never write in your own words that the sources lack some information. Either answer from \
the sources, or reply {not_found}.
5. If the sources answer only part of the question, answer that part and leave the rest out.
6. Answer in {language}.
7. Be concise: a few sentences at most."""

# ------------------------------------------------------------------ language

RO_STOPWORDS = frozenset("""a al ale am ar are as au ca care ce cel cea cei cele cand când
cat cât cate câte cine cu cum da dar de din dupa după e ea el este eu fi fie fost in în inca
încă intre între la le lui mai ma mă mi ne nici nu o ori pe pentru pot poate prin sa să se
si și sunt sau spre tot toate un una unde unei unui va vor""".split())
EN_STOPWORDS = frozenset("""a about all an and any are as at be been but by can could did do
does for from has have how i if in into is it its may me my no not of on or our should so
than that the their them then there these they this to was we were what when where which
who why will with would you your""".split())
RO_LETTERS = re.compile(r"[ăâîșțşţ]", re.IGNORECASE)
RO_ENDINGS = re.compile(r"(ul|ului|lor|ile|ele|ea|ii|ție|tie|eaza|ează|esc)$")
EN_ENDINGS = re.compile(r"(ing|ed|tion|ly|ness|ment|ies|ship)$")


def detect_language(question: str) -> str | None:
    """"ro", "en", or None when the question gives no clear signal.

    A question is only a few words, so besides function words this also uses
    Romanian letters and typical word endings. When undecided, the prompt asks for
    "the same language as the question" rather than risk forcing the wrong one.
    """
    if RO_LETTERS.search(question):
        return "ro"
    words = re.findall(r"[a-z]+", question.lower())
    long_words = [w for w in words if len(w) > 4]
    ro = sum(w in RO_STOPWORDS for w in words) + 0.5 * sum(bool(RO_ENDINGS.search(w)) for w in long_words)
    en = sum(w in EN_STOPWORDS for w in words) + 0.5 * sum(bool(EN_ENDINGS.search(w)) for w in long_words)
    if ro == en:
        return None
    return "ro" if ro > en else "en"


# ---------------------------------------------------------------- the prompt


def format_source(number: int, hit: Hit) -> str:
    """One source block: "[n] document — heading path (p. X)", then the chunk text."""
    chunk = hit.chunk
    where = chunk["source"]
    if chunk.get("heading_path"):
        where += f" — {chunk['heading_path']}"
    if chunk.get("page"):
        where += f" (p. {chunk['page']})"
    return f"[{number}] {where}\n{hit.text}"


def build_messages(question: str, hits: list[Hit]) -> tuple[list[dict], str | None]:
    """Chat messages for a question and its sources, plus the detected language."""
    language = detect_language(question)
    system = SYSTEM_PROMPT.format(
        not_found=NOT_FOUND,
        language=LANGUAGE_NAMES[language] if language else "the same language as the question",
    )
    sources = "\n\n".join(format_source(i, hit) for i, hit in enumerate(hits, start=1))
    user = f"Sources:\n\n{sources}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], language


# Sent once as a follow-up when a reply has no citation. Small models often answer
# correctly but forget to cite. The way out to NOT_FOUND is spelled out so that a
# refusal written in prose does not come back as an answer with a citation.
CITE_REQUEST = """Your reply has no citations. If your reply says that the sources do not \
contain the answer, reply with exactly {not_found} and nothing else. Otherwise rewrite it \
so that every claim is followed by the number of the source that states it, like [2], \
keeping only the claims that a source states."""

# -------------------------------------------------------------- the answer

# [1]   [1, 2]   [1][3]   [ 2 ]
CITATION = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")

# A refusal written in prose despite rule 4, with citations attached later:
#   "Nu există informații în sursele furnizate care să răspundă ... [3]"
# It is recognised by its first sentence, which says that the sources lack something and
# cites nothing yet. An answer that only qualifies itself ("..., dar nu specifică dacă")
# states its fact first, and a negative answer ("Nu există taxă [2].") cites its source,
# so neither is caught. On the stored answers the rule caught 7 replies, all of them
# prose refusals (judged wrong or missed), and no correct or partial answer.
PROSE_NEGATION = re.compile(
    r"\b(nu (exist[ăa]|se (poate|afl[ăa]|g[ăa]se[șş]te|men[țţ]ioneaz[ăa]|specific[ăa]|preciz[eă]az[ăa])"
    r"|con[țţ]in|specific[ăa]|men[țţ]ioneaz[ăa]|preciz[eă]az[ăa]|ofer[ăa]|furnizeaz[ăa]|indic[ăa])"
    r"|not (contain|specify|mention|provide|state|include)|no information|does not|do not)",
    re.IGNORECASE)
PROSE_SOURCES = re.compile(
    r"\b(surs|document|informa[țţ]i|furnizat|sources?\b|documents?\b|information\b|context\b)",
    re.IGNORECASE)
FIRST_SENTENCE = re.compile(r"^.*?(?:[.!?](?=\s)|$)", re.DOTALL)


def is_prose_refusal(text: str) -> bool:
    first = FIRST_SENTENCE.match(text.strip()).group(0)
    return bool(PROSE_NEGATION.search(first) and PROSE_SOURCES.search(first)
                and not CITATION.search(first))


@dataclass
class Answer:
    question: str
    text: str                    # what the user sees
    found: bool
    language: str | None
    sources: list[Hit]           # the numbered sources shown to the model, in order
    cited: list[int] = field(default_factory=list)              # valid, in first-use order
    invalid_citations: list[int] = field(default_factory=list)  # numbers with no source
    refusal: str | None = None   # "no_sources", "model" (the marker), "no_citations", "prose"
    raw: str = ""                # the model's reply as written
    first_raw: str | None = None  # the first reply, if citations had to be requested
    timings: dict = field(default_factory=dict)

    @property
    def cited_sources(self) -> list[Hit]:
        return [self.sources[n - 1] for n in self.cited]


def parse_answer(question: str, raw: str, sources: list[Hit], language: str | None) -> Answer:
    """Turn the model's reply into an Answer.

    - A reply containing the NOT_FOUND marker anywhere is a refusal, even with text
      around it.
    - Citation numbers with no matching source are kept apart as invalid (invented
      references), not silently dropped.
    - A reply that cites no source is treated as a refusal: in practice such replies
      are refusals written in prose.
    - A reply that opens by saying the sources lack the answer is a refusal too, even
      with citations further on (see is_prose_refusal).
    """
    text = raw.strip()
    cited, invalid = [], []
    for group in CITATION.findall(text):
        for number in (int(n) for n in re.split(r"\s*[,;]\s*", group)):
            target = cited if 1 <= number <= len(sources) else invalid
            if number not in target:
                target.append(number)

    refusal = None
    if not text or re.search(rf"\b{NOT_FOUND}\b", text, re.IGNORECASE):
        refusal = "model"
    elif not cited:
        refusal = "no_citations"
    elif is_prose_refusal(text):
        refusal = "prose"
    if refusal:
        return not_found(question, sources, language, refusal, raw=raw,
                         invalid_citations=invalid)
    return Answer(question=question, text=text, found=True, language=language,
                  sources=sources, cited=cited, invalid_citations=invalid, raw=raw)


def not_found(question: str, sources: list[Hit], language: str | None, refusal: str,
              **extra) -> Answer:
    """The "not found" answer, in the question's language (both languages if unknown)."""
    message = (NOT_FOUND_MESSAGES[language] if language
               else " / ".join(NOT_FOUND_MESSAGES.values()))
    return Answer(question=question, found=False, language=language, sources=sources,
                  text=message, refusal=refusal, **extra)
