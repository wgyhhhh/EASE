from enum import Enum


class Label(Enum):
    real = "real"
    NEI = "not enough information"
    fake = "fake"
    CONFLICTING = "conflicting evidence"
    CHERRY_PICKING = "cherry-picking"
    REFUSED_TO_ANSWER = "error: refused to answer"
    OUT_OF_CONTEXT = "out of context"
    MISCAPTIONED = "miscaptioned"


DEFAULT_LABEL_DEFINITIONS = {
    Label.real: "The knowledge from the fact-check supports or at least strongly implies the Claim. "
                     "Mere plausibility is not enough for this decision.",
    Label.NEI: "The fact-check does not contain sufficient information to come to a conclusion. For example, "
               "there is substantial lack of evidence. In this case, state which information exactly "
               "is missing. In particular, if no RESULTS or sources are available, pick this decision.",
    Label.fake: "The knowledge from the fact-check clearly refutes the Claim. The mere absence or lack of "
                   "supporting evidence is not enough reason for being refuted (argument from ignorance)."
}
