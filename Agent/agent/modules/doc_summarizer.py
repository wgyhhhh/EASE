from agent.common.report import Report
from agent.common.modeling import Model
from agent.prompts.prompts import SummarizeDocPrompt


class DocSummarizer:
    """Summarizes a given, finished fact-checking document. The resulting summary is
    equivalent to the justification of the verdict."""

    def __init__(self, llm: Model):
        self.llm = llm

    def summarize(self, doc: Report) -> str:
        summarize_doc_prompt = SummarizeDocPrompt(doc)
        summary = self.llm.generate(summarize_doc_prompt)
        return summary
