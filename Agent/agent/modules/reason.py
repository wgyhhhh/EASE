from agent.common.report import Report
from agent.common.modeling import Model
from agent.prompts.prompts import AnswerQuestionWithReasoning

class ReasonGeneration:

    def __init__(self, llm: Model):
        self.llm = llm

    def generation(self, doc: Report) -> str:
        reason_generation_prompt = AnswerQuestionWithReasoning(doc)
        summary = self.llm.generate(reason_generation_prompt)
        return summary
