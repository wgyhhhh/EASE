from agent.common.report import Report
from agent.common.modeling import Model
from agent.prompts.prompts import AnswerQuestionWithSentiment

class SentimentGeneration:

    def __init__(self, llm: Model):
        self.llm = llm

    def generation(self, doc: Report) -> str:
        reason_generation_prompt = AnswerQuestionWithSentiment(doc)
        summary = self.llm.generate(reason_generation_prompt)
        return summary