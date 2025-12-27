from typing import Collection

import pyparsing as pp

from agent.common.action import (Action)
from agent.evidence_retrieval.tools import IMAGE_ACTIONS
from agent.common import logger, Report, Model
from agent.prompts.prompts import PlanPrompt


class Planner:
    """Chooses the next actions to perform based on the current knowledge as contained
    in the FC document."""

    def __init__(self,
                 valid_actions: Collection[type[Action]],
                 llm: Model,
                 extra_rules: str):
        self.valid_actions = valid_actions
        self.llm = llm
        self.max_attempts = 5
        self.extra_rules = extra_rules

    def get_available_actions(self, doc: Report):
        available_actions = []
        completed_actions = set(type(a) for a in doc.get_all_actions())

        if doc.claim.has_image():  # TODO: enable multiple image actions for multiple images
            available_actions += [a for a in IMAGE_ACTIONS if a not in completed_actions]

        # TODO: finish this method

        return available_actions

    def plan_next_actions(self, doc: Report, all_actions=False) -> (list[Action], str):
        prompt = PlanPrompt(doc, self.valid_actions, self.extra_rules, all_actions)
        n_attempts = 0

        while n_attempts < self.max_attempts:
            n_attempts += 1

            response = self.llm.generate(prompt)
            if response is None:
                logger.warning("No new actions were found.")
                return [], ""

            actions = response["actions"]
            reasoning = response["reasoning"]

            # Remove actions that have been performed before
            performed_actions = doc.get_all_actions()
            actions = [action for action in actions if action not in performed_actions]

            if len(actions) > 0:
                return actions, reasoning
            else:
                performed_actions_str = ", ".join(str(obj) for obj in performed_actions)
                logger.warning(f'No new actions were found in this response:\n{response["response"]} and performed actions: {performed_actions_str}')
                return [], ""
