from agent.common import logger, Model, Content, Claim
from agent.utils.console import light_blue

class ClaimExtractor:
    def __init__(self, llm: Model,):
        self.llm = llm
        self.max_retries = 3

    def extract_claims(self, content: Content) -> list[Claim]:
        logger.log(f"Extracting claims from {content.__repr__()}")

        claims = [Claim(content.data, context=content)]

        logger.log("Extracted claims:")
        for claim in claims:
            logger.log(light_blue(f"{claim}"))

        # Add claims to content object
        content.claims = claims

        return claims
