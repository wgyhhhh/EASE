from .common import SearchResults, Source, WebSource
from .google_search import Google
from .knowledge_base import KnowledgeBase
from .search_platform import SearchPlatform
from .wiki_dump import WikiDump

_PLATFORMS = [Google, KnowledgeBase, WikiDump]

PLATFORMS = {platform.name: platform for platform in _PLATFORMS}
