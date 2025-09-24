"""Search agent template for coursework assignments.

Use this file as a starting point when implementing custom search agents.
Populate the ``solve`` method with the algorithm of your choice and feel free
to extend the helper functions.  The provided structure mirrors the API that
our automated tests expect (see :mod:`tests.test_search`).
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from src.search import Node, SearchAgent, SearchProblem


class CustomSearchAgent(SearchAgent):
    """Replace the placeholder logic with your own search procedure."""

    def solve(self, problem: SearchProblem) -> Optional[List[str]]:
        """Return a list of actions that solves *problem*.

        This method is intentionally separate from :meth:`search` so students can
        experiment without overriding engine instrumentation.  By default we
        simply delegate to :meth:`search` which you should customise below.
        """

        return self.search(problem)

    def search(self, problem: SearchProblem) -> Optional[List[str]]:  # pragma: no cover - template
        """Implement the core search algorithm here."""
        # TODO: implement your search procedure
        raise NotImplementedError("Search strategy not implemented yet")


def reconstruct_path(node: Node) -> List[str]:
    """Helper for turning a solution node into a list of actions."""

    return node.solution()
