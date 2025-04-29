# agents/heuristics/base_heuristic.py
import random
from typing import Dict, Set, List

from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import Bid, BidBundle, Campaign


class SimpleHeuristicMixin:
    """Utility methods shared by all toy heuristics."""

    # ----------  campaign-auction helpers ----------
    def _valid_campaign_bid(self, c: Campaign, raw_bid: float) -> float:
        """Clip bid into the legal window [0.1 R, R]."""
        lo, hi = 0.1 * c.reach, c.reach
        return max(lo, min(hi, raw_bid))

    # ----------  impression-auction helpers ----------
    def _remaining(self, c: Campaign):
        """Return remaining reach and budget for campaign *c*."""
        rem_reach = max(0, c.reach  - self.get_cumulative_reach(c))
        rem_budget = max(0, c.budget - self.get_cumulative_cost(c))
        return rem_reach, rem_budget

    def _one_segment_bidbundle(
        self, c: Campaign, bid_per_imp: float, daily_limit: float
    ) -> BidBundle:
        """Create a BidBundle that bids only in c.target_segment."""
        seg: Set[str] = c.target_segment
        bid_obj = Bid(
            bidder=self,
            auction_item=seg,
            bid_per_item=max(0.1, bid_per_imp),   # server minimum
            bid_limit=max(1.0, daily_limit),      # server minimum
        )
        return BidBundle(c.uid, daily_limit, {bid_obj})
    
    def on_new_game(self) -> None:      # satisfies the abstract requirement
        pass
