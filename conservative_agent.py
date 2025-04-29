# agents/heuristics/conservative_agent.py
import random
from typing import Dict, Set
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import BidBundle, Campaign
from simple_heuristic import SimpleHeuristicMixin


class ConservativeAgent(SimpleHeuristicMixin, NDaysNCampaignsAgent):
    """
    • Low-balls campaign bids at 20–35 % of reach.
    • In impression auctions bids 80 % of fair price, caps spend at 30 % of remaining budget.
    """

    def __init__(self, name="Conservative"):
        super().__init__()
        self.name = name

    # campaign auction
    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}
        for c in campaigns_for_auction:
            raw = c.reach * random.uniform(0.20, 0.35)
            bids[c] = self._valid_campaign_bid(c, raw)
        return bids

    # impression auction
    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        for c in self.get_active_campaigns():
            rem_R, rem_B = self._remaining(c)
            if rem_R == 0:
                continue
            price = 0.8 * (rem_B / rem_R)
            daily_limit = 0.3 * rem_B
            bundles.add(self._one_segment_bidbundle(c, price, daily_limit))
        return bundles
