# agents/heuristics/aggressive_agent.py
import random
from typing import Dict, Set
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import BidBundle, Campaign
from simple_heuristic import SimpleHeuristicMixin


class AggressiveAgent(SimpleHeuristicMixin, NDaysNCampaignsAgent):
    """
    • Always bids high (90–100 % of reach) to grab campaigns.
    • In impression auctions it bids 1.5× the “remaining-budget / remaining-reach”
      fair price and spends up to 60 % of the remaining budget per day.
    """

    def __init__(self, name="Aggressive"):
        super().__init__()
        self.name = name

    # campaign auction
    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}
        for c in campaigns_for_auction:
            raw = c.reach * random.uniform(0.9, 1.0)
            bids[c] = self._valid_campaign_bid(c, raw)
        return bids

    # impression auction
    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        for c in self.get_active_campaigns():
            rem_R, rem_B = self._remaining(c)
            if rem_R == 0:
                continue
            price = 1.5 * (rem_B / rem_R)
            daily_limit = 0.6 * rem_B
            bundles.add(self._one_segment_bidbundle(c, price, daily_limit))
        return bundles
