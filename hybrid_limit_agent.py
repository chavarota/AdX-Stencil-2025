import numpy as np
from typing import Dict, Set
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import Bid, BidBundle, Campaign

class HybridLimitAgent(NDaysNCampaignsAgent):
    def __init__(self, name: str = "A&M", eps: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.name = name
        self.beta_grid = np.linspace(0.5, 1.5, 11)
        self.Q = np.zeros((3, 11), dtype=np.float32)
        self.eps, self.a = eps, alpha
        self.prev_profit = 0.0
        self.choice = {}

    def on_new_game(self): 
        self.prev_profit = 0.0
        self.choice.clear()

    # hardcoded for now
    def get_campaign_bids(self, auctions: Set[Campaign]) -> Dict[Campaign, float]:
        return {c: 0.95 * c.reach for c in auctions}

    def get_ad_bids(self) -> Set[BidBundle]:
        day = self.get_current_day()
        bundles = set()

        # learn from yesterday’s profit
        reward = self.profit - self.prev_profit
        self.prev_profit = self.profit

        for b, i in self.choice.values():
            self.Q[b, i] += self.a * (reward - self.Q[b, i])
        self.choice.clear()         # we’ll fill it with today’s picks

        # bid for active campaigns
        for c in self.get_active_campaigns():
            R_left = max(0, c.reach  - self.get_cumulative_reach(c))
            B_left = max(0, c.budget - self.get_cumulative_cost(c))
            if R_left == 0 or B_left == 0:
                continue

            days_left = max(0, c.end_day - day)
            b = min(2, days_left // 3) # urgency bin: 0/1/2

            if np.random.rand() < self.eps:
                idx = np.random.randint(11) # explore
            else:
                idx = int(np.argmax(self.Q[b])) # exploit

            β = self.beta_grid[idx]
            price = β * (B_left / R_left)
            limit = B_left                           

            bundles.add(BidBundle(c.uid, limit, {Bid(self, c.target_segment, price, limit)}))

            # remember choice for tomorrow’s learning update
            self.choice[c.uid] = (b, idx)

        return bundles
