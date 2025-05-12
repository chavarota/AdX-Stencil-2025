from typing import Dict, Set, List
import numpy as np

from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator

from my_agent import MyNDaysNCampaignsAgent

def safe_mean(values: List[float], default: float = 0.0) -> float:
    """np.mean that never returns NaN on an empty list."""
    values = list(values)
    return np.mean(values) if values else default

class CostTracker2(NDaysNCampaignsAgent):
    def __init__(self) -> None:
        super().__init__()
        self.name = "A&M_new"

        # learned clearing prices per segment
        self._segment_price = {}

        # per‑campaign running stats
        self._campaign_stats = {}

        # population supply (leaf segments only)
        self._supply = {
            MarketSegment(("Male", "Young", "LowIncome")): 1836,
            MarketSegment(("Male", "Young", "HighIncome")):  517,
            MarketSegment(("Male", "Old", "LowIncome")): 1795,
            MarketSegment(("Male", "Old", "HighIncome")):  808,
            MarketSegment(("Female", "Young", "LowIncome")): 1980,
            MarketSegment(("Female", "Young", "HighIncome")):  256,
            MarketSegment(("Female", "Old", "LowIncome")): 2401,
            MarketSegment(("Female", "Old", "HighIncome")):  407,
        }

        self._initial_supply = self._supply.copy()

    def _segment_supply(self, seg: MarketSegment) -> int:
        """Return remaining supply for seg (aggregate from leaves if needed)."""
        if seg in self._supply:
            return max(0, self._supply[seg])
        return sum(v for s, v in self._supply.items() if seg <= s and v > 0)
    
    def _expected_fulfilment_cost(self, c: Campaign) -> float:
        """Reach * best CPM we currently know for the target segment."""
        seg = c.target_segment
        cpm = self._segment_price.get(
            seg,
            self._cheapest_superset_price(seg)            # fallback 1: broader seg
            or safe_mean(self._segment_price.values(), 1) # fallback 2: global avg
        )
        return c.reach * cpm
    
    def _debit_supply(self, seg: MarketSegment, qty: int) -> None:
        """Reduce supply for seg by `qty` impressions (affects only the leaf)."""
        current = self._supply.get(seg, self._segment_supply(seg))
        self._supply[seg] = max(0, current - qty)

    def on_new_game(self) -> None:
        self._segment_price.clear()
        self._campaign_stats.clear()

    def _record_price(self, campaign: Campaign) -> None:
        stats = self._campaign_stats.setdefault(
            campaign.uid,
            dict(cumulative_reach=self.get_cumulative_reach(campaign),
                 cumulative_cost=self.get_cumulative_cost(campaign),
                 prev_bid=0.0),
        )

        new_reach = self.get_cumulative_reach(campaign)
        new_cost = self.get_cumulative_cost(campaign)
        delta_r = new_reach - stats["cumulative_reach"]
        delta_c = new_cost - stats["cumulative_cost"]

        stats["cumulative_reach"] = new_reach
        stats["cumulative_cost"]  = new_cost

        # TODO: be smarter about this

        # if our bids were successful, update the clearing price
        if delta_r > 0 and delta_c > 0:
            avg_price = delta_c / delta_r
            blended = (avg_price + stats["prev_bid"]) / 2
            self._segment_price[campaign.target_segment] = max(avg_price, blended)                                            
        else:
            self._segment_price[campaign.target_segment] = stats["prev_bid"] + 0.2
    
    def _leaf_segments_with_price(self, segment: MarketSegment) -> List[MarketSegment]:
        """Return all leaf segments under the given segment whose price we know."""
        return [s for s in self._segment_price if s < segment and len(s) == 3] # CHANGED

    def _cheapest_superset_price(self, segment: MarketSegment) -> float | None:
        """Cheapest price among supersets that contain a given segment."""
        prices = [self._segment_price[s] for s in self._segment_price if segment < s and len(s) == 3] # CHANGED
        return min(prices) if prices else None
    
    def get_campaign_bids(self, auctions: Set[Campaign]) -> Dict[Campaign, float]:
        # Simple “95 % of reach” reserve for every auction
        # TODO: be smarter about this
        return {
            c: 1.4 * self._expected_fulfilment_cost(c)
            for c in auctions
        }
    
    def get_ad_bids(self) -> Set[BidBundle]:
        self._supply = self._initial_supply.copy()

        for c in self.get_active_campaigns():
            self._record_price(c) #TODO: be smarter about this
        
        today = self.get_current_day()

        def urgency(c: Campaign) -> float:
            return c.reach / max(1, c.end_day - today)

        campaigns = sorted(self.get_active_campaigns(), key=urgency, reverse=True)

        bundles = set()

        for c in campaigns:
            r_left = max(0, c.reach  - self.get_cumulative_reach(c))
            b_left = max(0, c.budget - self.get_cumulative_cost(c))
            if r_left == 0 or b_left == 0:
                continue

            # if c.end_day == today:
            #     price = 1.0
            #     bid_obj = Bid(
            #         bidder=self,
            #         auction_item=c.target_segment,
            #         bid_per_item=price,   # server minimum
            #         bid_limit=r_left,      # server minimum
            #     )
            #     b = BidBundle(c.uid, r_left, {bid_obj})
            #     bundles.add(b)
            #     continue


            tgt = c.target_segment #may not be a leaf segment
            tgt_price = self._segment_price.get(tgt) + 0.2

            bundle = set()
            spent = 0

            # try to satisfy the campaign with leaf segments
            leaf_subsets = []
            for s in self._supply:
                if s.issubset(tgt) and (len(s) == 3) and (s in self._segment_price) and (self._supply[s] > 0):
                    leaf_subsets.append(s)
            
            leaf_subsets.sort(key=lambda s: self._segment_price[s])

            leaf_reach_left = r_left
            leaf_cost = 0
            leaf_bids = set()

            for leaf in leaf_subsets:
                if leaf_reach_left == 0:
                    break

                num_items_to_take = min(self._supply[leaf], leaf_reach_left)
                price = self._segment_price[leaf]
                total_cost = num_items_to_take * price
                
                leaf_reach_left -= num_items_to_take
                leaf_cost += total_cost
                leaf_bids.add(Bid(self, leaf, price, total_cost))
            
            direct_cost = float("inf")
            if tgt_price is not None:
                direct_cost = r_left * tgt_price
            
            if leaf_reach_left == 0 and leaf_cost < direct_cost:
                bundle = leaf_bids
                spent = leaf_cost
                self._campaign_stats[c.uid]["prev_bid"] = safe_mean(
                    [self._segment_price[s] for s in leaf_subsets[:1]],
                    default=self._campaign_stats[c.uid]["prev_bid"]
                )
            else:
                bundle = leaf_bids
                spent = leaf_cost
                remaining_r = r_left - (r_left - leaf_reach_left)
                remaining_b = b_left - spent

                tgt_price = tgt_price or self._cheapest_superset_price(tgt) or 0.9 * remaining_b / remaining_r

                bundle.add(Bid(self, tgt, tgt_price, remaining_b))
                self._campaign_stats[c.uid]["prev_bid"] = tgt_price

            for bid in bundle:
                if len(bid.item) == 3:
                    self._debit_supply(
                        bid.item,
                        int(bid.bid_limit / bid.bid_per_item)
                    )

            bundles.add(BidBundle(c.uid, b_left, bundle))

        return bundles
    
if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [CostTracker2()] + [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)