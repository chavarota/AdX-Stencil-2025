from typing import Dict, Set, List
import numpy as np

from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator

from my_agent import MyNDaysNCampaignsAgent

# --------------------------------------------------------------------------- #
#                              Helper utilities                               #
# --------------------------------------------------------------------------- #

def safe_mean(values: List[float], default: float = 0.0) -> float:
    """np.mean that never returns NaN on an empty list."""
    return np.mean(values) if values else default


# --------------------------------------------------------------------------- #
#                              Cost‑tracker agent                             #
# --------------------------------------------------------------------------- #

class CostTracker(NDaysNCampaignsAgent):
    """
    Heuristic bidder that
      • learns req‑CPM for every segment it purchases
      • cascades prices from supersets / subsets when data are missing
      • allocates impressions greedily from the cheapest known subsets
      • tracks remaining population supply in `self._supply`
    """

    # ----------------------------- initialisation -------------------------- #

    def __init__(self) -> None:
        super().__init__()
        self.name: str = "A&M_new"

        # learned clearing prices per segment
        self._segment_price: Dict[MarketSegment, float] = {}

        # per‑campaign running stats
        self._campaign_stats: Dict[int, Dict[str, float]] = {}

        # population supply (leaf segments only)
        self._supply: Dict[MarketSegment, int] = {
            MarketSegment(("Male",   "Young", "LowIncome")): 1836,
            MarketSegment(("Male",   "Young", "HighIncome")):  517,
            MarketSegment(("Male",   "Old",   "LowIncome")): 1795,
            MarketSegment(("Male",   "Old",   "HighIncome")):  808,
            MarketSegment(("Female", "Young", "LowIncome")): 1980,
            MarketSegment(("Female", "Young", "HighIncome")):  256,
            MarketSegment(("Female", "Old",   "LowIncome")): 2401,
            MarketSegment(("Female", "Old",   "HighIncome")):  407,
        }

        self._initial_supply = self._supply.copy()

    # --------------------------- lifecycle hooks --------------------------- #

    def on_new_game(self) -> None:
        self._segment_price.clear()
        self._campaign_stats.clear()

    # ------------------------- inventory helpers --------------------------- #

    def _segment_supply(self, seg: MarketSegment) -> int:
        """Return remaining supply for seg (aggregate from leaves if needed)."""
        if seg in self._supply:
            return max(0, self._supply[seg])
        return sum(v for s, v in self._supply.items() if seg <= s and v > 0)

    def _debit_supply(self, seg: MarketSegment, qty: int) -> None:
        """Reduce supply for seg by `qty` impressions (affects only the leaf)."""
        current = self._supply.get(seg, self._segment_supply(seg))
        self._supply[seg] = max(0, current - qty)

    # --------------------------- price learning --------------------------- #

    def _record_price(self, campaign: Campaign) -> None:
        """Update req‑CPM for campaign’s target segment based on new data."""
        stats = self._campaign_stats.setdefault(
            campaign.uid,
            dict(cumulative_reach=self.get_cumulative_reach(campaign),
                 cumulative_cost=self.get_cumulative_cost(campaign),
                 prev_bid=0.0),
        )

        # deltas since last tick
        new_reach = self.get_cumulative_reach(campaign)
        new_cost  = self.get_cumulative_cost(campaign)
        delta_r   = new_reach - stats["cumulative_reach"]
        delta_c   = new_cost  - stats["cumulative_cost"]

        stats["cumulative_reach"] = new_reach
        stats["cumulative_cost"]  = new_cost

        if delta_r > 0 and delta_c > 0:
            avg_price = delta_c / delta_r
            blended   = (avg_price + stats["prev_bid"]) / 2
            # self._segment_price[campaign.target_segment] = max(avg_price + 0.2,
            #                                                    blended)
            self._segment_price[campaign.target_segment] = max(avg_price + 0.1, blended)
                                                                
        else:
            # self._segment_price[campaign.target_segment] = stats["prev_bid"] + 0.2
            self._segment_price[campaign.target_segment] = stats["prev_bid"] + 0.1

    # --------------------------- bidding helpers -------------------------- #

    def _known_subsets(self, seg: MarketSegment) -> List[MarketSegment]:
        """Return 3‑attribute subsets of `seg` for which we know a price."""
        return [s for s in self._segment_price if s < seg and len(s) == 3]

    def _cheapest_superset_price(self, seg: MarketSegment) -> float | None:
        """Cheapest price among supersets that contain `seg`."""
        prices = [self._segment_price[s] for s in self._segment_price if seg < s]
        return min(prices) if prices else None

    # -------------------------- required interface ------------------------ #

    def get_campaign_bids(self, auctions: Set[Campaign]) -> Dict[Campaign, float]:
        # Simple “95 % of reach” reserve for every auction
        return {c: 0.95 * c.reach for c in auctions}

    def get_ad_bids(self) -> Set[BidBundle]:
        self._supply = self._initial_supply.copy()
        # 1) update clearing prices from today’s results
        for c in self.get_active_campaigns():
            self._record_price(c)

        # 2) sort campaigns by urgency (impressions / remaining days)
        today = self.get_current_day()

        def urgency(c: Campaign) -> float:
            return c.reach / max(1, c.end_day - today)

        campaigns = sorted(self.get_active_campaigns(), key=urgency, reverse=True)

        # 3) build bid bundles
        bundles: Set[BidBundle] = set()

        for c in campaigns:
            # ------------------------------------------------------------ #
            # 1.  Basic campaign state                                      #
            # ------------------------------------------------------------ #
            r_left = max(0, c.reach  - self.get_cumulative_reach(c))
            b_left = max(0, c.budget - self.get_cumulative_cost(c))
            if r_left == 0 or b_left == 0:
                continue

            tgt      = c.target_segment
            tgt_cpm  = self._segment_price.get(tgt, None)          # may be None
            bundle   = set()                                       # bids we will submit
            spent    = 0                                           # $ already committed

            # ------------------------------------------------------------ #
            # 2.  Try to satisfy campaign with CHEAP LEAF SUPERSETS first   #
            # ------------------------------------------------------------ #
            leaf_supers = [
                s for s in self._supply
                if tgt < s and len(s) == 3
                and s in self._segment_price
                and self._segment_supply(s)      > 0
            ]
            leaf_supers.sort(key=lambda s: self._segment_price[s])          # cheapest first

            leaf_r_left = r_left
            leaf_cost   = 0
            leaf_bids   = set()

            for leaf in leaf_supers:
                if leaf_r_left == 0:
                    break
                take   = min(self._segment_supply(leaf), leaf_r_left)
                price  = self._segment_price[leaf]
                cost   = take * price
                leaf_r_left -= take
                leaf_cost   += cost
                leaf_bids.add(Bid(self, leaf, price, cost))

            # ------------------------------------------------------------ #
            # 3.  Compute DIRECT‑TARGET option cost                         #
            # ------------------------------------------------------------ #
            direct_cost = float("inf")
            if tgt_cpm is not None:
                direct_cost = r_left * tgt_cpm

            # ------------------------------------------------------------ #
            # 4.  Choose cheaper plan                                       #
            # ------------------------------------------------------------ #
            if leaf_r_left == 0 and leaf_cost < direct_cost:
                # Cheaper *and* enough supply → just use leaves
                bundle = leaf_bids
                spent  = leaf_cost
                self._campaign_stats[c.uid]["prev_bid"] = safe_mean(
                    [self._segment_price[s] for s in leaf_supers[:1]],     # cheapest CPM
                    default=self._campaign_stats[c.uid]["prev_bid"]
                )
            else:
                # Either leaves were too expensive or insufficient supply
                # ‑ add any leaf bids we already accepted
                bundle = leaf_bids
                spent  = leaf_cost
                # buy remaining impressions on tgt (or fallback)
                remaining_r = r_left - (r_left - leaf_r_left)
                remaining_b = b_left - spent

                # choose CPM for tgt (known, superset fallback, or heuristic)
                tgt_cpm = tgt_cpm or self._cheapest_superset_price(tgt) \
                        or 0.9 * (remaining_b / max(1, remaining_r))

                bundle.add(Bid(self, tgt, tgt_cpm, remaining_b))
                self._campaign_stats[c.uid]["prev_bid"] = tgt_cpm

            # ------------------------------------------------------------ #
            # 5.  Finalise bids and debit inventory                         #
            # ------------------------------------------------------------ #
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
    test_agents = [CostTracker()] + [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)