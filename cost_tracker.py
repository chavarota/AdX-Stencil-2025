from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
from typing import Set, Dict
import numpy as np
from my_agent import MyNDaysNCampaignsAgent

class CostTracker(NDaysNCampaignsAgent):
    def __init__(self):
        super().__init__()
        self.name = "A&M_new"
        self.segment_2_req_bid = {}
        self.campaign_2_stats = {}
        self.dists = {
            MarketSegment(("Male", "Young", "LowIncome")): 1836,
            MarketSegment(("Male", "Young", "HighIncome")): 517,
            MarketSegment(("Male", "Old", "LowIncome")): 1795,
            MarketSegment(("Male", "Old", "HighIncome")): 808,
            MarketSegment(("Female", "Young", "LowIncome")): 1980,
            MarketSegment(("Female", "Young", "HighIncome")): 256,
            MarketSegment(("Female", "Old", "LowIncome")): 2401,
            MarketSegment(("Female", "Old", "HighIncome")): 407,
            # MarketSegment(("Female", "Young")): 2236,
            # MarketSegment(("Female", "Old")): 2808,
            # MarketSegment(("Male", "Old")): 2603,
            # MarketSegment(("Male", "Young")): 2353,
            # MarketSegment(("Female", "LowIncome")): 4381,
            # MarketSegment(("Female", "HighIncome")): 663,
            # MarketSegment(("Male", "LowIncome")): 3631,
            # MarketSegment(("Male", "HighIncome")): 1325,
            # MarketSegment(("Young", "LowIncome")): 3816,
            # MarketSegment(("Young", "HighIncome")): 773,
            # MarketSegment(("Old", "LowIncome")): 4196,
            # MarketSegment(("Old", "HighIncome")): 1215,
        }

    def on_new_game(self):
        self.segment_2_req_bid = {}
        self.campaign_2_stats = {}

    # hardcoded for now
    def get_campaign_bids(self, auctions: Set[Campaign]) -> Dict[Campaign, float]:
        return {c: 0.95 * c.reach for c in auctions}

    def get_ad_bids(self) -> Set[BidBundle]:
        # update the price for each segment
        for c in self.get_active_campaigns():
            if c.uid not in self.campaign_2_stats:
                self.campaign_2_stats[c.uid] = {
                    "cumulative_reach": self.get_cumulative_reach(c),
                    "cumulative_cost": self.get_cumulative_cost(c),
                    "prev_bid": 0,
                }
            else:
                previous_reach = self.campaign_2_stats[c.uid]["cumulative_reach"]
                previous_cost = self.campaign_2_stats[c.uid]["cumulative_cost"]

                current_reach = self.get_cumulative_reach(c)
                current_cost = self.get_cumulative_cost(c)

                self.campaign_2_stats[c.uid]["cumulative_reach"] = current_reach
                self.campaign_2_stats[c.uid]["cumulative_cost"] = current_cost

                delta_reach = current_reach - previous_reach
                delta_cost = current_cost - previous_cost

                if delta_reach > 0 and delta_cost > 0:
                    # Calculate the average price for the segment
                    avg_price = delta_cost / delta_reach
                    self.segment_2_req_bid[c.target_segment] = max(
                        avg_price + 0.2,
                        (avg_price + self.campaign_2_stats[c.uid]["prev_bid"]) / 2      # simple average of two scalars
                    )
                else:
                    self.segment_2_req_bid[c.target_segment] = self.campaign_2_stats[c.uid]["prev_bid"] + 0.2

        #day = self.get_current_day()
        bundles = set()

        active_campaigns = self.get_active_campaigns()
        # sort by urgency
        today = self.get_current_day()

        def urgency(c: Campaign) -> float:
            # days left, but never smaller than 1
            days_left = max(1, c.end_day - today)
            # campaigns that have already started get urgency > 1
            return c.reach / days_left          # or any other monotone score

        active_campaigns_sorted = sorted(active_campaigns, key=urgency, reverse=True)


        # bid for active campaigns
        for c in active_campaigns_sorted:
            R_left = max(0, c.reach  - self.get_cumulative_reach(c))
            B_left = max(0, c.budget - self.get_cumulative_cost(c))

            if R_left == 0 or B_left == 0:
                continue

            target_segment = c.target_segment
            if target_segment in self.segment_2_req_bid:
                price = self.segment_2_req_bid[target_segment]
                bundles.add(BidBundle(c.uid, B_left, {Bid(self, target_segment, price, B_left)}))
                self.campaign_2_stats[c.uid]["prev_bid"] = price
            else:
                contains_subsets = False
                subsets = []
                for segment in self.segment_2_req_bid.keys():
                    if segment.issubset(target_segment) and target_segment != segment and len(segment) == 3:
                        contains_subsets = True
                        subsets.append(segment)
                # TODO: the subsets list needs to only return absolute subsets that contain no subsets themselves.
                # this means that we also need to initialize the segment_2_req_bid with prices, potentially from supersets.
                # another TODO is to sort campaigns by urgency, so that we can bid for the most urgent ones first.
                # a third TODO is that we need to check if we have already sent out bids for the segment in a previous campaign on the same day

                if not contains_subsets:
                    if target_segment in self.segment_2_req_bid:
                        price = self.segment_2_req_bid[target_segment]
                    else:
                        # set the price equal to the price of the nearest superset present in the segment_2_req_bid
                        superset = None
                        superset_price = float("inf")
                        for segment in self.segment_2_req_bid.keys():
                            if target_segment.issubset(segment) and segment != target_segment and len(segment) == 3:
                                superset = segment
                                superset_price = self.segment_2_req_bid[segment]
                                break
                        if superset is not None:
                            price = superset_price
                        else:
                            price = 0.9 * (B_left / R_left)

                    bundles.add(BidBundle(c.uid, B_left, {Bid(self, target_segment, price, B_left)}))
                    self.dists[target_segment] -= B_left / price
                    self.campaign_2_stats[c.uid]["prev_bid"] = price
                else:
                    dynamic_r_left = R_left
                    bid_lists = []
                    prices = []
                    while dynamic_r_left > 0 and len(subsets) > 0:
                        cheapest_subset = None
                        cheapest_subset_price = float("inf")
                        for subset in subsets:
                            if subset in self.segment_2_req_bid:
                                subset_price = self.segment_2_req_bid[subset]
                                if subset_price < cheapest_subset_price:
                                    cheapest_subset = subset
                                    cheapest_subset_price = subset_price
                            else:
                                #set the price equal to the price of the nearest superset present in the segment_2_req_bid
                                superset = None
                                superset_price = float("inf")
                                for segment in self.segment_2_req_bid.keys():
                                    if subset.issubset(segment) and segment != subset and len(segment) == 3:
                                        superset = segment
                                        superset_price = self.segment_2_req_bid[segment]
                                        break
                                if superset is not None and superset_price < cheapest_subset_price:
                                    cheapest_subset = subset
                                    cheapest_subset_price = superset_price
                        if cheapest_subset is None:
                            break
                        num_people_avail_from_segment = self.dists[cheapest_subset]
                        dynamic_r_left -= num_people_avail_from_segment
                        num_people_required = min(num_people_avail_from_segment, dynamic_r_left)
                        self.dists[cheapest_subset] -= num_people_required
                        num_dollars_required = num_people_required * cheapest_subset_price
                        subsets.remove(cheapest_subset)
                        prices.append(cheapest_subset_price)
                        bid_lists.append(Bid(self, cheapest_subset, cheapest_subset_price, num_dollars_required))
                    avg_price = np.average(prices)
                    self.campaign_2_stats[c.uid]["prev_bid"] = avg_price
                    bundles.add(BidBundle(c.uid, B_left, bid_lists))
                    

            # price = 1.1 * (B_left / R_left)

            # bundles.add(BidBundle(c.uid, B_left, {Bid(self, c.target_segment, price, B_left)}))

        return bundles

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [CostTracker()] + [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)