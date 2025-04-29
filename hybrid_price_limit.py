# """
# hybrid_price_limit.py
# ─────────────────────
# One file = one idea:  use a tiny RL module to pick
#     β  (price multiplier)         in [0.5, 2.0]
#     λ  (fraction of budget to spend today) in [0, 1]
# for every campaign we hold.

# We treat ONE *10-day* tournament as ONE RL step
# (sparse reward but simplest integration).
# """

# # ---------------------------------------------------------------------------
# # IMPORTS   (standard first, then course framework, then RL library)
# # ---------------------------------------------------------------------------
# import random, numpy as np, gymnasium as gym, pathlib, sys
# from typing import Dict, Set, List

# # Course-provided modules
# from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
# from agt_server.agents.utils.adx.structures import Bid, BidBundle, Campaign
# from agt_server.local_games.adx_arena        import AdXGameSimulator
# from agt_server.agents.test_agents.adx.tier1.my_agent import (
#     Tier1NDaysNCampaignsAgent,
# )
# from agt_server.agents.utils.adx.states import CampaignBidderState
# from agt_server.local_games.adx_arena import calculate_effective_reach


# # RL library (Stable-Baselines3)
# from stable_baselines3 import SAC

# # ------------------------------------------------------------------
# def simulate_single_day(arena: AdXGameSimulator, day: int):
#     """
#     Execute the body of the staff 'for day in …' loop for ONE day.
#     We pasted those lines almost verbatim, but removed the outer loops.
#     """
#     # --- update agents with current day ---------------------------------
#     for agent in arena.agents:
#         agent.current_day = day

#     # --- (1) generate campaigns for *tomorrow* and ask agents for bids ---
#     if day + 1 < arena.num_days + 1:
#         new_campaigns = [arena.generate_campaign(start_day=day + 1)
#                          for _ in range(arena.campaigns_per_day)]
#         new_campaigns = [c for c in new_campaigns if c.end_day <= arena.num_days]
#         agent_bids = {a: a.get_campaign_bids(new_campaigns) for a in arena.agents}
#     else:
#         new_campaigns, agent_bids = [], {}

#     # --- (2) solicit ad bids & run ad auctions --------------------------
#     ad_bids = []
#     for agent in arena.agents:
#         ad_bids.extend(agent.get_ad_bids())
#     users = arena.generate_auction_items(10_000)
#     arena.run_ad_auctions(ad_bids, users, day)

#     # --- (3) update quality-scores, profits, etc. -----------------------
#     for agent in arena.agents:
#         state = arena.states[agent]
#         todays_profit = 0.0
#         new_qs_count, new_qs_val = 0, 0.0

#         for camp in state.campaigns.values():
#             if camp.start_day <= day <= camp.end_day and day == camp.end_day:
#                 imps   = state.impressions[camp.uid]
#                 cost   = state.spend[camp.uid]
#                 reach  = calculate_effective_reach(imps, camp.reach)
#                 todays_profit += reach * state.budgets[camp.uid] - cost
#                 new_qs_count  += 1
#                 new_qs_val    += reach

#         if new_qs_count:
#             new_qs_val /= new_qs_count
#             state.quality_score = ((1 - arena.α) * state.quality_score +
#                                    arena.α * new_qs_val)
#             agent.quality_score = state.quality_score

#         state.profits += todays_profit
#         agent.profit  += todays_profit

#     # --- (4) run campaign auctions & endowments -------------------------
#     arena.run_campaign_auctions(agent_bids, new_campaigns)
#     for agent in arena.agents:
#         if random.random() < min(1, agent.quality_score):
#             camp = arena.generate_campaign(start_day=day)
#             camp.budget = camp.reach
#             state = arena.states[agent]
#             state.add_campaign(camp)
#             agent.my_campaigns.add(camp)
#             arena.campaigns[camp.uid] = camp


# # ========== 1. Small helper mix-in  ========================================
# class SimpleHeuristicMixin:
#     """Utility functions so we don't repeat boiler-plate."""

#     @staticmethod
#     def _valid_campaign_bid(c: Campaign, raw: float) -> float:
#         """Clip bid into legal window [0.1*reach ,  reach]."""
#         return max(0.1 * c.reach, min(c.reach, raw))

#     # ------------------------------------------------------------------
#     def _remaining(self, c: Campaign):
#         """Return remaining reach & budget (never negative)."""
#         R_left = c.reach  - self.get_cumulative_reach(c)
#         B_left = c.budget - self.get_cumulative_cost(c)
#         return max(0, R_left), max(0, B_left)

#     # ------------------------------------------------------------------
#     def _bundle(self, c: Campaign, price: float, limit: float) -> BidBundle:
#         """Build a 1-segment BidBundle with safety minima."""
#         bid = Bid(
#             bidder       = self,
#             auction_item = c.target_segment,
#             bid_per_item = max(0.1, price),     # server minimum price
#             bid_limit    = max(1.0, limit),     # server minimum limit
#         )
#         return BidBundle(c.uid, limit, {bid})


# # ========== 2. Proxy agent that the RL wrapper controls ====================
# class _Proxy(SimpleHeuristicMixin, NDaysNCampaignsAgent):
#     """Owns campaigns.  External code sets (β,λ) BEFORE we submit ad bids."""

#     def __init__(self):
#         super().__init__()
#         self.name  = "HybridProxy"
#         self._βλ   = (1.0, 0.5)     # default knobs
#         self.focus = None           # campaign chosen today (first unfinished)

#     # ----- abstract hook (nothing to reset per game) -------------------
#     def on_new_game(self): pass

#     # ----- campaign-auction strategy: always bid 95 % of reach ----------
#     def get_campaign_bids(self, auctions: Set[Campaign]) -> Dict[Campaign, float]:
#         return {c: self._valid_campaign_bid(c, 0.95 * c.reach) for c in auctions}

#     # ----- external setter called by the Gym wrapper -------------------
#     def set_control(self, β: float, λ: float):
#         self._βλ = (β, λ)

#     # ----- impression-auction bids: RL for one campaign, heuristic rest
#     def get_ad_bids(self) -> Set[BidBundle]:
#         bundles = set()
#         self.focus = None  # will become first unfinished campaign
#         for c in self.get_active_campaigns():
#             R_left, B_left = self._remaining(c)
#             if R_left == 0 or B_left == 0:
#                 continue
#             if self.focus is None:
#                 self.focus = c                 # pick first unfinished campaign

#             # choose knobs
#             if c.uid == self.focus.uid:
#                 β, λ = self._βλ               # RL-controlled today
#                 β = 0.5 + 1.5 * β            # map [0,1] → [0.5,2.0]
#             else:
#                 β, λ = (1.5, 0.6)             # fixed heuristic

#             price = β * (B_left / R_left)
#             limit = λ * B_left
#             bundles.add(self._bundle(c, price, limit))
#         return bundles


# # ========== 3. Gym wrapper : one EPISODE = one 10-day game = one step ======
# class SingleGameEnv(gym.Env):
#     """
#     * observation  : dummy 5-vector (not used by learner, but Gym needs one)
#     * action space : Box([0,1]^2) -> β, λ
#     * reward       : cumulative profit after the 10-day game ends
#     """
#     metadata = {"render_modes": []}          # Gym bookkeeping

#     def __init__(self, opponents: List[NDaysNCampaignsAgent]):
#         self.proxy = _Proxy()
#         self.opps  = opponents
#         self.sim   = AdXGameSimulator()      # course simulator
#         self.action_space      = gym.spaces.Box(low=0.0, high=1.0,
#                                                 shape=(2,), dtype=np.float32)
#         self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
#                                                 shape=(5,), dtype=np.float32)

#     # ------------------------------------------------------------------
#     def reset(self, *, seed=None, options=None):
#         # 1) Register the exact agent *objects* that will play this tournament
#         self.sim.agents = self.opps + [self.proxy]
#         self.sim.current_day = 1

#         # 2) Create a fresh CampaignBidderState for each agent
#         self.sim.states = {}
#         for idx, agent in enumerate(self.sim.agents):
#             agent.agent_num = idx                 # what init_agents used to do
#             agent.on_new_game()                   # per-game reset hook
#             self.sim.states[agent] = CampaignBidderState(idx)

#         # 3) Empty campaign dictionary for this game
#         self.sim.campaigns = {}

#         # 4) First observation: quality score + day fraction; rest zeros
#         Q  = self.proxy.get_quality_score() or 1.0
#         obs0 = np.array([Q, 1/10, 0, 0, 0], dtype=np.float32)
#         self._profit_prev = 0.0                   # for reward diff later
#         return obs0, {}



#     # ------------------------------------------------------------------
#     def step(self, action):
#         # 1. unpack the action sent by the agent
#         β, λ = action
#         self.proxy.set_control(float(β), float(λ))   # ← apply it

#         # 2. run one simulated day
#         day = self.sim.current_day
#         simulate_single_day(self.sim, day)

#         # 3. compute reward (today’s profit increment)
#         reward = self.proxy.profit - getattr(self, "_profit_prev", 0.0)
#         self._profit_prev = self.proxy.profit

#         # 4. advance day counter, build next observation (for agent’s NEXT action)
#         self.sim.current_day += 1
#         done = self.sim.current_day > self.sim.num_days

#         Q = self.proxy.get_quality_score() or 1.0
#         day_frac = min(self.sim.current_day, 10) / 10
#         obs_next = np.array([Q, day_frac, 0, 0, 0], dtype=np.float32)

#         return obs_next, reward, done, False, {}





# # ========== 4. Competition agent (loads saved SAC model) ===================
# class HybridLimitAgent(SimpleHeuristicMixin, NDaysNCampaignsAgent):
#     """Heuristic skeleton for legality; SAC picks β,λ every morning."""

#     def __init__(self, name="HybridRL", model_path: str = "sac_price_limit.zip"):
#         super().__init__()
#         self.name = name
#         self._model_path = model_path
#         self.pi = None  # will hold SAC policy

#     # ------------------------------------------------------------------
#     def on_new_game(self):
#         """Load weights once per grading simulation."""
#         if self.pi is None:
#             try:
#                 self.pi = SAC.load(self._model_path, device="cpu")
#             except FileNotFoundError:
#                 print(f"[{self.name}] Model file not found -> fallback heuristic")
#                 self.pi = None

#     # ------------------------------------------------------------------
#     def get_campaign_bids(self, auctions: Set[Campaign]) -> Dict[Campaign, float]:
#         return {c: self._valid_campaign_bid(c, 0.95 * c.reach) for c in auctions}

#     # ------------------------------------------------------------------
#     def get_ad_bids(self) -> Set[BidBundle]:
#         bundles = set()
#         for c in self.get_active_campaigns():
#             R_left, B_left = self._remaining(c)
#             if R_left == 0 or B_left == 0:
#                 continue

#             if self.pi:
#                 # dummy obs fed to SAC; its training ignored obs anyway
#                 obs = np.array([
#                     self.get_quality_score(),
#                     self.get_current_day() / 10,
#                     R_left / c.reach,
#                     B_left / c.budget,
#                     (c.end_day - self.get_current_day()) / 3.0
#                 ], dtype=np.float32)
#                 β, λ = self.pi.predict(obs, deterministic=True)[0]
#                 β = 0.5 + 1.5 * β
#             else:  # fallback if model missing
#                 β, λ = 1.5, 0.6

#             price = β * (B_left / R_left)
#             limit = λ * B_left
#             bundles.add(self._bundle(c, price, limit))
#         return bundles


# # ========== 5. Training harness (executes only if run directly) ============
# if __name__ == "__main__":
#     """
#     $ python hybrid_price_limit.py     # trains SAC once and saves model
#     """
#     print(">> TRAINING SAC on whole-game environment (may take 1-2 h CPU)")

#     # ----- prepare environment with 9 Tier-1 opponents ----------------
#     opps = [Tier1NDaysNCampaignsAgent(name=f"TA{i}") for i in range(9)]
#     env  = SingleGameEnv(opps)

#     # ----- set up SAC (tiny network; sparse reward needs many steps) ---
#     model = SAC(
#         "MlpPolicy",
#         env,
#         batch_size=256,
#         learning_rate=3e-4,
#         gamma=0.995,
#         policy_kwargs=dict(net_arch=[128, 128]),
#         verbose=1,
#     )

#     model.learn(total_timesteps=10000)  # sparse reward -> more steps
#     model.save("sac_price_limit")
#     print(">> Saved to sac_price_limit.zip")

#     # quick sanity check
#     test_agent = HybridLimitAgent()
#     arena = AdXGameSimulator()
#     agents = [test_agent] + opps
#     arena.run_simulation(agents, num_simulations=50)
#     print("<<< done")

"""
hybrid_price_limit.py  –  simplest per-day learning agent
---------------------------------------------------------
• No Gym, no Torch, no changes to staff simulator.
• Action  : choose β ∈ {0.5, 0.6, …, 1.5}
• Context : 3-level urgency  (days_left // 3  →  0,1,2)
• Update  : ε-greedy tabular Q-learning on daily profit delta
"""

import numpy as np
from typing  import Dict, Set
from agt_server.agents.base_agents.adx_agent          import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures           import Bid, BidBundle, Campaign

# ------------- helper mix-in (keeps bids legal) -------------------
class SimpleMixin:
    def _remaining(self, c: Campaign):
        return max(0, c.reach  - self.get_cumulative_reach(c)), \
               max(0, c.budget - self.get_cumulative_cost(c))

    def _bundle(self, c: Campaign, price: float, limit: float) -> BidBundle:
        bid = Bid(self, c.target_segment, max(0.1, price), max(1.0, limit))
        return BidBundle(c.uid, max(1.0, limit), {bid})


# ==================   learning agent   =============================
class HybridLimitAgent(SimpleMixin, NDaysNCampaignsAgent):
    def __init__(self, name: str = "HybridRL", eps: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.name        = name
        self.beta_grid   = np.linspace(0.5, 1.5, 11)   # 0.5 … 1.5
        self.Q           = np.zeros((3, 11), dtype=np.float32)
        self.eps, self.a = eps, alpha
        self.prev_profit = 0.0

    # -------- every new game: reset running profit -----------------
    def on_new_game(self): 
        self.prev_profit = 0.0

    # -------- bid for new campaigns (simple heuristic) -------------
    def get_campaign_bids(self, auctions: Set[Campaign]) -> Dict[Campaign, float]:
        return {c: max(0.1*c.reach, 0.95*c.reach) for c in auctions}

    # -------- daily impression bids (learning happens here) --------
    def get_ad_bids(self) -> Set[BidBundle]:
        day          = self.get_current_day()
        bundles      = set()

        # ---- (A) learn from yesterday’s outcome -------------------
        reward       = self.profit - self.prev_profit
        self.prev_profit = self.profit
        for c in self.get_active_campaigns():
            if hasattr(c, "_bin"):
                b, i = c._bin, c._idx
                self.Q[b, i] += self.a * (reward - self.Q[b, i])

        # ---- (B) build bids for today -----------------------------
        for c in self.get_active_campaigns():
            R_left, B_left = self._remaining(c)
            if R_left == 0 or B_left == 0:  # already done
                continue

            days_left = max(0, c.end_day - day)
            b         = min(2, days_left // 3)         # urgency bin 0/1/2

            if np.random.rand() < self.eps:            # ε-greedy explore
                idx = np.random.randint(11)
            else:
                idx = int(np.argmax(self.Q[b]))

            β         = self.beta_grid[idx]
            price     = β * (B_left / R_left)          # valid b/c limit = B_left
            limit     = B_left
            bundles.add(self._bundle(c, price, limit))

            # tag campaign for tomorrow’s learning update
            c._bin, c._idx = b, idx

        return bundles
