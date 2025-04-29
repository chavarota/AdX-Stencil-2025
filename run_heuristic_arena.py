# run_heuristic_arena.py
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator

# from agents.heuristics      import RandomBidAgent
# from agents.heuristics.aggressive_agent    import AggressiveAgent
# from agents.heuristics.conservative_agent  import ConservativeAgent

# from aggressive_agent import AggressiveAgent
# from conservative_agent import ConservativeAgent
from hybrid_limit_agent import HybridLimitAgent

if __name__ == "__main__":
    test_agents = [
        HybridLimitAgent(name="HybridRL"),
        # AggressiveAgent(),
        # ConservativeAgent(),
        Tier1NDaysNCampaignsAgent(name="TA-Tier1-1"),
        Tier1NDaysNCampaignsAgent(name="TA-Tier1-2"),
        Tier1NDaysNCampaignsAgent(name="TA-Tier1-3"),
        Tier1NDaysNCampaignsAgent(name="TA-Tier1-4"),
        Tier1NDaysNCampaignsAgent(name="TA-Tier1-5"),
        Tier1NDaysNCampaignsAgent(name="TA-Tier1-6"),
        Tier1NDaysNCampaignsAgent(name="TA-Tier1-7"),
    ]

    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)
