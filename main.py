from env import Environment
#from agent import Agents
from greedyagent import GreedyAgents as Agents
from astaragent import OptimalAStarAgent as Agents
#from version1 import AgentsVersion2 as Agents
#from astaragent import OptimalAStarAgent as Agents
#from bfs_greedy import AStarOptimalAgent as Agent
from greedy_agent_optimal import GreedyAgentsOptimal as Agents

import numpy as np

# Các loại agent có thể sử dụng
from greedyagent import GreedyAgents
from astaragent import OptimalAStarAgent
from greedy_agent_optimal import GreedyAgentsOptimal

def run_experiment(agent_type, map_file, num_agents, n_packages, max_time_steps, seed, show_render=False):
    """
    Chạy thí nghiệm với một loại agent
    """
    np.random.seed(seed)
    
    # Chọn loại agent
    if agent_type.lower() == "greedy":
        Agents = GreedyAgents
        agent_name = "GreedyAgent"
    elif agent_type.lower() == "astar":
        Agents = OptimalAStarAgent
        agent_name = "AStarAgent"
    elif agent_type.lower() == "greedy_optimal":
        Agents = GreedyAgentsOptimal
        agent_name = "GreedyOptimal"
    else:
        raise ValueError(f"Agent type {agent_type} not supported")
    
    # Khởi tạo môi trường
    env = Environment(map_file=map_file, max_time_steps=max_time_steps,
                      n_robots=num_agents, n_packages=n_packages,
                      seed=seed)
    
    state = env.reset()
    agents = Agents()
    agents.init_agents(state)
    print(f"Agent: {agent_name}")
    
    done = False
    t = 0
    while not done:
        actions = agents.get_actions(state)
        next_state, reward, done, infos = env.step(actions)
        state = next_state
        if show_render:
            env.render()
        t += 1

    print("\nEpisode finished")
    print(f"Agent: {agent_name}")
    print(f"Total reward: {infos['total_reward']}")
    print(f"Total time steps: {infos['total_time_steps']}\n")
    
    return {
        "agent": agent_name,
        "reward": infos['total_reward'],
        "time_steps": infos['total_time_steps']
    }

def compare_agents(map_file, num_agents, n_packages, max_time_steps, seed):
    """
    So sánh các loại agent khác nhau
    """
    print(f"\n===== So sánh các Agent trên map {map_file} =====")
    print(f"Parameters: seed={seed}, num_agents={num_agents}, n_packages={n_packages}, max_time_steps={max_time_steps}")
    
    results = []
    
    for agent_type in ["greedy", "astar", "greedy_optimal"]:
        result = run_experiment(agent_type, map_file, num_agents, n_packages, max_time_steps, seed, False)
        results.append(result)
    
    # In bảng so sánh
    print("\n===== KẾT QUẢ SO SÁNH =====")
    print(f"Map: {map_file}")
    print("| Agent | Total Reward | Time Steps |")
    print("|-------|-------------|------------|")
    
    for result in results:
        print(f"| {result['agent']} | {result['reward']:.2f} | {result['time_steps']} |")
    
    return results

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning for Delivery")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--n_packages", type=int, default=10, help="Number of packages")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--max_time_steps", type=int, default=1000, help="Maximum time steps for the environment")
    parser.add_argument("--map", type=str, default="map.txt", help="Map name")
    parser.add_argument("--agent", type=str, default="greedy_optimal", 
                        choices=["greedy", "astar", "greedy_optimal"], 
                        help="Agent type to use")
    parser.add_argument("--compare", action="store_true", help="Compare all agents")
    parser.add_argument("--render", action="store_true", help="Show rendering")

    args = parser.parse_args()
    
    if args.compare:
        # So sánh tất cả các agent
        compare_agents(args.map, args.num_agents, args.n_packages, args.max_time_steps, args.seed)
    else:
        # Chạy một agent
        run_experiment(args.agent, args.map, args.num_agents, args.n_packages, 
                       args.max_time_steps, args.seed, args.render)
