import os
from pacman import runGames
from layout import getLayout
from ghostAgents import RandomGhost
from multiAgents import ReflexAgent,MultiAgentSearchAgent,MinimaxAgent,AlphaBetaAgent,ExpectimaxAgent,MonteCarloTreeRandomSearchAgent, MonteCarloTreeImprovedSearchAgent
from scipy.stats import ttest_ind, t

def get_layout_names(layouts_dir='layouts'):
    layout_files = [f for f in os.listdir(layouts_dir) if f.endswith('.lay')]
    layout_names = [os.path.splitext(f)[0] for f in layout_files]
    return layout_names

def run_single_game(agent_class, layout_name):
    layout = getLayout(layout_name)
    if layout is None:
        print(f"Error: Layout {layout_name} not found.")
        return None

    pacman = agent_class()
    ghosts = [RandomGhost(i + 1) for i in range(2)]
    display = None
    
    games = runGames(layout=layout,
                     pacman=pacman,
                     ghosts=ghosts,
                     display=display,
                     numGames=1,
                     record=None,
                     catchExceptions=True)

    final_score = games[0].state.getScore()
    print(f"Final score for {agent_class.__name__} on {layout_name}: {final_score}")
    return final_score

def main():
    agents = [ReflexAgent,MultiAgentSearchAgent,MinimaxAgent,AlphaBetaAgent,ExpectimaxAgent,MonteCarloTreeRandomSearchAgent, MonteCarloTreeImprovedSearchAgent]
    layouts = get_layout_names()

    random_search_scores = []
    improved_search_scores = []
    reflex_search_scores=[]
    multiagent_search_scores=[]
    minimax_search_scores=[]
    alpha_beta_search_scores=[]
    expectimax_search_scores=[]

    for layout_name in layouts:
        for agent_class in agents:
            score = run_single_game(agent_class, layout_name)
            if agent_class == ReflexAgent:
                reflex_search_scores.append(score)
                pass
            elif agent_class == MultiAgentSearchAgent:
                multiagent_search_scores.append(score)
                pass
            elif agent_class == MinimaxAgent:
                minimax_search_scores.append(score)
                pass
            elif agent_class == AlphaBetaAgent:
                alpha_beta_search_scores.append(score)
                pass
            elif agent_class == ExpectimaxAgent:
                expectimax_search_scores.append(score)
                pass
            elif agent_class == MonteCarloTreeRandomSearchAgent:
                random_search_scores.append(score)
            elif agent_class == MonteCarloTreeImprovedSearchAgent:
                improved_search_scores.append(score)
    print("Random Search Scores:", random_search_scores)
    print("Improved Search Scores:", improved_search_scores)

    # calculating the t-test
    t_stat, p_value = ttest_ind(random_search_scores, improved_search_scores)
    print("T-statistic:", t_stat)
    print("P-value:", p_value)

    # set the significance level (alpha)
    alpha = 0.05

    # computing the degrees of freedom (df)
    df = len(random_search_scores) + len(improved_search_scores) - 2
    print('degree of freedom: ',df)
    
    # Calculate the critical t-value
    critical_t = t.ppf(1 - alpha/2, df)
    print("Critical t-value:", critical_t)

    # Decision based on critical t-value
    if abs(t_stat) > critical_t:
        print("Reject the null hypothesis: There is a significant difference between the means.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference between the means.")

if __name__ == "__main__":
    main()
