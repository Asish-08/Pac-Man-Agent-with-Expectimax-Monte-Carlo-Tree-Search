import os
from pacman import runGames
from multiAgents import ReflexAgent, MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent, MonteCarloTreeRandomSearchAgent,MonteCarloTreeImprovedSearchAgent 
from layout import getLayout
from ghostAgents import RandomGhost
from graphicsDisplay import PacmanGraphics  # This class name might vary
from tabulate import tabulate

def get_layout_names(layouts_dir='layouts'):

    layout_files = [f for f in os.listdir(layouts_dir)]
    layout_names = [os.path.splitext(f)[0] for f in layout_files]
    return layout_names

def run_single_game(agentClass, layoutName):

    layout = getLayout(layoutName)
    if layout is None:
        print(f"Error: Layout {layoutName} not found.")
        return None

    pacman = agentClass() 
    ghosts = [RandomGhost(i + 1) for i in range(2)]  
    display= None
    #display = PacmanGraphics(1.0)
    
    games = runGames(layout=layout,
                     pacman=pacman,
                     ghosts=ghosts,
                     display=display,
                     numGames=1,
                     record= None,
                     catchExceptions=True)  

    final_score = games[0].state.getScore()  # Example of accessing the final score
    print(f"Final score for {agentClass.__name__} on {layoutName}: {final_score}")

    return final_score
def main():
    agents = [ReflexAgent, MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent, MonteCarloTreeRandomSearchAgent,MonteCarloTreeImprovedSearchAgent ]
    layouts = get_layout_names('layouts' )
    results = []


    for layout in layouts:
        for agent in agents:
            score = run_single_game(agent, layout)
            results.append({'agent': agent.__name__, 'layout': layout, 'score': score})

    # Print collected data as a table
    print(tabulate(results, headers='keys', tablefmt='grid'))

    print("done")
    quit()
if __name__ == "__main__":
    main()
