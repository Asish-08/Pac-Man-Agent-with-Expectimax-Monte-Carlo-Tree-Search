class MonteCarloTreeSearchAgent(MultiAgentSearchAgent):
    class MCTSNode:
        def __init__(self, game_state, move=None, parent=None, agentIndex=0):
            self.game_state = game_state
            self.move = move
            self.parent = parent
            self.children = []
            self.wins = 0
            self.visits = 0
            self.untried_actions = game_state.getLegalActions(agentIndex)
            self.is_terminal = game_state.isWin() or game_state.isLose()
            self.agentIndex = agentIndex
            self.pacman_direction = None
            self.past_positions = {} 
            self.action_history = []  # Stores recent actions
            self.ACTION_HISTORY_SIZE = 3  # How many actions to remember

        def uct_select_child(self, exploration_strength=math.sqrt(1)):
            if not self.children:
                return None

            log_parent_visits = math.log(self.visits)
            uct_values = [(child, (child.wins / child.visits) + exploration_strength * math.sqrt((2 * log_parent_visits) / child.visits)) for child in self.children]
            
            max_uct_value = max(uct_values, key=lambda x: x[1])[1]

            best_children = [child for child, uct_value in uct_values if uct_value == max_uct_value]
            
            return random.choice(best_children) if best_children else None

    def getAction(self, gameState):
        rootNode = self.MCTSNode(gameState, agentIndex=0)
        iter = 100
        for _ in range(iter):  # Fixed number of iterations for simplicity
            node = self.select_node(rootNode)
            if node is not None:  # Ensure we have a valid node to work with
                score = self.simulate(node.game_state)
                self.backpropagate(node, score)
        # Select the best action to take from the root node's children
        bestAction = max(rootNode.children, key=lambda x: x.visits).move
        return bestAction
    
    def backpropagate(self, node, score):
        while node is not None:  # Propagate until the root node
            node.visits += 1  # Increment visit count
            node.wins += score  # Add the score to wins
            node = node.parent  # Move to the parent node

    def adjust_depth_and_strategy(self, closestFood, food_in_vicinity, legal_actions_count):
        """
        Adjusts the search depth and strategy based on proximity to the closest food,
        the presence of food in the immediate vicinity, and the number of legal actions,
        which indirectly reflects the complexity of Pac-Man's environment.
        """
        # Initialize with nuanced default values
        max_depth = 5  # Basic depth for most scenarios
        ifwhile_stmt = 'Default'

        # When food is directly accessible
        if closestFood == 1:
            max_depth = 1  # Encourage taking direct food
            ifwhile_stmt = 'Direct Food'
        
        # Adjust for immediate vicinity food and food clusters
        elif food_in_vicinity != 0:
            if closestFood > 10:
                max_depth = 6  # Increase depth for exploring towards distant clusters
                ifwhile_stmt = 'Explore Towards Cluster'
            else:
                # For food in vicinity but not closest, adjust depth for strategic navigation
                max_depth = 2
                ifwhile_stmt = 'Strategic Navigation'

        # For distant food with no immediate vicinity food
        elif closestFood > 10 and food_in_vicinity == 0:
            max_depth = 20  # Significantly increase depth for distant exploration
            ifwhile_stmt = 'Expand for Distant Food'

        # Adjusting for moderate distances or complex paths
        elif 5 <= closestFood <= 10:
            max_depth += 3  # Increase for moderate distances
            ifwhile_stmt = 'Intermediate Distance'
            if legal_actions_count <= 3:  # Adjusting further for complexity
                max_depth += 2
                ifwhile_stmt += ', Likely in Alleys'

        return max_depth, ifwhile_stmt


    def simulate(self, game_state):
        current_state = game_state
        depth = 0   
        newPos = current_state.getPacmanPosition()
        newFood = current_state.getFood()
        print(newFood)
        foodList = newFood.asList()
        foodDist = [manhattanDistance(foodPos, newPos) for foodPos in foodList]
        closestFood = min(foodDist, default=float('inf'))
        vicinity_radius = 2
        food_in_vicinity = sum(manhattanDistance(newPos, food) <= vicinity_radius for food in foodList)

        # Initialize max_depth to a default high value; will be adjusted.
        max_depth = 20
        legal_actions = current_state.getLegalActions(0)
        legal_actions_count = len(legal_actions)

        # Adjust depth and strategy based on current state.
        max_depth, ifwhile_stmt = self.adjust_depth_and_strategy(closestFood, food_in_vicinity, legal_actions_count)

        epsilon = 0.3  # Base probability for exploration
        print("If-While Stmt:", ifwhile_stmt, "| Closest Food:", closestFood, "| Max-Depth:", max_depth, "| Food in Vicinity:", food_in_vicinity)

        while not current_state.isWin() and not current_state.isLose() and depth < max_depth:
            possibleActions = current_state.getLegalActions(0)
            if not possibleActions:
                break

            # # If 'Stop' is a possible action and there are other options, remove 'Stop' from the list
            # if 'Stop' in possibleActions and len(possibleActions) > 1:
            #     possibleActions.remove('Stop')

            best_score = -float('inf')
            best_actions = [] 

            # Re-evaluate the closest food distance and epsilon based on the new position.
            newPos = current_state.getPacmanPosition()
            newFood = current_state.getFood()
            foodList = newFood.asList()
            closestFood = min([manhattanDistance(foodPos, newPos) for foodPos in foodList], default=float('inf'))

            # Adjust epsilon based on the new closest food distance
            if closestFood >= 3 and closestFood <= 10:
                epsilon *= 1.5
            elif closestFood > 10 and food_in_vicinity == 0:
                epsilon *= 1.9     # Reduce exploration for far away food to focus on targeted movement.

            for action in possibleActions:
                successor_state = current_state.generateSuccessor(0, action)
                score = self.evaluationFunction(successor_state)
                newFood = successor_state.getFood()
                foodList = newFood.asList()
                closestFood = min([manhattanDistance(foodPos, newPos) for foodPos in foodList], default=float('inf'))

                # Check for immediate ghost threats and adjust strategy if necessary.
                new_pacman_pos = successor_state.getPacmanPosition()
                ghost_states = successor_state.getGhostStates()
                ghost_threat = False
                for ghost_state in ghost_states:
                    if manhattanDistance(new_pacman_pos, ghost_state.getPosition()) <= 1 and ghost_state.scaredTimer == 0:
                        ghost_threat = True
                        break  # Focus on escape; don't consider this action if ghost too close.

                successor_state = current_state.generateSuccessor(0, action)
                score = self.evaluationFunction(successor_state)

                score -=  10 * closestFood

                if not ghost_threat:
                    # Choose action based on epsilon-greedy approach, with adjustments for immediate food or ghost proximity.
                    if random.random() < epsilon:
                        best_actions.append(action)
                    else:
                        if score > best_score:
                            best_score = score
                            best_actions = [action]
                        elif score == best_score:
                            best_actions.append(action)

            if best_actions:
                action = random.choice(best_actions)
                current_state = current_state.generateSuccessor(0, action)
                depth += 1
            else:
                break

        return self.evaluationFunction(current_state)  # Evaluate and return the score of the final state.


    # def simulate(self, game_state):
    #     current_state = game_state
    #     depth = 0   
    #     newPos = current_state.getPacmanPosition()
    #     newFood = current_state.getFood()
    #     print(newFood)
    #     foodList = newFood.asList()
    #     foodDist = [manhattanDistance(foodPos, newPos) for foodPos in foodList]
    #     closestFood = min(foodDist, default=float('inf'))
    #     vicinity_radius = 4
    #     food_in_vicinity = sum(manhattanDistance(newPos, food) <= vicinity_radius for food in foodList)

    #     max_depth = 15  
    #     legal_actions = current_state.getLegalActions(0)
    #     legal_actions_count = len(legal_actions)

    #     # Call the function with the current state's parameters
    #     max_depth, ifwhile_stmt = self.adjust_depth_and_strategy(closestFood, food_in_vicinity, legal_actions_count)


    #     # ifwhile_stmt = 'OG'

    #     # if closestFood <=4 and food_in_vicinity != 0:
    #     #     if len(current_state.getLegalActions(0)) == 5:
    #     #         max_depth = 3
    #     #         ifwhile_stmt = 'Large'
    #     #     elif len(current_state.getLegalActions(0)) == 4:
    #     #         max_depth = 2  
    #     #         ifwhile_stmt = 'Medium'
    #     #     elif len(current_state.getLegalActions(0)) == 3:
    #     #         max_depth = 5
    #     #         ifwhile_stmt = 'Small'
    #     #     elif len(current_state.getLegalActions(0)) == 2:
    #     #         max_depth = 6
    #     #         ifwhile_stmt = 'Very Small'
    #     # elif closestFood >=5 and closestFood <= 10 and food_in_vicinity == 0:
    #     #     if len(current_state.getLegalActions(0)) == 5:
    #     #         max_depth = 8
    #     #         ifwhile_stmt = 'Expand Less Large-1'
    #     #     elif len(current_state.getLegalActions(0)) == 4:
    #     #         max_depth = 8
    #     #         ifwhile_stmt = 'Expand Less Medium-1'
    #     #     elif len(current_state.getLegalActions(0)) == 3:
    #     #         max_depth = 18
    #     #         ifwhile_stmt = 'Expand Less Small-1'
    #     #     elif len(current_state.getLegalActions(0)) == 2:
    #     #         max_depth = 18
    #     #         ifwhile_stmt = 'Expand Less Very Small-1'
    #     # elif closestFood >= 10 and food_in_vicinity == 0:
    #     #     max_depth = 15
    #     #     ifwhile_stmt = 'Expand'
    #     # elif closestFood <= 2 and food_in_vicinity == 1:
    #     #     max_depth = 2
    #     #     ifwhile_stmt = 'Tight'

        
    #     epsilon = 0.1
    #     print("If-While Stmt:",ifwhile_stmt,"| Closest Food:",closestFood,"| Max-Depth:",max_depth, "| Food in Vicinity:",food_in_vicinity)


    #     while not current_state.isWin() and not current_state.isLose() and depth < max_depth:
    #         possibleActions = current_state.getLegalActions(0)
    #         if not possibleActions:
    #             break

    #         best_score = -float('inf')
    #         best_actions = [] 

    #         newPos = current_state.getPacmanPosition()
    #         newFood = current_state.getFood()
    #         foodList = newFood.asList()
    #         closestFood = min([manhattanDistance(foodPos, newPos) for foodPos in foodList], default=float('inf'))


    #         # Increase epsilon if the food is very far from the pacman position
    #         if closestFood > 5 and closestFood < 10:
    #             epsilon = 0.2
    #         elif closestFood > 10:
    #             epsilon = 0

    #         for action in possibleActions:
    #             successor_state = current_state.generateSuccessor(0, action)
    #             score = self.evaluationFunction(successor_state)

    #             # Immediate ghost threat check
    #             new_pacman_pos = successor_state.getPacmanPosition()
    #             ghost_states = successor_state.getGhostStates()
    #             for ghost_state in ghost_states:
    #                 if manhattanDistance(new_pacman_pos, ghost_state.getPosition()) <= 2:
    #                     if ghost_state.scaredTimer == 0: 
    #                         continue

    #             if random.random() < epsilon:
    #                 best_actions.append(action)
    #             else:
    #                 if score > best_score:
    #                     best_score = score
    #                     best_actions = [action]
    #                 elif score == best_score:
    #                     best_actions.append(action)

    #         action = random.choice(best_actions)
    #         current_state = current_state.generateSuccessor(0, action)
    #         depth += 1

    #     # node.past_positions = {pacman_position: 1} 
    #     return self.evaluationFunction(current_state)  # Evaluate the final state

    def select_node(self, node):
        while not node.is_terminal:
            if node.untried_actions:
                return self.expand_node(node)
            else:
                node = node.uct_select_child()  # Select a child based on UCT
        return node

    def expand_node(self, node):
        action = node.untried_actions.pop(random.randint(0, len(node.untried_actions) - 1))
        next_state = node.game_state.generateSuccessor(node.agentIndex, action)
        if next_state is None:
            return None  # In case the game ends or an error occurs
        child_node = self.MCTSNode(next_state, action, node, node.agentIndex)
        node.children.append(child_node)
        return child_node

    def calculate_cluster_bonus(self, food_positions):
        cluster_radius = 2  # Customize this radius for your map 
        max_cluster_bonus = 0

        for food_pos in food_positions:
            cluster_size = 0
            for other_food_pos in food_positions:
                if manhattanDistance(food_pos, other_food_pos) <= cluster_radius:
                    cluster_size += 1

            max_cluster_bonus = max(max_cluster_bonus, cluster_size)

        return max_cluster_bonus * 5  # Scale up the bonus, adjust '5' as needed

    def evaluationFunction(self, currentGameState):
        successorGameState = currentGameState

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        positionScore = successorGameState.getScore()

        # Prioritize avoiding immediate danger
        min_ghost_distance = float('inf')
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            dist = manhattanDistance(ghostState.getPosition(), newPos)
            min_ghost_distance = min(min_ghost_distance, dist)

            if scaredTime == 0 and dist <= 1:  # IMMEDIATE THREAT!
                return -float('inf')  # Pacman should avoid this state at all costs

        # Calculate ghost influence with more nuance 
        ghostAlert = 0
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            dist = manhattanDistance(ghostState.getPosition(), newPos)
            if scaredTime > 0:
                ghostAlert -= 0.01 / (dist + 1)  # Lessen fear of scared ghosts
            else:
                if dist < 3:  # Increased danger zone
                    ghostAlert += 3 / (dist + 1)
                elif dist < 5:
                    ghostAlert += 1 / (dist + 1)

        # Food calculations (unchanged)
        foodList = newFood.asList()
        closestFood = min([manhattanDistance(foodPos, newPos) for foodPos in foodList], default=float('inf'))

        foodScale = 40
        ghostScale = -40  # You might adjust ghostScale for desired risk-taking
        positionScore += foodScale / closestFood + ghostScale * ghostAlert

        # if closestFood >=1 and closestFood < 3:
        #     positionScore *= 100
        # elif closestFood >=10:
        #     positionScore /= 2
        # else:
        #     positionScore *= 0.01

        food_positions = successorGameState.getFood().asList() 
        closest_food_dist = min(manhattanDistance(newPos, food) for food in food_positions)
        cluster_bonus = self.calculate_cluster_bonus(food_positions)  # You'd need to define this function

        positionScore -= 15 / closest_food_dist  # Stronger focus on nearby food
        positionScore += cluster_bonus

        return positionScore