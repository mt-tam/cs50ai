import csv
import sys
import time

# Implement spinner while loading
import itertools
spinner = itertools.cycle(['-', '/', '|', '\\'])

from util import Node, StackFrontier, QueueFrontier

# Maps names to a set of corresponding person_ids
names = {}

# Maps person_ids to a dictionary of: name, birth, movies (a set of movie_ids)
people = {}

# Maps movie_ids to a dictionary of: title, year, stars (a set of person_ids)
movies = {}


def load_data(directory):
    """
    Load data from CSV files into memory.
    """
    # Load people
    with open(f"{directory}/people.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            people[row["id"]] = {
                "name": row["name"],
                "birth": row["birth"],
                "movies": set()
            }
            if row["name"].lower() not in names:
                names[row["name"].lower()] = {row["id"]}
            else:
                names[row["name"].lower()].add(row["id"])

    # Load movies
    with open(f"{directory}/movies.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[row["id"]] = {
                "title": row["title"],
                "year": row["year"],
                "stars": set()
            }

    # Load stars
    with open(f"{directory}/stars.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                people[row["person_id"]]["movies"].add(row["movie_id"])
                movies[row["movie_id"]]["stars"].add(row["person_id"])
            except KeyError:
                pass


def main():
    if len(sys.argv) > 2:
        sys.exit("Usage: python degrees.py [directory]")
    directory = sys.argv[1] if len(sys.argv) == 2 else "large"

    # Load data from files into memory
    print("Loading data...")
    load_data(directory)
    print("Data loaded.")
    
    source = person_id_for_name(input("Name: "))
    if source is None:
        sys.exit("Person not found.")
    target = person_id_for_name(input("Name: "))
    if target is None:
        sys.exit("Person not found.")

    
    path = shortest_path(source, target)

    if path is None:
        print("Not connected.")
    else:
        degrees = len(path)
        print(f"{degrees} degrees of separation.")
        path = [(None, source)] + path
        for i in range(degrees):
            person1 = people[path[i][1]]["name"]
            person2 = people[path[i + 1][1]]["name"]
            movie = movies[path[i + 1][0]]["title"]
            print(f"{i + 1}: {person1} and {person2} starred in {movie}")
    


def shortest_path(source, target):
    """
    Returns the shortest list of (movie_id, person_id) pairs
    that connect the source to the target.

    If no possible path, returns None.
    """
    start_time = time.time()
    
    # Create Initial State as Node
    initial_state = Node(source, None, None)
    print("\nWorking on it ...\n")
    
    # Create Frontier and add Initial State
    frontier = QueueFrontier()
    frontier.add(initial_state)

    # Keep track of Explored Nodes
    explored_set = []

    # Keep track of Number of Steps
    no_steps = 0
    
    while True: 

        no_steps += 1
        
        # LOADING SPINNER
        sys.stdout.write(next(spinner))   # write the next character
        sys.stdout.flush()                # flush stdout buffer (actual character display)
        sys.stdout.write('\b')            # erase the last written char
              
        
        # If Frontier is empty, then no solution
        if frontier.empty():
            return None

        # Else, remove a node from frontier (Queue Frontier removes First Node)
        explored_node = frontier.remove()
   
        # If node contains goal state, return the solution
        if explored_node.state == target: 
            
            # Compute path between initial state and explored node
            final_path = find_path(explored_node)

            # Keep track of cost in no. of steps
            print(f"<> Number of steps: #{no_steps}")

            print("\n--- %s seconds ---\n" % (time.time() - start_time))
            return final_path

            
        # Else, add Node to the explored set
        explored_set.append(explored_node)
        
        # Extract only person IDs or states from list of nodes
        explored_states = list(map(lambda x: x.state, explored_set))

        # Expand node, add resulting nodes to the frontier
        for p in neighbors_for_person(explored_node.state): 
            
            action = p[0] # movie id
            parent = explored_node # parent node
            state = p[1] # person id
            
            # Skip if already explored
            if state in explored_states:
                continue
            # Skip if already in frontier
            elif frontier.contains_state(state):
                continue
            else:
                # Create next node
                next_node = Node(state, parent, action)

                # If node contains goal state, return the solution
                if next_node.state == target:
                            
                    # Compute path between initial state and explored node
                    final_path = find_path(next_node)

                    # Keep track of cost in no. of steps
                    print(f"<> Number of steps: #{no_steps}")

                    print("\n--- %s seconds ---\n" % (time.time() - start_time))
                    return final_path

                # Else, add node to frontier    
                frontier.add(next_node)         

  
def person_id_for_name(name):
    """
    Returns the IMDB id for a person's name,
    resolving ambiguities as needed.
    """
    person_ids = list(names.get(name.lower(), set()))
    if len(person_ids) == 0:
        return None
    elif len(person_ids) > 1:
        print(f"Which '{name}'?")
        for person_id in person_ids:
            person = people[person_id]
            name = person["name"]
            birth = person["birth"]
            print(f"ID: {person_id}, Name: {name}, Birth: {birth}")
        try:
            person_id = input("Intended Person ID: ")
            if person_id in person_ids:
                return person_id
        except ValueError:
            pass
        return None
    else:
        return person_ids[0]


def neighbors_for_person(person_id):
    """
    Returns (movie_id, person_id) pairs for people
    who starred with a given person.
    """
    movie_ids = people[person_id]["movies"]
    neighbors = set()
    for movie_id in movie_ids:
        for person_id in movies[movie_id]["stars"]:
            neighbors.add((movie_id, person_id))
    return neighbors


def find_path(node):
    
    path = []

    # Add pairs to path at the start of list
    path.insert(0, (node.action, node.state))

    # Get the parent node
    parent = node.parent

    # Iterate through all parents until reaching initial state (with parent = None)
    while parent.parent is not None:
        path.insert(0, (parent.action, parent.state))
        parent = parent.parent
        
    return path


if __name__ == "__main__":
    main()
    