# def hanoi_solver(disks, goal_peg=2):
#     """ Solve the tower of Hanoi starting from any (legal) configuration
#         Args:
#             disks: initial configuration
#             goal_peg: the goal peg, where you want to build the tower, 2= rightmost peg
#         Returns:    
#             the min n. of moves necessary to solve the task from the given config.
#     """
#     n = len(disks)
#     # Calculate the next target peg for each of the disks
#     target = goal_peg
#     targets = [0] * n # need this for loop below
#     for i in range(n-1,-1,-1):
#         targets[i] = target
#         if disks[i] != target:
#             # To allow for this move, the smaller disk needs to get out of the way
#             target = 3 - target - disks[i]
        
#     i=0
#     move_counter=0
#     while i <n: # Not yet solved ?
#         # Find the disk that should move
#         for i in range(n):
#             if targets[i] != disks[i]:
#                 target = targets[i]
#                 ## ====== Uncomment if want to print moves =====
#                 #print(f"moved disk {i} from peg {disks[i]} to {target}")
#                 ## ===========================================
#                 move_counter +=1
#                 disks[i] = target # Make move
#                 # Update the next targets of the smaller disks
#                 for j in range(i-1,-1,-1):
#                     targets[j] = target
#                     target = 3 - target - disks[j]
#                 break
#             i+=1 # if all targets match, add 1 to i to terminate while loop   
#     return move_counter

from functools import lru_cache

@lru_cache(maxsize=None)
def hanoi_solver(state: tuple, goal_peg: int = 2) -> int:
    """
    Compute minimal moves to solve an n-disk Tower of Hanoi
    from any legal config 'state', ending with all disks on 'goal_peg'.
    
    state: tuple of length n, where state[i] ∈ {0,1,2} is the peg of disk i
           (i=0 is smallest disk).
    goal_peg: which peg (0,1,2) will hold all disks in the end.
    """
    moves = 0
    target = goal_peg
    n = len(state)

    # Process from largest disk down to smallest
    for i in range(n - 1, -1, -1):
        if state[i] != target:
            # disk i must move: add its 2^i cost
            moves += 1 << i
            # flipping target for the smaller stack
            target = 3 - target - state[i]

    return moves