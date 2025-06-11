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
