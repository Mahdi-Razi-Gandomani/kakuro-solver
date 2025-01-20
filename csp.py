# Find the next empty cell in the board
def find_empty_location(board, size, pos):
    for row in range(size):
        for col in range(size):
            if(board[row][col]== (0, 0)):
                pos[0]= row
                pos[1]= col
                return True
    return False

# Find constraints (sum hints) for a given cell
def find_cons(board, row, col):
    r = row
    # Find vertical constraint
    while r >= 0:
        if board[r][col][0] != 0:
            v = (r, col)
            break
        r -= 1

    c = col
    # Find horizontal constraint
    while c >= 0:
        if board[row][c][0] != 0:
            h = (row, c)
            break
        c -= 1
    
    return v, h

# Check if placing a number is valid
def is_valid(board, size, row, col, num):
    v, h = find_cons(board, row, col)

    # Check vertical sum validity
    r = v[0]
    c = v[1]
    vv = board[r][c][0]
    sum = num
    rem = 0
    r += 1
    while r < size and board[r][c][0] == 0:
        if board[r][c][1] == num:
            return False
        sum += board[r][c][1]
        if board[r][c][1] == 0:
            rem += 1
        r += 1

    if vv != -1 and sum > vv:
        return False
    if vv != -1 and rem == 1 and sum != vv:
        return False
    if (rem - 1) * 9 + sum < vv:
        return False

    # Check horizontal sum validity
    r = h[0]
    c = h[1]
    hv = board[r][c][1]
    sum = num
    rem = 0
    c += 1
    while c < size and board[r][c][0] == 0:
        if board[r][c][1] == num:
            return False
        sum += board[r][c][1]
        if board[r][c][1] == 0:
            rem += 1
        c += 1

    if hv != -1 and sum > hv:
        return False
    if hv != -1 and rem == 1 and sum != hv:
        return False
    if (rem - 1) * 9 + sum < hv:
        return False
    
    return True

# Solve the Kakuro puzzle using backtracking
def solve_kakuro(board, size):    
    pos =[0, 0]

    # If no empty cell is found, puzzle is solved
    if(not find_empty_location(board, size, pos)):
        return True

    row = pos[0]
    col = pos[1]
    
    for num in range(1, 10):
        if(is_valid(board, size, row, col, num)):
            board[row][col]= (0, num)
            if(solve_kakuro(board, size)):
                return True

            board[row][col] = (0, 0)
                  
    return False