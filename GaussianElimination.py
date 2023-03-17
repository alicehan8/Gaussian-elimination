import numpy as np
import warnings

def swapRows(A, i, j):
    """
    interchange two rows of A
    operates on A in place
    """
    tmp = A[i].copy()
    A[i] = A[j]
    A[j] = tmp

def relError(a, b):
    """
    compute the relative error of a and b
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return np.abs(a-b)/np.max(np.abs(np.array([a, b])))
        except:
            return 0.0

def rowReduce(A, i, j, pivot):
    """
    reduce row j using row i with pivot pivot, in matrix A
    operates on A in place
    """
    factor = A[j][pivot] / A[i][pivot]
    for k in range(len(A[j])):
        if np.isclose(A[j][k], factor * A[i][k]):
            A[j][k] = 0.0
        else:
            A[j][k] = A[j][k] - factor * A[i][k]


# stage 1 (forward elimination)
def forwardElimination(B):
    """
    Return the row echelon form of B
    """
    A = B.copy().astype(float)
    m, n = np.shape(A)
    for i in range(m-1):
        # Let leftmostNonZeroCol be the position of the leftmost nonzero value 
        # in row i or any row below it 
        leftmostNonZeroRow = m
        leftmostNonZeroCol = n
        ## for each row below row i (including row i)
        for h in range(i,m):
            ## search, starting from the left, for the first nonzero
            for k in range(i,n):
                if (A[h][k] != 0.0) and (k < leftmostNonZeroCol):
                    leftmostNonZeroRow = h
                    leftmostNonZeroCol = k
                    break
        # if there is no such position, stop
        if leftmostNonZeroRow == m:
            break
        # If the leftmostNonZeroCol in row i is zero, swap this row 
        # with a row below it
        # to make that position nonzero. This creates a pivot in that position.
        if (leftmostNonZeroRow > i):
            swapRows(A, leftmostNonZeroRow, i)
        # Use row reduction operations to create zeros in all positions 
        # below the pivot.
        for h in range(i+1,m):
            rowReduce(A, i, h, leftmostNonZeroCol)
    return A

#################### 

# If any operation creates a row that is all zeros except the last element,
# the system is inconsistent; stop.
def inconsistentSystem(B):
    """
    B is assumed to be in echelon form; return True if it represents
    an inconsistent system, and False otherwise
    """
    # remove the next line
    A = B.copy().astype(float)
    row, col = np.shape(A)
    #The nonzeros in the matrix
    nonzerosRow, nonzerosCol = np.nonzero(A) 
    #m, n = np.shape(nonzeros)
    m = len(nonzerosRow)
    n = len(nonzerosCol)
    
    if nonzerosRow[m - 1] == nonzerosRow[m - 2]:
        return False
    elif nonzerosCol[n-1] != col - 1:
            return False
    else:
        return True
        
    

def backsubstitution(B):
    """
    return the reduced row echelon form matrix of B
    """
    A = B.copy().astype(float)
    row, col = np.shape(A)
    nonzeroRow, nonzeroCol = np.nonzero(A)
    #m = len(nonzeroRow)
    #n = len(nonzeroCol)

    for i in reversed(range(row)):
        n, = np.nonzero(A[i])
        if (len(n) == 0):
            i += 1
        else:
            pivot = A[i][n[0]]
            A[i] = A[i] / pivot
        
    for h in reversed(range(row)):
        for k in reversed(range(h)):
            n, = np.nonzero(A[h])
            if(len(n) != 0):
                pivot = n[0]
                rowReduce(A, h, k, pivot)
    
    return A
        
#####################


