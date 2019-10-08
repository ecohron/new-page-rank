import numpy as np
import copy


# The primary function is pageRank(links), which takes in a list of lists 
# containing integers.
# lists[i] represents the other entries in the list which i is linked to
# The output is a ranks, a list of integers representing the importance of the
# nodes in the input

def pageRank(links):
    A = [[0]*len(links) for i in range(len(links))]
    x = np.ones(len(links))

    alpha = .85
    N = len(links)

    #Populating the probability matrix
    for j in range(len(links)):
        totalConnections = len(links[j])
        if totalConnections != 0:
            for i in links[j]:
                #Each of the entries of the links
                A[i][j] = 1/totalConnections   
        else:
            for i in range(N):
                A[i][j] = 1/N

    #adjusts the matrix to Google form
    for i in range(len(A)):
        for j in range(len(A)):
            A[i][j] *= alpha
            A[i][j] += (1-alpha)/N

    #Converts to a numpy array
    B = np.array(A)
    C = B.copy()

    #Precision threshold is set to the machine precision for floats
    threshold = np.finfo(float).eps

    #Multiplies probability matrix by itself until the columns are nearly identical
    while error(B) > threshold:
        B = np.matmul(B,B)

    #Creates an n-vector with each entry as 1/N
    x = []
    for i in range(N):
        x.append(1/N)

    x = np.matmul(B, x).tolist()
    
    #Creates a copy of x
    x0 = copy.copy(x)
    ranks = []

    #Ranks the indices of the values of x in decreasing order
    for i in range(len(x0)):
        biggest = 0

        for j in range(len(x)):
            if x[j] > biggest:
                biggest = x[j]

        ranks.append(x0.index(biggest))
        x.remove(biggest)

    return ranks

#Gets the largest difference of the norms from one col to the next
def error(B):
    error = 0
    for colNum in range(len(B) - 1):
        b1 = np.linalg.norm(B[:, colNum])
        b2 = np.linalg.norm(B[:, colNum + 1])
        error = max(abs(b1 - b2), error)

    return error
