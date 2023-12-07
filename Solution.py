########################################################
# Written by: Vighanesh Sharma
########################################################

# NOTE:
# 1. All pivot elements must be non-zero, I have not implemented row swapping for in Gauss-Jordan Method
# 2. NumPy is used to display well formatted matrix and hence a requirement to run this code 
# 3. NumPy can display -0 which is equivalent to a 0 only



class ProblemMatrix:
    def __init__(self, n, d, mat_dict):
        self.n = n
        self.d = d
        self.mat_dict = mat_dict


#
# Inverse using Gauss-Jordan elimination method
#                 
def inverseMatrix(problem_matrix: ProblemMatrix):

    try:
        aug_mat_dict = {}
        N = problem_matrix.n
        D = problem_matrix.d
        size = N*D
        for i in range(N):
            aug_mat_dict[i] = [0]*(N*D-i*D)
            aug_mat_dict[-i] = [0]*(N*D-i*D)
        aug_mat_dict[0] = [1 for i in range(size)]
        
        prob_mat_dict = problem_matrix.mat_dict


        #Converting one half of matrix to 0
        for l in range(N*D):
            for j in range(N-1-l//D):
                factor = prob_mat_dict[-j-1][l]/prob_mat_dict[0][l]
                for i in range((l//D)+1):
                    prob_mat_dict[-j-1-i][l-i*D] -= factor*prob_mat_dict[-i][l-i*D]
                    aug_mat_dict[-j-1-i][l-i*D] -= factor*aug_mat_dict[-i][l-i*D]
                for i in range(N-(l//D)-1):
                    if i > j:
                        prob_mat_dict[i-j][l+(j+1)*D] -= factor*prob_mat_dict[i+1][l]
                        aug_mat_dict[i-j][l+(j+1)*D] -= factor*aug_mat_dict[i+1][l]
                    else:
                        prob_mat_dict[i-j][l+(i+1)*D] -= factor*prob_mat_dict[i+1][l]
                        aug_mat_dict[i-j][l+(i+1)*D] -= factor*aug_mat_dict[i+1][l]


        #Scaling diagonal elements to 1
        for l in range(N*D):
            divisor = prob_mat_dict[0][l]
            for i in range(N-l//D):
                prob_mat_dict[i][l] /= divisor
                aug_mat_dict[i][l] /= divisor
            for i in range(l//D):
                prob_mat_dict[-i-1][l-(i+1)*D] /= divisor
                aug_mat_dict[-i-1][l-(i+1)*D] /= divisor


        #Converting the other half to 0
        for l in range(N*D):
            for j in range(l//D):
                factor = prob_mat_dict[l//D-j][j*D+l%D]
                for i in range(N-(l//D)):
                    prob_mat_dict[l//D-j+i][j*D+l%D] -= factor*prob_mat_dict[i][l]
                    aug_mat_dict[l//D-j+i][j*D+l%D] -= factor*aug_mat_dict[i][l]
                for i in range(l//D):
                    if l//D-j-1 >= i:
                        prob_mat_dict[l//D-j-1-i][j*D+l%D] -= factor*prob_mat_dict[-i-1][l-(i+1)*D]
                        aug_mat_dict[l//D-j-1-i][j*D+l%D] -= factor*aug_mat_dict[-i-1][l-(i+1)*D]
                    else:
                        prob_mat_dict[l//D-j-1-i][l-(i+1)*D] -= factor*prob_mat_dict[-i-1][l-(i+1)*D]
                        aug_mat_dict[l//D-j-1-i][l-(i+1)*D] -= factor*aug_mat_dict[-i-1][l-(i+1)*D]

        return ProblemMatrix(N,D, aug_mat_dict)

    except ZeroDivisionError as e:
        print("Matrix not invertible")
    except Exception as e:
        print("Exception occured while inverting:")
        print(e)
   


#Using naive method to calculate, can improve complexity for large size using Strassen algorithm
def multiply_matrix(p_matrix1: ProblemMatrix, p_matrix2: ProblemMatrix):
    mat_dict = {}
    
    # avoiding unnecessary computation where multiplication will be 0
    if p_matrix1.n == p_matrix2.n and p_matrix1.d == p_matrix2.d:
        N = p_matrix1.n
        D = p_matrix1.d
        for i in range(N):
            mat_dict[i] = [0]*(N*D-i*D)
            mat_dict[-i] = [0]*(N*D-i*D)

        res_matrix = ProblemMatrix(N, D, mat_dict)
        size = N*D
        for i in range(size):
            for j in range(size):
                if abs(i-j)%D == 0:
                    res_matrix.mat_dict[(i-j)//D][min(i,j)] = 0
                    for k in range(size):
                        if abs(i-k)%D == 0 and abs(k-j)%D == 0:
                            res_matrix.mat_dict[(i-j)//D][min(i,j)] += p_matrix1.mat_dict[(i-k)//D][min(i,k)]*p_matrix2.mat_dict[(k-j)//p_matrix2.d][min(k,j)]

    elif p_matrix1.n*p_matrix1.d == p_matrix2.n*p_matrix2.d:
        N1 = p_matrix1.n
        D1 = p_matrix1.d
        N2 = p_matrix2.n
        D2 = p_matrix2.d
        size = N1*D1
        for i in range(size):
            mat_dict[i] = [0]*(size-i)
            mat_dict[-i] = [0]*(size-i)

        res_matrix = ProblemMatrix(size, 1, mat_dict)
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    if abs(i-k)%D1 == 0 and abs(k-j)%D2 == 0:
                        res_matrix.mat_dict[(i-j)//1][min(i,j)] += p_matrix1.mat_dict[(i-k)//D1][min(i,k)]*p_matrix2.mat_dict[(k-j)//D2][min(k,j)]
    return res_matrix



def PMToNormal(problem_matrix : ProblemMatrix):
    size = problem_matrix.n*problem_matrix.d
    normal_matrix = [[0]*size for j in range(size)]
    mat_dict = problem_matrix.mat_dict

    for i in mat_dict:
        for j in range(len(mat_dict[i])):
            try:
                if i < 0:
                    normal_matrix[j][j-(i)*problem_matrix.d] = mat_dict[i][j]
                else:
                    normal_matrix[j+(i)*problem_matrix.d][j] = mat_dict[i][j]
            except Exception as e:
                print(e)
                print(i, j)

                
    return normal_matrix

def NormalToPM(normal_matrix, n, d):
    mat_dict = {}
    
    for i in range(n):
        mat_dict[i] = [0]*(n*d-i*d)
        mat_dict[-i] = [0]*(n*d-i*d)

    for i in range(len(normal_matrix)):
        for j in range(len(normal_matrix[0])):
            if abs(j-i)%d == 0:
                try:
                    mat_dict[(i-j)//d][min(i,j)] = normal_matrix[i][j]
                except Exception as e:
                    print(e)
                    print(i, j, normal_matrix[i][j])
                    

    prob_mat = ProblemMatrix(n, d, mat_dict)
    return prob_mat


if __name__ == "__main__":
    
    import numpy as np
    

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    print("\n------------------------EXAMPLE 1---------------------------\n")
    #Making 2 copies since python functions are pass by value, 
    # computation might change original's values
    mat_dict =  {-1:[2,3], 0:[1,2,1,2], 1:[2,1]}
    problem_matrix1 = ProblemMatrix(2, 2, mat_dict)
    mat_dictcp =  {-1:[2,3], 0:[1,2,1,2], 1:[2,1]}
    problem_matrix1cp = ProblemMatrix(2, 2, mat_dictcp)

    print("Matrix dictionary 1, N = 2 and D = 2")
    print(mat_dict)
    print()
    print("Usual Matrix structure 2D array:")
    print(np.matrix(PMToNormal(problem_matrix1)))
    print()
    print("Inverse Matrix through our calculation:")
    inverse1 = inverseMatrix(problem_matrix1cp)
    print(inverse1.mat_dict)
    print("Usual Representation:")
    print(np.matrix(PMToNormal(inverse1)))
    print()
    print("Multiplication of original matrix with calculated inverse")
    print("If output is an identity matrix then inverse is calculated correctly")
    print(multiply_matrix(inverse1, problem_matrix1).mat_dict)
    print("Its Usual representation")
    print(np.matrix(PMToNormal(multiply_matrix(inverse1, problem_matrix1))))
    
    print("\n------------------------EXAMPLE 2---------------------------\n")

    #Making 2 copies since python functions are pass by value, 
    # computation might change original's values
    mat_dict =  {-2:[1], -1:[1,1], 0:[1,2,3], 1:[1,1], 2:[1]}
    problem_matrix1 = ProblemMatrix(3, 1, mat_dict)
    mat_dictcp =  {-2:[1], -1:[1,1], 0:[1,2,3], 1:[1,1], 2:[1]}
    problem_matrix1cp = ProblemMatrix(3, 1, mat_dictcp)

    print("Matrix dictionary 1, N = 3 and D = 1")
    print(mat_dict)
    print()
    print("Usual Matrix structure 2D array:")
    print(np.matrix(PMToNormal(problem_matrix1)))
    print()
    print("Inverse Matrix through our calculation:")
    inverse1 = inverseMatrix(problem_matrix1cp)
    print(inverse1.mat_dict)
    print("Usual Representation:")
    print(np.matrix(PMToNormal(inverse1)))
    print()
    print("Multiplication of original matrix with calculated inverse")
    print("If output is an identity matrix then inverse is calculated correctly")
    print(multiply_matrix(inverse1, problem_matrix1).mat_dict)
    print("Its Usual representation")
    print(np.matrix(PMToNormal(multiply_matrix(inverse1, problem_matrix1))))

    print("\n------------------------EXAMPLE 3---------------------------\n")
    #Making 2 copies since python functions are pass by value, 
    # computation might change original's values
    mat_dict =  {0:[1,2,3,384,5,46,7,8,49,10,117,12], 1:[1,2,3,44,5,463,57,38,9], 2:[1,23,834,4,5,6], 3:[1,24,3], -1:[1,2,435,4,5,46,7,8,9], -2:[1,32,3,44,5,6], -3:[1,2,3]}
    problem_matrix1 = ProblemMatrix(4, 3, mat_dict)
    mat_dictcp =  {0:[1,2,3,384,5,46,7,8,49,10,117,12], 1:[1,2,3,44,5,463,57,38,9], 2:[1,23,834,4,5,6], 3:[1,24,3], -1:[1,2,435,4,5,46,7,8,9], -2:[1,32,3,44,5,6], -3:[1,2,3]}
    problem_matrix1cp = ProblemMatrix(4, 3, mat_dictcp)

    print("Matrix dictionary 1, N = 4 and D = 3")
    print(mat_dict)
    print()
    print("Usual Matrix structure 2D array:")
    print(np.matrix(PMToNormal(problem_matrix1)))
    print()
    print("Inverse Matrix through our calculation:")
    inverse1 = inverseMatrix(problem_matrix1cp)
    print(inverse1.mat_dict)
    print("Usual Representation:")
    print(np.matrix(PMToNormal(inverse1)))
    print()
    print("Multiplication of original matrix with calculated inverse")
    print("If output is an identity matrix then inverse is calculated correctly")
    print(multiply_matrix(inverse1, problem_matrix1).mat_dict)
    print("Its Usual representation")
    print(np.matrix(PMToNormal(multiply_matrix(inverse1, problem_matrix1))))