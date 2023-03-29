import numpy as np
import decimal

def func1(t: float, w: float):
    return t - (w**2)
    #return w - (t**2) + 1

def problem1(a: float, b: float, N: float, alpha: float):
    h: float = (b-a)/N
    t: float = a
    w: float = alpha

    #wi+1 = wi + h(function)

    for i in range(1, N+1):
        w = w + (h * func1(t, w))
        t = t + h
    
    print(w)
    print()


def problem2(a: float, b: float, N: float, alpha: float):
    h: float = (b-a)/N
    t: float = a
    w: float = alpha
    
    for i in range(1, N+1):
        #print(w)
        k1 = h * func1(t, w)
        k2 = h * func1(t+(h/2), w + (k1/2))
        k3 = h * func1(t+(h/2), w + (k2/2))
        k4 = h * func1(t+h, w + k3)
        w = w + ((k1+(2*k2)+(2*k3)+k4)/6)
        t = t + h
    print(w)
    print()

def compute_lcm(x, y):

   # choose the greater number
   if x > y:
       greater = x
   else:
       greater = y

   while(True):
       if((greater % x == 0) and (greater % y == 0)):
           lcm = greater
           break
       greater += 1

   return lcm

def problem3(A, b):
    n = len(b)
    Ab = np.concatenate((A, b.reshape(n,1)), axis=1)
    x = np.zeros(n)

    for i in range(n):
        if Ab[i][i] == 0.0:
            break
            
        for j in range(i+1, n):
            ratio = Ab[j][i]/Ab[i][i]
            
            for k in range(n+1):
                Ab[j][k] = Ab[j][k] - ratio * Ab[i][k]

    # Back Substitution
    x[n-1] = Ab[n-1][n]/Ab[n-1][n-1]

    for i in range(n-2,-1,-1):
        x[i] = Ab[i][n]
        
        for j in range(i+1,n):
            x[i] = x[i] - Ab[i][j]*x[j]
        
        x[i] = x[i]/Ab[i][i]
    
    sol = np.concatenate((Ab, x.reshape(n,1)), axis=1)

    y = sol[:,n+1]
    print(y)
    print()



def problem4(A, b):
    #  1  1  0  3
    #  2  1 -1  1
    #  3 -1 -1  2
    # -1  2  3 -1
    #make the upper triangle and keep track of every row operation
    n = len(b)
    U = A
    L = np.identity(n)
    for i in range(n):
        for j in range(i+1, n):
            factor: float = float(A[j,i] / A[i,i])
            for z in range(i, n):
                num = float(factor * A[i,z])
                U[j,z] = float(A[j,z] - num)
            L[j,i] = float(factor)

    #i got a little carried away, this part in not necessary
    y = np.zeros(n)
    y[0] = float(b[0] / U[0,0])
    for i in range(1, n):
        sum:float = float(b[i])
        for j in range(0, i):
            sum = float(sum - float(L[i,j] * y[j]))
        y[i] = float(sum)
    
    detU:float = U[0,0]
    for i in range(1, n):
        detU = float(detU * U[i,i])
    
    detL:float = L[0,0]
    for i in range(1, n):
        detL = float(detL * L[i,i])

    determinant = (detU * detL)

    print(float(determinant - (1e-14)))
    print()
    print(L)
    print()
    print(U)
    print()

def problem5(A):
    n = len(A)
    trueFlag = True
    for i in range(n):
        max = A[i,i]
        sum = 0
        for j in range(n):
            if j == i:
                continue
            sum = sum + abs(A[i,j])
        if sum > max:
            trueFlag = False

    print(trueFlag)
    print()

def problem6(A):
    AT = np.transpose(A)
    symmetric = np.array_equal(A, AT)

    pivotsPositive = True
    U = A
    n = len(A)
    #lets see if all of the pivots are positive
    for i in range(n):
        for j in range(i+1, n):
            factor: float = float(A[j,i] / A[i,i])
            for z in range(i, n):
                num = float(factor * A[i,z])
                U[j,z] = float(A[j,z] - num)

    for i in range(n):
        if U[i,i] < 0:
            pivotsPositive = False
            break

    print(symmetric and pivotsPositive)


if __name__ == "__main__":
    ans1 = problem1(0, 2, 10, 1)
    ans2 = problem2(0, 2, 10, 1)
    A = np.array([[2, -1, 1], [1, 3, 1], [-1, 5, 4]])
    b = np.array([6, 0, -3])
    ans3 = problem3(A, b)

    A2 = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]],dtype=float)
    b2 = np.array([1, 1, -3, 4], dtype=float)
    ans4 = problem4(A2, b2)

    A3 = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 4]])
    ans5 = problem5(A3)

    A4 = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
    ans6 = problem6(A4)