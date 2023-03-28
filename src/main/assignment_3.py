import numpy as np

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



def problem4():
    print()

def problem5():
    print()

def problem6():
    print()





if __name__ == "__main__":
    ans1 = problem1(0, 2, 10, 1)
    ans2 = problem2(0, 2, 10, 1)
    A = np.array([[2, -1, 1], [1, 3, 1], [-1, 5, 4]])
    b = np.array([6, 0, -3])
    ans3 = problem3(A, b)
    #ans4 = problem4()
    #ans5 = problem5()
    #ans6 = problem6()