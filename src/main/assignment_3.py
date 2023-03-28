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


def problem2():
    print()

def problem3():
    print()

def problem4():
    print()

def problem5():
    print()

def problem6():
    print()





if __name__ == "__main__":
    ans1 = problem1(0, 2, 10, 1)
    #ans2 = problem2()
    #ans3 = problem3()
    #ans4 = problem4()
    #ans5 = problem5()
    #ans6 = problem6()