from joblib import Parallel, delayed
from timeit import default_timer as timer

def yourfunction(k):
    s=3.14*k*k
    return
    #print("Area of a circle with a radius ", k, " is:", s)




if __name__ == "__main__":
    start = timer()
    element_run = Parallel(n_jobs=-1, verbose=50)(delayed(yourfunction)(k) for k in range(1,10))
    #for k in range(10):
    #    yourfunction(k)
    end = timer()
    print(f"Runtime: {end-start}")