#############################
# Directions generator from a given value H
### How to use? Type in command line
    ## python H_generator.py H2value1 H2value2 ... H2valuen
# Se crearán entonces archivos para cada valor de H separado, con las direcciones generadas.
# For each given value of H, a file will be created, with the directions generated.
# Last modification: 18/02/22
## We've added a function makeH2database for databases. 
#   from H_generator import makeH2database
#   makeH2database(H2max,threshold,filename) 
#############################

# Modules
import sys
import numpy as np
import itertools
import random

def H2_shortlist(H2):
    '''
    It returns a list of Miller indices (h,k,l), which are solutions of the equation
    H^2=h^2+k^2+l^2. Also, the values satisfy h>k>l.
    Input: H2 (H^2 value)
    '''
    HKL = []
    for h in range(H2):
        for k in range(h,int(np.sqrt(H2)+1)):
            for l in range(k,int(np.sqrt(H2)+1)):
                if (h**2+k**2+l**2==H2):
                    hkl = [h,k,l]
                    HKL.append(hkl)
    return HKL

def HKL_sign(hkl):
    '''
    It returns all possible permutations of (h,k,l) changing signs (+-)
    Input: shortlist [h,k,l]
    '''
    combinations = []
    if hkl.count(0)== 0: # Caso sin ceros
    # Pick up two random elements
        options = [(0,1),(0,2),(1,2)]
        choice = random.choice(options)
        for sign_1 in (1,-1):
            my_hkl = list(hkl)
            my_hkl[choice[0]] = sign_1*my_hkl[choice[0]]
            for sign_2 in (1,-1):
                my_hkl[choice[1]] = sign_2*my_hkl[choice[1]]
                combinations.append(np.array(my_hkl))
        return combinations
    elif hkl.count(0)==1: # 1 zero case
        index_0 = hkl.index(0)
        my_hkl = list(hkl)
        options = [0,1,2]
        del options[index_0]
        choice = random.choice(options)
        for sign in (1,-1):
            my_hkl[choice] = sign*my_hkl[choice]
            combinations.append(np.array(my_hkl))
        return np.array(combinations)
    else: # 2 zeros
        return [np.array(hkl)]

def total_directions(H2,unpack=False):
    '''
    Ir returns a list of lists, with every possible combination of vectors, all of them build from a given H^2 value.
    Input: H2 (valor de H^2).If unpack=True, it returns a single list with all directions.
    '''
    directions = []
    shortlists = H2_shortlist(H2)
    
    for shortlist in shortlists:
        if shortlist.count(0)<2:
            permutations = list(itertools.permutations(shortlist))
            # Delete equivalent elements
            permutations = list(dict.fromkeys(permutations))
        else:
            permutations = []
            h,k,l = shortlist
            permutations.append([h,k,l]);permutations.append([l,h,k]);permutations.append([k,l,h])
            
        for permutation in permutations:
            if unpack==False:
                directions.append(HKL_sign(permutation))
            else:
                vectors = HKL_sign(permutation)
                for vector in vectors:
                    directions.append(vector)
    return directions

def writeH2_file(H2,filename):
    """
    It saves a txt file with the directions generated.
    Input: H2,filename
    """
    directions = total_directions(H2,unpack=True)
    header = 'H2 value: '+str(H2)+'\n'
    header2 = 'Number of directions: '+ str(len(directions))
    np.savetxt(filename,directions,fmt="%d",header=header+header2)

def makeH2database(H2max,threshold,filename='H2_database.txt'):
    """
    It creates a file 'filename.txt' from a H^2 value y and number of possible directions, only if N_Dir>=threshold,
    from 0 to H2max.
    """
    Dir = []
    N = []
    for i in range(H2max):
        if total_directions(i)==None:
            Dir.append(0)
        else:
            Dir.append(len(total_directions(i,unpack=True)))
        N.append(i)
        print('{}/{} completed.'.format(i+1,H2max),end='\r',flush=True)
    Dir = np.array(Dir,dtype=np.int32)
    N = np.array(N,dtype=np.int32)
    
    Dir_N = np.column_stack((N,Dir))
    Dir_N_filtrada = Dir_N[Dir_N[:,1]>=threshold]
    np.savetxt(filename,Dir_N_filtrada,fmt='%d',header='H2 Database - Max value {}, Min Directions {}'.format(H2max,threshold))
# main
if __name__ == "__main__":
    for i in range(1,len(sys.argv)):
        H2 = sys.argv[i]
        filename = 'H2_'+H2+'.txt'
        writeH2_file(int(H2),filename)