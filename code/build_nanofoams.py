"""
##################################################
 build_nanofoams.py - Última modificación 5/11/21
 
## Parámetros para pasar por línea de comandos: input_file.txt n_workers
## Ejemplo:
    # python build_nanofoams.py input_file.txt 4
Si el usuario decide no pasar n_workers, se corre con todos los cores*threads disponibles.
## Devuelve:
    # 1 archivo data_name.txt con la data de los átomos, clasificados en fase sólida/porosa
    # 1 archivo log_name.txt con datos varios de la estructura
###################################################
"""
# Librerias
import numpy as np
import random
import sys
import scipy as sp
from scipy import special
from loky import ProcessPoolExecutor
from tqdm import tqdm
from os.path import exists as file_exists
###################################################

print("Loading parameters...",end='',flush=True)

struct_info = np.loadtxt(sys.argv[1],dtype=np.float32)

lattice = int(struct_info[0])

# Tamaño de caja
x_size = struct_info[1]
y_size = struct_info[2]
z_size = struct_info[3]

# Parámetro de red
lattice_parameter = struct_info[4]

phi_b = struct_info[5]
if not (0<=phi_b<=1):
    raise Exception('phi_b must be a number between 0 and 1.')

## Con phi_b calculamos xi como en el paper
xi = np.sqrt(2)* sp.special.erfinv(2*phi_b-1)

# Número de ondas
N = int(struct_info[7])
# Constante a
a = struct_info[8]
# Semilla
np.random.seed(int(struct_info[9]))

# Tamanos reducidos de caja
x_size_reduced = int(x_size / lattice_parameter)
y_size_reduced = int(y_size / lattice_parameter)
z_size_reduced = int(z_size / lattice_parameter)

print(' Ready.')

# Usamos funciones de los otros dos scripts para chequear L_mean, y
# generar H2_file con direcciones
print('\nSearching in database for nearest H2 value...')
import H_generator

if not file_exists('H2_database.txt'):
    H_generator.makeH2database(500, 48)
    print('Warning: H2_database.txt not found. New database file created instead.')
else:
    print('Using existing directions file '+'H2_database.txt')

H2_database = np.loadtxt('H2_database.txt',dtype=np.int32)
L_mean = struct_info[6]
a = struct_info[8]
H2_real = ((a/L_mean)*(0.53*phi_b+0.41))**2

print('H^2 value calculated:',H2_real)
    
# Ahora buscamos en la base de datos a ver cuál es el H^2 más cercano 
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
H2_final = find_nearest(H2_database[:,0],H2_real)
    
print('Nearest value of H^2 in database:',H2_final)
    
L_mean_new = (a/np.sqrt(H2_final))*(0.53*phi_b+0.41)
print('New L_mean:',L_mean_new)
print('Relative error in mean ligament diameter:',np.abs(1-(L_mean/L_mean_new)),'\n')

filename_H2 = 'H2_'+str(H2_final)+'.txt'

if not file_exists(filename_H2):
    H_generator.writeH2_file(H2_final, filename_H2)
    print('Directions file created.\n')
else:
    print('Using existing directions file '+filename_H2)
    
pi2a = 2*np.pi/a
q0 = pi2a * np.sqrt(H2_final)

# Direcciones y numero de direcciones - desde txt 
print("Loading directions from txt...",end='',flush=True)
directions = np.loadtxt(filename_H2,dtype=np.float32)
N_dir = len(directions)
print(" Ready.\n")
print("Initializing phases and vectors...")
# Inicializamos array de fases
Phi = np.pi*(-1 + 2*np.random.random_sample((N_dir,int(N/N_dir))))
print('Phi shape:',Phi.shape)
Qi = pi2a * directions
Qi = Qi.astype(np.float32)
print('Number of waves, Number of directions:',(Phi.shape[0]*Phi.shape[1],N_dir))
print("Mean phi,Std phi: ",(np.mean(Phi),np.mean(np.std(Phi,axis=1))))
print("Mean wave direction:",(1/N_dir)*np.sum(Qi,axis=0),'\n')
###############################################
# Armado y evaluacion de la funcion f
sqr2N = np.sqrt(2/(Phi.shape[0]*Phi.shape[1])).astype(np.float32)

# Clasificación de las partículas
def type_particle(f):
    if f>xi or np.isclose(f,xi):
        return 2 # Fase porosa
    else: return 1 # Fase sólida

# Función vectorizada...
type_particle = np.vectorize(type_particle)

print("Initializing atoms...")
# Grilla cuadrada 
x,y,z = np.meshgrid(range(x_size_reduced),range(y_size_reduced),range(z_size_reduced),indexing='ij')

x = lattice_parameter*x.astype(np.float32)
y = lattice_parameter*y.astype(np.float32)
z = lattice_parameter*z.astype(np.float32)
points =  np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)

#BCC
if lattice == 1: 
    print('Lattice type: BCC')
    x_bcc = x + lattice_parameter/2
    y_bcc = y + lattice_parameter/2
    z_bcc = z + lattice_parameter/2        
    points_bcc = np.stack((x_bcc.ravel(), y_bcc.ravel(), z_bcc.ravel()), axis=1)
    points = np.concatenate((points,points_bcc),axis=0)

#FCC
elif lattice==2:
    print('Lattice type: FCC')
    x_fcc = x + lattice_parameter/2
    y_fcc = y + lattice_parameter/2
    z_fcc = z + lattice_parameter/2
    
    points_fcc_xy = np.stack((x_fcc.ravel(), y_fcc.ravel(), z.ravel()), axis=1)
    points_fcc_xz = np.stack((x_fcc.ravel(), y.ravel(), z_fcc.ravel()), axis=1)
    points_fcc_yz = np.stack((x.ravel(), y_fcc.ravel(), z_fcc.ravel()), axis=1)
    points = np.concatenate((points,points_fcc_xy,points_fcc_xz,points_fcc_yz),axis=0)
else:
    print('Lattice type: Cubic')
    pass

NParticles = points.shape[0]
print("Ready. Number of atoms: "+str(NParticles),'\n')


print("Evaluating f in lattice...")

chunk_size=32 # Hiperparámetro de la generación de la espuma
chunks = int(NParticles/32) 
point_batches = np.array_split(points,chunks,axis=0)
# Versión paralelizada
def chunk_F(chunk):
    grid_dot = np.dot(Qi,chunk.T)
    terminos = np.cos(np.transpose([grid_dot])+Phi).astype(np.float32)
    return sqr2N*np.sum(terminos,axis=(1,2))

# Get the number of threads for the work
if len(sys.argv) == 2: # No argument was passed for number of threads
    n_workers = None
    print("Using all threads available...")
else:
    n_workers = int(sys.argv[2])
    print("Using {} threads".format(n_workers))

with ProcessPoolExecutor(max_workers=n_workers) as e:
    F = list(tqdm(e.map(chunk_F,point_batches),total=len(point_batches)))

print("Concatenate atoms and classify...")
F = np.concatenate(F)

types = type_particle(F) # Clasificación según F

print("Evaluation completed.\n")
##################################
# Guardamos todo en .txt
print("Saving data file...")


## Para abrir en Ovito, ponemos un preambulo:

timestep = "ITEM: TIMESTEP \n0\n"
NAtoms = "ITEM: NUMBER OF ATOMS \n"+str(NParticles)+'\n'
bounds = "ITEM: BOX BOUNDS pp pp pp \n"

size_x = '0.0 '+str(x_size)+'\n'
size_y = '0.0 '+str(y_size)+'\n'
size_z = '0.0 '+str(z_size)+'\n'
dimensions = size_x+size_y+size_z

properties = "ITEM: ATOMS id type x y z f\n"

preamble = timestep+NAtoms+bounds+dimensions+properties

with open('data_'+sys.argv[1][:-4]+'.txt','w') as f:
    f.write(preamble)
    for i in range(NParticles):
        f.write('%s %s %s %s %s %s\n' % (i+1,types[i],points[i][0],points[i][1],points[i][2],F[i]))
print("Data file saved succesfully.\n")
##################################
print("Saving log...")
# Guardamos en otro archivo información de interés en un log
## Diámetro medio de ligamento
L_mean = L_mean_new
## Surface to volume ratio
S = (2*q0/(np.pi*np.sqrt(3)))*np.exp(-xi/2)
## Surface to solid volume ratio
S_B = S/phi_b
## Gaussian curvature
k_g = ((q0**2)/6)*((xi**2)-1)
## Mean curvature
k_m = -(xi*q0/2)*np.sqrt(np.pi/6)
## Genus per unit volume
G_v = ((q0**3)/(12*(np.pi**2) * np.sqrt(3))) * ((1-xi**2))* np.exp(-(xi**2)/2)
## Reduced genus per unit volume
g_v = G_v*(L_mean**3)

log_names = ['q0','phi_B','xi','L_mean (mean ligament diameter)','S (surface to volume ratio)','S_B (surface area to volume ratio)',
     'k_g (gaussian curvature)','k_m (mean curvature)','G_v (genus per unit volume)',
     'g_v (Reduced genus per unit volume)']
log_values = [q0,phi_b,xi,L_mean,S,S_B,k_g,k_m,G_v,g_v] 

with open('log_'+sys.argv[1][:-4]+'.txt' , 'w') as f:
    f.write('# Structure data: \n \n')
    f.write('## General:\n \n')
    for i in range(4):
        f.write('%s = %s \n' % (log_names[i],log_values[i]))
    f.write('\n## Surface data: \n \n')
    for i in range(4,6):
        f.write('%s = %s \n' % (log_names[i],log_values[i]))
    f.write('\n## Curvature:\n\n')
    for i in range(6,8):
        f.write('%s = %s \n' % (log_names[i],log_values[i]))
    f.write('\n## Genus:\n')
    for i in range(8,10):
        f.write('%s = %s \n' % (log_names[i],log_values[i]))
print("Log completed. Exiting program.\n")