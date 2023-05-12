import cla
from cla.systems import solve

# Define the file name with the matrix A
Matriz_A = 'tasks\data\Matriz_A.dat'
# Define the file name with vector B
vetores_B = ['tasks\data\Vetor_B_01.dat', 'tasks\data\Vetor_B_02.dat', 'tasks\data\Vetor_B_03.dat']
# Read matrix A
with open(Matriz_A, 'r') as f:
    # Read all lines from the file (from data file)
    lines = f.readlines()

    # Remove possible line break characters (\n)
    lines = [line.strip() for line in lines]

    # Separate each element of each line, converting them to float
    matrix = [[float(elem) for elem in line.split()] for line in lines]
A = cla.Array(matrix)

#TODO: Remove duplicated code // the code below do the same that the code above, but to the vector B

while True:
    method = input("\nWhich decomposition method would you like to use? LU (ICOD=1) or Cholesky (ICOD=2): ")
    if method == "1":
        print("╔═══════════════════════╗")
        print("║   LU decomposition    ║")
        print("╚═══════════════════════╝")
        i=0
        L, U = cla.decomposition.lu_decomposition(A)  #Só carrego 1x o A
        for vetor_b in vetores_B:
            with open(vetor_b, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                vetor = [[float(elem) for elem in line.split()] for line in lines]
            B = cla.Array(vetor)
            X = solve(A, B, method='lu')
            print("╔═══════════════════════╗")
            print("║       VETOR_B_0{}      ║".format(i+1))
            print("╚═══════════════════════╝")
            i=i+1
            print ("A = {}".format(A))
            print ("B = {}".format(B))
            print ("L = {}".format(L))
            print ("U = {}".format(U))
            print ("X = {}".format(X))
        break
            
    elif method == "2":
        print("╔════════════════════════╗")
        print("║ Cholesky decomposition ║")
        print("╚════════════════════════╝")
        i=0
        L, LT = cla.decomposition.cholesky_decomposition(A) #Só carrego 1x o A
        for vetor_b in vetores_B:
            with open(vetor_b, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                vetor = [[float(elem) for elem in line.split()] for line in lines]
            B = cla.Array(vetor)
            X = solve(A, B, method='cholesky')
            print("╔═══════════════════════╗")
            print("║       VETOR_B_0{}      ║".format(i+1))
            print("╚═══════════════════════╝")
            i=i+1
            print ("A = {}".format(A))
            print ("B = {}".format(B))
            print ("L = {}".format(L))
            print ("LT = {}".format(LT))
            print ("X = {}".format(X))
        break
    else:
        print("\n(!!!) Invalid method! Please, try again.")