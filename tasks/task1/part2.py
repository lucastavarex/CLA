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

while True:
    method = input("\nWhich iterative method would you like to use? Jacobi (ICOD=3) or Gauss-Seidel (ICOD=4): ")
    if method == "3":
        print("╔═══════════════════════╗")
        print("║     Jacobi Method     ║")
        print("╚═══════════════════════╝")
        i=0
        for vetor_b in vetores_B:
            with open(vetor_b, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                vetor = [[float(elem) for elem in line.split()] for line in lines]
            B = cla.Array(vetor)
            try:
              X = cla.systems.solve(A, B, method='jacobi')
              print("╔═══════════════════════╗")
              print("║       VETOR_B_0{}      ║".format(i+1))
              print("╚═══════════════════════╝")
              i=i+1
              print ("A = {}".format(A))
              print ("B = {}".format(B))
              print ("X = {}".format(X))
            except ValueError as e:
              print (e)
        break


    elif method == "4":
          print("╔═══════════════════════╗")
          print("║  Gauss-Seidel Method  ║")
          print("╚═══════════════════════╝")
          i=0
          for vetor_b in vetores_B:
            with open(vetor_b, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                vetor = [[float(elem) for elem in line.split()] for line in lines]
            B = cla.Array(vetor)
            try:
              X = cla.systems.solve(A, B, method='gauss_seidel')
              print("╔═══════════════════════╗")
              print("║       VETOR_B_0{}      ║".format(i+1))
              print("╚═══════════════════════╝")
              i=i+1
              print ("A = {}".format(A))
              print ("B = {}".format(B))
              print ("X = {}".format(X))
            except ValueError as e:
              print (e)
          break

    else:
      print("\n(!!!) Invalid method! Please, try again.")