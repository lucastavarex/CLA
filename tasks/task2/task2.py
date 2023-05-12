import cla
import numpy as np
from cla.utils import det_jacobi

print("╔════════════════════════════════════════╗")
print("║  Task 2 - Computacional Linear Algebra ║")
print("║        Professor: Luis Sagrilo         ║")
print("║ Student: Lucas Tavares Da S. Ferreira  ║")
print("║              DRE: 120152739            ║")
print("║                  2023.1                ║")
print("╠════════════════════════════════════════╣")
print("║                                        ║")
print("║         Welcome to task 2!             ║")
print("║                                        ║")
print("╚════════════════════════════════════════╝")

# Define the file name with the matrix A
Matriz_A = 'tasks\data\Matriz_A.dat'

with open(Matriz_A, 'r') as f:
    # Read all lines from the file
    lines = f.readlines()

    # Remove possible line break characters (\n)
    lines = [line.strip() for line in lines]

    # Separate each element of each line, converting them to float
    matrix = [[float(elem) for elem in line.split()] for line in lines]
A = cla.Array(matrix)

while True:
    method = input("\nWhich method for calculating the eigenvalues/eigenvectors would you like to use?\nPower method (ICOD=1) or Jacobi method (ICOD=2): ")
    if method == "1":
        eigenvalues, eigenvectors = cla.eigen.power_method(A)
        print("╔══════════════════════════════════════════════════════════════════════════════════════╗")
        print("║ The Power method calculates the largest eigenvalue and its corresponding eigenvector ║".format(A))
        print("╚══════════════════════════════════════════════════════════════════════════════════════╝")
        print ("A = {}".format(A))
        print ("Eigenvalue = {}".format(eigenvalues))
        print ("Eigenvector = {}".format(eigenvectors))
        break
    elif method == "2":
        eigenvalues, eigenvectors = cla.jacobi_eigen(A)
        print ("A = {}".format(A))
        print ("Eigenvalues = {}".format(eigenvalues))
        print ("Eigenvectors = {}".format(eigenvectors))
        calc_det = input("\nDo you wish to calculate the determinant? yes(y) or no(n):  ")
        if calc_det in ["y", "Y", "yes"]:
            print ("Determinante = {}\n".format(det_jacobi(A))) #multip. elem. da diag. principal
            break
        break
    else:
        print("\n(!!!) Invalid method! Please, try again.")