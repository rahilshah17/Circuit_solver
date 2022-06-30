# -*- coding: utf-8 -*-


"""
NAME: RAHIL AMIT SHAH
ROLL NO. : EE20B104
ASSIGNMENT NO. : 2
AIM : TO CREATE A PROGRAM IN PYTHON THAT SOLVE AC AS WELL AS DC CIRCUITS
"""





from sys import argv              #importing the required modules
import numpy as np
import cmath
import math

#==============================================================================
# Reading from the netlist file and obtaining data
CIRC = '.circuit'                 #Using constants to denote the starting and  
END = '.end'                      #ending commands.  
AC_CMD = '.ac'

if len(argv) != 2:                                              
    print("The no. of arguments given are not accurate.")       
    exit()
    #checking if the no. of arguments given are correct.
    #Sometimes a user may make a mistake in making command line argumnents.
    
    
try:
    with open(argv[1]) as f:        #opening the required file
        lines = f.readlines()       # Reading the lines of the file
        start = -3; end = -5        #Initiating the start and end variables
        for line in lines:              # extracting circuit definition start and end lines
            if CIRC == line[:len(CIRC)]:
                start = lines.index(line)
            elif END == line[:len(END)]:
                end = lines.index(line)
                break
        if start >= end:                # validating circuit block. Start must be less than end for
                                        #any valid circuit.
            print('Invalid circuit definition')
            exit(0)
        

        
except IOError:                         # An error occurs when an invalid file is given as input.
    print('Invalid file')
    exit()



# Finding omega if the ciruit is ac
omega = 0
for x in lines[end:]:    
    if(x.split()[0] == AC_CMD):            
        print("it's an ac circuit")
        freq = float(lines[end+1].split()[2])
        omega = 2*(math.pi)*freq
#==============================================================================






#==============================================================================
#==============================================================================
# DEFINING THE NECESSARY FUNCTIONS AND CLASSES


#printing the information of elements one by one in order

def info(elt,det, n1, n2, n3, n4, dep, value):
    if (det == "It is not a valid element"):
        print("It is not a valid element")
        return None
  
    #for resistors, capacitors, inductors and independant voltage sources.
    #printing the information of the circuit and it's elements.
    print("The type of the element is:" ,det)
    print("The elt is :",elt,"\nIt is connected between:",n1,"\t - \t",n2)
    print("The value is :",value)

    
    #For dependant sources.
    if (dep != "none"):   
        print("It is a dependant source. It depends on",dep,"between:",n3,"\t-\t",n4)
    
    else:
        pass



#Storing the different components as objects in the class.
class component:
    
    def __init__(self, elt, det, n1, n2, n3, n4, dep, value):
        self.elt = elt
        self.dep = dep
        self.value = value
        self.det = det
        if (self.dep == 'none'):
            self.type = self.det[0]
            self.n1 = n1
            self.n2 = n2

        elif(n3>0 and n4<0):
            self.type = self.det[0]
            self.n1 = n1
            self.n2 = n2
            self.n3 = n3
        else:
            self.type = self.det[0]
            self.n1 = n1
            self.n2 = n2
            self.n3 = n3
            self.n4 = n4

#This is the object list. It contains all the objects
objs = list()



#Sometimes when the circuit is inconsistent it gives determinent of (A) as 0
#if it is true it will give an error in our analysis
def check_sol(A):
    A = np.array(A)
    if(np.linalg.det(A) == 0):
        print("The circuit is inconsistent")
    else:
        return 0

# Function to invert a matrix
def inverse(A,order):
    A = np.array(A)
    order = order
    # Initiating a matrix a
    a = np.zeros((order,2*order),dtype = complex)
    
    # Reading matrix coefficients
    for i in range(order):
        for j in range(order):
            a[i][j] = A[i][j]
            
    # Augmenting Identity Matrix of Order we require 
    for i in range(order):        
        for j in range(order):
            if i == j:
                a[i][j+order] = 1
    
    # Applying Guass Jordan Elimination
    for i in range(order):
        if a[i][i] == 0.0:
            exit('Divide by zero detected!, Circuit is incosistent')
            
            
        for j in range(order):
            if i != j:
                ratio = a[j][i]/a[i][i]
    
                for k in range(2*order):
                    a[j][k] = a[j][k] - ratio * a[i][k]
    # Row operation to make principal diagonal element to 1
    for i in range(order):
        divisor = a[i][i]
        for j in range(2*order):
            a[i][j] = a[i][j]/divisor

    inv = np.zeros([order,order],dtype = complex)
    for i in range(order):
        for j in range(order, 2*order):
            inv[i][j-order] = a[i][j]
            
    return inv


#function to give solution in complex numbers in an ac circuit    
def comp_solution(Answer,objs,voltage_sources,dist_nodes):
    print("\n\n\n")
    print("The solution of this circuit is (in complex form):")
    Answer = Answer
    objs = objs
    voltage_sources = voltage_sources
    dist_nodes = dist_nodes
    print("The voltagees at respective nodes is(are):")
    for i in range(len(dist_nodes) - 1):
        if dist_nodes[i] == "GND":
            print("V_GND:",0)
            continue
        else:
            print("V_{}:".format(dist_nodes[i]),Answer[i])
     
    print("\n\n")        
    print("Current through volatge sources is(are):")
    print("direction - (first node to second node)")
    for i in range(len(voltage_sources)):
        print("I(",objs[voltage_sources[i]].n1,"-",objs[voltage_sources[i]].n2,") :",Answer[i+len(dist_nodes)-1])
    
    
    return 0



#function to give the solution in real quantities in an ac circuit
def real_solution(Answer,objs,voltage_sources,dist_nodes):
    print("\n\n\n")
    print("The solution of this circuit is:")
    Answer = Answer
    objs = objs
    voltage_sources = voltage_sources
    dist_nodes = dist_nodes
    print("The voltagees at respective nodes is(are):")
    for i in range(len(dist_nodes) -1):
        if dist_nodes[i] == "GND":
            print("V_GND:",0)
            continue
        else:
            mod=cmath.sqrt((Answer[i].real)**2+(Answer[i].imag)**2)
            pha = cmath.phase(Answer[i])
            print("V_{}:".format(dist_nodes[i]),mod.real,"cos({}t + {}) ".format(omega,pha))
     
    print("\n\n")        
    print("Current through volatge sources is(are):")
    print("direction - (first node to second node)")
    for i in range(len(voltage_sources)):
        mod_i=cmath.sqrt((Answer[i+len(dist_nodes)-1].real)**2+(Answer[i+len(dist_nodes)-1].imag)**2)        
        pha_i = cmath.phase(Answer[i+len(dist_nodes)-1])
        print("I(",objs[voltage_sources[i]].n1,"-",objs[voltage_sources[i]].n2,") :",mod_i.real,"cos({}t + {}) ".format(omega,pha_i))    
    return 0




# Function to give solution for dc circuits
def solution(Answer,objs,voltage_sources,dist_nodes):
    print("\n\n\n")
    print("The solution of this circuit is:")
    Answer = Answer
    objs = objs
    voltage_sources = voltage_sources
    dist_nodes = dist_nodes
    print("The voltagees at respective nodes is(are):")
    for i in range(len(dist_nodes)-1):
        if dist_nodes[i] == "GND":
            print("V_GND:",0)
            continue
        else:
            print("V_{}:".format(dist_nodes[i]),Answer[i].real)
     
    print("\n\n")
    if (len(voltage_sources) != 0):
        print("Current through volatge sources is(are):")
        print("direction - (first node to second node)")
        for i in range(len(voltage_sources)):
            print("I(",objs[voltage_sources[i]].n1,"-",objs[voltage_sources[i]].n2,") :",Answer[i+len(dist_nodes)-1].real)
    return 0


#==============================================================================
#==============================================================================






#==============================================================================
#Finding and analyzing the token values.
print("\n\n")
print("Different elements with their values.\n\n")    
for x in lines[start+1:end]:
    arr2 = x.split()   #Analyzing tokens
    
    try:
        value = 0
        elt = arr2[0]
        detr = (arr2[0])[0]             # Finding what type of element it is?
        n1 = arr2[1]
        n2 = arr2[2]
        if (detr == 'R'):
            det = "Resistor"
        elif (detr == 'L'):
            det = "Inductor"
        elif (detr == 'C'):
            det = "Capacitor"
        elif (detr == 'V'):
            det = "Independant voltage source"
            ac_dc = arr2[3]
            n3 = -1
            n4 = -1
            dep = "none"
            # checking if it's an ac or dc source
            if(ac_dc == "ac"):
                vpp = float(arr2[4])
                phase = float(arr2[5])
                z = complex(0,(phase))
                value = (vpp/2)*(cmath.exp(z))
            else:
                value = float(arr2[3])

        elif (detr == 'I'):
            det = "Independant current source"
            ac_dc = arr2[3]
            n3 = -1
            n4 = -1
            dep = "none"
            # checking if it's an ac or dc source
            if(ac_dc == 'ac'):
                vpp = float(arr2[4])
                phase = float(arr2[5])
                z = complex(0,(phase))
                value = (vpp/2)*(cmath.exp(z))
            else:
                value = float(arr2[3])

        
        elif (detr == 'E'):
            det = "Voltage controlled voltage source"
        elif (detr == 'G'):
            det = "Voltage controlled current source"
        elif (detr == 'H'):
            det = "Current controlled voltage source"
        elif (detr == 'F'):
            det = "Current controlled current source"
        else:
            det = "It is not a valid element"
            continue
#check for dependant sources       
        if (value == 0):
            arr3 = arr2[3].split()
            value = float(arr3[0])
            n3 = -1
            n4 = -1
            dep = "none"
            vpp = 0
            phase = 0
            if (detr == "C"):
                C = value
                value = complex(0,-1/(omega*C))
            if (detr == "L"):
                L = value
                value = complex(0,(omega*L))
            
        objs.append(component(elt,det,n1,n2,n3,n4,dep,value))
        info(elt,det,n1,n2,n3,n4,dep,value)
    except Exception as e:        #If an error occurs while analyzing the 
                               # tokens, we output an error and print it here.
        print("Error",e)
#==============================================================================






# Finding the nodes in the ciruit
all_nodes = list()
for i in range(len(objs)):
    all_nodes.append(objs[i].n1)
    all_nodes.append(objs[i].n2)
    if (objs[i].dep != "none"):
        all_nodes.append(objs[i].n3)
        all_nodes.append(objs[i].n4)
    else:
        pass
    
# Finding the distinct nodes in the circuit
dist_nodes = [] 
for i in all_nodes: 
    if i not in dist_nodes: 
        dist_nodes.append(i)


#making 'GND' as the zeroth element in this array
dist_nodes.remove('GND')
dist_nodes.insert(0,'GND')



# Forming a dictionary to store the key to distinct nodes as sequential numbers.
# This for the ease of the programmer one can do without it too.
nodes_dict = {}
for i in range(1,len(dist_nodes)+1,1):
        nodes_dict[dist_nodes[i-1]] = (i-1)




# number and values of independant voltage sources components
voltage_sources = []
for i in range(len(objs)):
    if (objs[i].det == "Independant voltage source"):
        voltage_sources.append(i)
    else:
        continue

        

#==============================================================================
# =============================================================================
# According to MNA
# A = [[G,B]
#       [C,D]]
# 
# where:
# G = conductance matrix(nxn)
# B = Satifies the currents through voltage sources for kvl(nxm)
# C = B = Satifies the currents through voltage sources for kvl(mxn)
# D = Zero matrix when there are dependant sources(mxm)
# 
# n = no. of nodes in the circuit
# m = no. of voltage sources
# 
# =============================================================================


# =============================================================================
# implementing the above algorithm
# 
# =============================================================================



# =============================================================================
# Matrix G
# 
# initaiting a zero matrix of required dimensions
G = np.zeros([len(dist_nodes) -1,len(dist_nodes) -1],dtype=complex)
for i in range(len(objs)):
    if(objs[i].det == "Resistor" or objs[i].det == "Inductor" or objs[i].det == "Capacitor"):
         if(nodes_dict.get(objs[i].n1) == 0 and nodes_dict.get(objs[i].n2) == 0):
              continue

         if(nodes_dict.get(objs[i].n1) == 0):
             G[nodes_dict.get(objs[i].n2) - 1][nodes_dict.get(objs[i].n2) - 1] += 1/(objs[i].value)
             continue
         if(nodes_dict.get(objs[i].n2) == 0):
             G[nodes_dict.get(objs[i].n1) - 1][nodes_dict.get(objs[i].n1) - 1] += 1/(objs[i].value)
         else:
             G[nodes_dict.get(objs[i].n1) - 1][nodes_dict.get(objs[i].n1) - 1] += 1/(objs[i].value)
             G[nodes_dict.get(objs[i].n2) - 1][nodes_dict.get(objs[i].n2) - 1] += 1/(objs[i].value)
             G[nodes_dict.get(objs[i].n1) - 1][nodes_dict.get(objs[i].n2) - 1] += -1/(objs[i].value)
             G[nodes_dict.get(objs[i].n2) - 1][nodes_dict.get(objs[i].n1) - 1] += -1/(objs[i].value)

# =============================================================================



# =============================================================================
# Matrix B
#  
B = np.zeros([len(dist_nodes)-1,len(voltage_sources)],dtype=complex)
 
for i in range(len(voltage_sources)):
     if(nodes_dict.get(objs[voltage_sources[i]].n1) == 0):
         B[nodes_dict.get(objs[voltage_sources[i]].n2) - 1][i] += -1
         continue
     if(nodes_dict.get(objs[voltage_sources[i]].n2) == 0):
         B[nodes_dict.get(objs[voltage_sources[i]].n1) - 1][i] += 1
         continue
     else:
         B[nodes_dict.get(objs[voltage_sources[i]].n1) - 1][i] += 1
         B[nodes_dict.get(objs[voltage_sources[i]].n2) - 1][i] += -1
 
# ============================================================================= 



# =============================================================================
# Matrix C and D
# 
C = np.matrix.transpose(B)

D = np.zeros([len(voltage_sources),len(voltage_sources)],dtype = complex)

med1 = np.concatenate((G,B),axis = 1)
# #print(med1)
med2 = np.concatenate((C,D),axis = 1)

# =============================================================================



# =============================================================================
# Finally Matrix A

A = np.concatenate((med1,med2),axis = 0)

# =============================================================================
#==============================================================================




#==============================================================================
# =============================================================================
# According to MNA we know:
# Z = [I
#      e]
# I = cuurents from the current sources
# e = Voltages of voltage sources 
# =============================================================================


# =============================================================================
# implementing the above algorithm
# finding I and e
# 

I = np.zeros([len(dist_nodes)-1,1],dtype = complex)

for i in range(len(objs)):
    if (objs[i].det == "Independant current source"):
        for a in range(1,len(dist_nodes)):
            if(nodes_dict(objs[i].n1) == a):
                I[a][0] += objs[i].value
            if(nodes_dict(objs[i].n2) == a):
                I[a][0] += -objs[i].value
                

e = np.zeros([len(voltage_sources),1],dtype = complex)

for x in range(len(voltage_sources)):
    e[x][0] += objs[voltage_sources[x]].value

Z = np.concatenate((I,e),axis = 0)

# =============================================================================
#==============================================================================



#==============================================================================
# =============================================================================
# We know Ax = Z for the ciruit
# 
# Hence the solution for x should be
# 
#     x = (A^(-1))*Z
# 
# Where:
#     A = matrix of KCL equations
#     x = matrix of unknowns
#     Z = matrix of current sources and voltage sources
# 
# =============================================================================




# =============================================================================
# finding unkowns using the above algorithm
# 


#==============================================================================
# Sometimes the circuit have a combination of voltage sources as a loop it 
#           disobey kvl for that loop
# or sometimes only current sources at a node which will disobey kcl at that node.

# In both the cases we will have either infinte soltions or no solutions.
# This is true only when det(A) == 0. Hence, we check for that condition.
 
check_sol(A)
#==============================================================================

# The above step will ensure that determinent is not zero and we don't have 
#          these cases and our circuit is consistent.

print("A")
if(check_sol(A) == 0):
    print("The circuit is consistent")
    Answer = np.dot(inverse(A, len(voltage_sources) + len(dist_nodes) - 1),Z)
    if(omega!=0):    
        comp_solution(Answer,objs,voltage_sources,dist_nodes)
        real_solution(Answer,objs,voltage_sources,dist_nodes)
        
    else:
        solution(Answer,objs,voltage_sources,dist_nodes)
        
else:
    print("The circuit is inconsitent. Please give a valid circuit to solve.")

# =============================================================================
#==============================================================================





# =============================================================================
# =============================================================================
# =============================================================================
# # # 
# # #             THE END
# # #           THANK    YOU
# # # 
# =============================================================================
# =============================================================================
# =============================================================================
