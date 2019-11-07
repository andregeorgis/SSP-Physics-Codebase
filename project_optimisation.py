'''
    Note that this code uses the more complex final equation - equation that was
    intended for optimisation and made fewer assumptions (the equation in the
    final report).
'''

'''Various Python Packages that help with graphing and solving'''
import matplotlib.pyplot as plt ## For graphing
import numpy as np ## For processing data and forming coordinate systems
import pylab ## For graph aesthetics
from matplotlib.widgets import Slider ## Was meant to be for interactive graphs
from mpl_toolkits.mplot3d import Axes3D ## For 3D Graphing
from scipy.optimize import fsolve ## Used for solving differential systems

'''Constants'''
o = 2 # Sigma - Not really a constant
k = 2 # Kappa - Not really a constant
t = 0.0001 # Inner Wall Thickness
L = t # Length
R = 0.001 # Radius


'''Equations for project translated into Python'''
def theta(x, a, o=2, k=2):
    return k * (o ** 2) * (L / t) * ((k * (o ** 2))/(q(x, a, o, k)) + 1) * Q(x, a)

def Q(x, a, o=2, k=2):
    return ((2 / 3) * np.sin(a / 2) * f_3(x, a, o, k)) / (F(x, a, o, k))

def F(x, a, o=2, k=2):
    return ((np.pi / 4) * (((o + 1) ** 4) - (o ** 4)) + (((a + np.sin(a)) * f_4(x, a, o, k)) / 8)) * q(x, a, o, k) + (4 / (9 * np.pi)) * (np.sin(a / 2) ** 2) * (f_3(x, a, o, k) ** 2)

def q(x, a, o=2, k=2):
    return 2 * o + 1 + (a / (2 * np.pi)) * (2 * o + 2 + x) * x

def f_3(x, a, o=2, k=2):
    return ((o + 1 + x) ** 3) - ((o + 1) ** 3)

def f_4(x, a, o=2, k=2):
    return ((o + 1 + x) ** 4) - ((o + 1) ** 4)






'''Values'''
samples = 1000 ## How many times we sample x and alpha
xMax = 2 ## Maximum value for x
xlist = np.linspace(0, xMax, samples) ## Take x from 0 to xMax
alist = np.linspace(0, 6.28, samples) ## Take alpha from 0 to 2 * pi
X, A = np.meshgrid(xlist, alist) ## Make the coordinate system for x and alpha

Z = theta(X, A) ## Z is the z-axis, and we initially set it to the bending angle

'''Finds the optimum graphs of alpha and x in terms of sigma and kappa by
   computing various graphs of the bending angle and finding the maximum'''
def optimiseAX():
    ## Initialising lists that will be holding data values
    aMatrix = []
    xMatrix = []
    counter = 0

    tempA = 0
    tempX = 0

    ## Loop through a specific range of sigma - currently we are going through
    ## sigma from 0.1 to 10 in steps of 0.1 (since we start at 1 and divide by
    ## 10 to get 0.1 - and end at 100 and divide by 10 to get 10)
    for o in range(1, 101):
        ## Adding new list to prepare for new data values
        aMatrix.append([])
        xMatrix.append([])

        ## Keeping track of sigma
        print(o)
        o /= 10
        o = round(o, 4)

        ## Loop through a specific range of kappa - currently we are going
        ## through kappa from 0.1 to 10 in steps of 0.1 (since we start at 1 and
        ## divide by 10 to get 0.1 - and end at 100 and divide by 10 to get 10)
        for k in range(1, 101):
            ## Keeping track of kappa
            print(k)
            k /= 10
            k = round(k, 4)

            ## Computing graph of the bending angle
            Z = theta(X, A, o, k)

            ## Find the maximum of this graph - returns the list index not the
            ## actual values of alpha and x
            tempA, tempX = np.where(Z == np.amax(Z))

            ## Add the maximum to the lists holding data values - with a
            ## transformation to the actual values of alpha and x
            aMatrix[counter].append(6.28 * tempA[0] / samples)
            xMatrix[counter].append(xMax * tempX[0] / samples)

        ## To keep track of where we are in the list
        counter += 1

    ## Changing data values into plottable collection of data
    aArr = np.matrix(aMatrix)
    xArr = np.matrix(xMatrix)
    ## Setting up coordinate system with sigma and kappa
    oList = np.asarray([round(x / 10, 4) for x in range(1, 101)])
    kList = np.asarray([round(x / 10, 4) for x in range(1, 101)])

    oList, kList = np.meshgrid(oList, kList)

    ## Setting up plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ## Plotting the 3D surface for optimum alpha with appropriate labelling and
    ## titleing
    ax.plot_surface(oList, kList, aArr, cmap='hsv')
    ax.set_xlabel('o')
    ax.set_ylabel('k')
    ax.set_zlabel('a')

    ## Display plot
    plt.show()

    ## Setting up plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ## Plotting the 3D surface for optimum x with appropriate labelling and
    ## titleing
    ax.plot_surface(oList, kList, xArr, cmap='hsv')
    ax.set_xlabel('o')
    ax.set_ylabel('k')
    ax.set_zlabel('x')

    ## Display plot
    plt.show()

'''Same as above but saving to a text file so that it can be read from the file
   at a faster execution time rather than doing the computations all over again
   (for context, this function may take several hours but reading from file may
   take at most around 10 seconds)'''
def optimiseAXArchive():
    ## Initialising lists that will be holding data values
    aMatrix = []
    xMatrix = []
    counter = 0

    tempA = 0
    tempX = 0

    ## In file written as "x1 | x2 | x3" so first value is archived as "x1", and
    ## all other values are archived as "| xn"
    first = True

    ## Choosing file to archive to
    f = open("archive4.txt", "w")

    ## Loop through a specific range of sigma - currently we are going through
    ## sigma from 0 to 15 in steps of 0.05 (since we start at 0 and divide by
    ## 300/15 to get 0 - and end at 300 and divide by 300/15 to get 15)
    for o in range(0, 301):
        ## Adding new list to prepare for new data values
        aMatrix.append([])
        xMatrix.append([])

        ## Keeping track of sigma
        print(o)
        o *= 15 / 300
        o = round(o, 4)

        ## The first value in line is archived differently
        first = True

        ## Loop through a specific range of kappa - currently we are going through
        ## kappa from 0 to 15 in steps of 0.05 (since we start at 0 and divide by
        ## 300/15 to get 0 - and end at 300 and divide by 300/15 to get 15)
        for k in range(0, 301):
            ## Keeping track of kappa
            print(k)
            k *= 15 / 300
            k = round(k, 4)

            ## Computing the bending angle graph
            Z = theta(X, A, o, k)

            ## Finding maximum
            tempA, tempX = np.where(Z == np.amax(Z))

            ## Adding to lists that collect data
            aMatrix[counter].append(round(6.28 * tempA[0] / samples, 8))
            xMatrix[counter].append(round(xMax * tempX[0] / samples, 8))

            ## Record the value in the file
            if first:
                f.write(f'{round(6.28 * tempA[0] / samples, 8)}, {round(xMax * tempX[0] / samples, 8)}')
                first = False
            else:
                f.write(f'|{round(6.28 * tempA[0] / samples, 8)}, {round(xMax * tempX[0] / samples, 8)}')

        ## To keep track of where we are in the list
        counter += 1

        ## Start on new line in file
        f.write('\n')

'''Restores the file "archive1.txt" that archived the optimum alpha and x for
   sigma and kappa values between 0 and 15 at steps of 0.05'''
def restore1_3D():
    ## Opening the file
    f = open("archive1.txt", "r");

    ## Initialising lists that will be holding data values
    aMatrix = []
    xMatrix = []
    counter = 0

    ## Loop through a specific range of sigma - currently we are going through
    ## sigma from 0 to 15 in steps of 0.05 (since we start at 0 and divide by
    ## 300/15 to get 0 - and end at 300 and divide by 300/15 to get 15)
    for u in range(1, 301):
        ## Read the next line and parse accordingly
        line = f.readline().split("|")
        aMatrix.append([])
        xMatrix.append([])

        ## The first set of x and alpha is accidentally archived wrong
        first = True

        ## Loop through a specific range of kappa - currently we are going through
        ## kappa from 0 to 15 in steps of 0.05 (since we start at 0 and divide by
        ## 300/15 to get 0 - and end at 300 and divide by 300/15 to get 15)
        for k in range(1, 301):
            ## Add to data lists
            if first:
                tempA, tempX = line[k - 1].split(" ")
                first = False
            else:
                tempA, tempX = line[k - 1].split(", ")
            aMatrix[counter].append(float(tempA))
            xMatrix[counter].append(float(tempX))

        ## To keep track of where we are in the list
        counter += 1

    ## Changing data values into plottable collection of data
    aArr = np.matrix(aMatrix)
    xArr = np.matrix(xMatrix)
    ## Setting up coordinate system with sigma and kappa
    oList = np.asarray([round(x / 10, 4) for x in range(1, 301)])
    kList = np.asarray([round(x / 10, 4) for x in range(1, 301)])

    oList, kList = np.meshgrid(oList, kList)

    ## Setting up plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ## Plotting the 3D surface for optimum alpha with appropriate labelling and
    ## titleing
    graph = ax.plot_surface(oList, kList, aArr, cmap='hsv')
    plt.title('Optimum Alpha')
    ax.set_xlabel('Sigma')
    ax.set_ylabel('Kappa')
    ax.set_zlabel('Alpha')

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('Alpha Colour Guide')

    ## Display plot
    plt.show()

    ## Setting up the plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ## Plotting the 3D surface for optimum x with appropriate labelling and
    ## titleing
    graph = ax.plot_surface(oList, kList, xArr, cmap='hsv')
    plt.title('Optimum X')
    ax.set_xlabel('Sigma')
    ax.set_ylabel('Kappa')
    ax.set_zlabel('X')

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('X Colour Guide')

    ## Display plot
    plt.show()

def restore1_contour():
    ## Opening the file
    f = open("archive1.txt", "r");

    ## Initialising lists that will be holding data values
    aMatrix = []
    xMatrix = []
    counter = 0

    ## Loop through a specific range of sigma - currently we are going through
    ## sigma from 0 to 15 in steps of 0.05 (since we start at 0 and divide by
    ## 300/15 to get 0 - and end at 300 and divide by 300/15 to get 15)
    for u in range(1, 301):
        ## Read the next line and parse accordingly
        line = f.readline().split("|")
        aMatrix.append([])
        xMatrix.append([])

        ## The first set of x and alpha is accidentally archived wrong
        first = True

        ## Loop through a specific range of kappa - currently we are going through
        ## kappa from 0 to 15 in steps of 0.05 (since we start at 0 and divide by
        ## 300/15 to get 0 - and end at 300 and divide by 300/15 to get 15)
        for k in range(1, 301):
            ## Add to data lists
            if first:
                tempA, tempX = line[k - 1].split(" ")
                first = False
            else:
                tempA, tempX = line[k - 1].split(", ")
            aMatrix[counter].append(float(tempA))
            xMatrix[counter].append(float(tempX))

        ## To keep track of where we are in the list
        counter += 1

    ## Changing data values into plottable collection of data
    aArr = np.matrix(aMatrix)
    xArr = np.matrix(xMatrix)
    ## Setting up coordinate system with sigma and kappa
    oList = np.asarray([round(x / 10, 4) for x in range(1, 301)])
    kList = np.asarray([round(x / 10, 4) for x in range(1, 301)])

    oList, kList = np.meshgrid(oList, kList)

    ## Setting up plot
    fig = plt.figure()
    ax = plt.axes()

    ## Plotting the contour for optimum alpha with appropriate labelling and
    ## titleing
    graph = ax.contourf(oList, kList, aArr, cmap='hsv')
    plt.title('Optimum Alpha')
    ax.set_xlabel('Sigma')
    ax.set_ylabel('Kappa')

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('Alpha Colour Guide')

    ## Display plot
    plt.show()

    ## Setting up the plot
    fig = plt.figure()
    ax = plt.axes()

    ## Plotting the contour for optimum x with appropriate labelling and
    ## titleing
    graph = ax.contourf(oList, kList, xArr, cmap='hsv')
    plt.title('Optimum X')
    ax.set_xlabel('Sigma')
    ax.set_ylabel('Kappa')

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('X Colour Guide')

    ## Display plot
    plt.show()

def restore2_3D():
    ## Opening the file
    f = open("archive2.txt", "r");

    ## Initialising lists that will be holding data values
    aMatrix = []
    xMatrix = []
    counter = 0

    ## Ignore first line
    f.readline()

    ## Loop through a specific range of sigma - currently we are going through
    ## sigma from 0.04 to 2 in steps of 0.04 (since we start at 1 and divide by
    ## 25 to get 0.04 - and end at 50 and divide by 25 to get 2)
    for u in range(1, 51):
        ## Read the next line and parse accordingly
        line = f.readline().split("|")
        aMatrix.append([])
        xMatrix.append([])

        ## Loop through a specific range of kappa - currently we are going through
        ## kappa from 0.04 to 2 in steps of 0.04 (since we start at 1 and divide by
        ## 25 to get 0.04 - and end at 50 and divide by 25 to get 2)
        for k in range(1, 51):
            ## Add to data lists
            tempA, tempX = line[k].split(", ")
            aMatrix[counter].append(float(tempA))
            xMatrix[counter].append(float(tempX))

        ## To keep track of where we are in the list
        counter += 1


    ## Changing data values into plottable collection of data
    aArr = np.matrix(aMatrix)
    xArr = np.matrix(xMatrix)
    ## Setting up coordinate system with sigma and kappa
    oList = np.asarray([round(x / 25, 4) for x in range(1, 51)])
    kList = np.asarray([round(x / 25, 4) for x in range(1, 51)])

    oList, kList = np.meshgrid(oList, kList)

    ## Setting up plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ## Plotting the 3D surface for optimum alpha with appropriate labelling and
    ## titleing
    graph = ax.plot_surface(oList, kList, aArr, cmap='hsv')
    ax.set_xlabel('o')
    ax.set_ylabel('k')
    ax.set_zlabel('a')

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('Colour Guide')

    ## Display plot
    plt.show()

    ## Setting up the plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ## Plotting the 3D surface for optimum x with appropriate labelling and
    ## titleing
    graph = ax.plot_surface(oList, kList, xArr, cmap='hsv')
    ax.set_xlabel('o')
    ax.set_ylabel('k')
    ax.set_zlabel('x')

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('Colour Guide')

    ## Display plot
    plt.show()

def restore3_3D():
    ## Opening the file
    f = open("archive3.txt", "r");

    ## Initialising lists that will be holding data values
    aMatrix = []
    xMatrix = []
    counter = 0

    ## Ignore first line
    f.readline()

    ## Loop through a specific range of sigma - currently we are going through
    ## sigma from 0.05 to 5 in steps of 0.05 (since we start at 1 and divide by
    ## 20 to get 0.05 - and end at 100 and divide by 20 to get 5)
    for u in range(1, 101):
        ## Read the next line and parse accordingly
        line = f.readline().split("|")
        aMatrix.append([])
        xMatrix.append([])

        ## Loop through a specific range of kappa - currently we are going through
        ## kappa from 0.05 to 5 in steps of 0.05 (since we start at 1 and divide by
        ## 20 to get 0.05 - and end at 100 and divide by 20 to get 5)
        for k in range(1, 101):
            ## Add to data lists
            tempA, tempX = line[k].split(", ")
            aMatrix[counter].append(float(tempA))
            xMatrix[counter].append(float(tempX))

        ## To keep track of where we are in the list
        counter += 1

    ## Changing data values into plottable collection of data
    aArr = np.matrix(aMatrix)
    xArr = np.matrix(xMatrix)
    ## Setting up coordinate system with sigma and kappa
    oList = np.asarray([round(x / 20, 4) for x in range(1, 101)])
    kList = np.asarray([round(x / 20, 4) for x in range(1, 101)])

    oList, kList = np.meshgrid(oList, kList)

    ## Setting up plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ## Plotting the 3D surface for optimum alpha with appropriate labelling and
    ## titleing
    graph = ax.plot_surface(oList, kList, aArr, cmap='hsv')
    ax.set_xlabel('o')
    ax.set_ylabel('k')
    ax.set_zlabel('a')

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('Colour Guide')

    ## Display plot
    plt.show()

    ## Setting up the plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ## Plotting the 3D surface for optimum x with appropriate labelling and
    ## titleing
    graph = ax.plot_surface(oList, kList, xArr, cmap='hsv')
    ax.set_xlabel('o')
    ax.set_ylabel('k')
    ax.set_zlabel('x')

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('Colour Guide')

    ## Display plot
    plt.show()


def restore4_3D():
    ## Opening the file
    f = open("archive4.txt", "r");

    ## Initialising lists that will be holding data values
    aMatrix = []
    xMatrix = []
    counter = 0

    ## Ignore first line
    f.readline()

    ## Loop through a specific range of sigma - currently we are going through
    ## sigma from 0.05 to 15 in steps of 0.05 (since we start at 1 and divide by
    ## 300/15 to get 0.05 - and end at 300 and divide by 300/15 to get 15)
    for u in range(1, 301):
        ## Read the next line and parse accordingly
        line = f.readline().split("|")
        aMatrix.append([])
        xMatrix.append([])

        ## Loop through a specific range of kappa - currently we are going through
        ## kappa from 0.05 to 15 in steps of 0.05 (since we start at 1 and divide by
        ## 300/15 to get 0.05 - and end at 300 and divide by 300/15 to get 15)
        for k in range(1, 301):
            ## Add data to lists
            tempA, tempX = line[k].split(", ")
            aMatrix[counter].append(float(tempA))
            xMatrix[counter].append(float(tempX))

        ## To keep track of where we are in the list
        counter += 1


    ## Changing data values into plottable collection of data
    aArr = np.matrix(aMatrix)
    xArr = np.matrix(xMatrix)
    ## Setting up coordinate system with sigma and kappa
    oList = np.asarray([round(x * 15 / 300, 4) for x in range(1, 301)])
    kList = np.asarray([round(x * 15 / 300, 4) for x in range(1, 301)])

    oList, kList = np.meshgrid(oList, kList)

    ## Changes font size
    plt.rcParams.update({'font.size': 20})

    ## Setting up plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ## Plotting the 3D surface for optimum alpha with appropriate labelling and
    ## titleing
    graph = ax.plot_surface(oList, kList, aArr, cmap='hsv')
    plt.title('Optimum Alpha in Radians')
    ax.set_xlabel('Sigma', labelpad = 20)
    ax.set_ylabel('Kappa', labelpad = 20)
    ax.set_zlabel('Alpha (Rad)', labelpad = 20)

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('Alpha Colour Guide (Rad)', labelpad = 20)

    ## Display plot
    plt.show()

    ## Changes font size
    plt.rcParams.update({'font.size': 20})

    ## Setting up the plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ## Plotting the 3D surface for optimum x with appropriate labelling and
    ## titleing
    graph = ax.plot_surface(oList, kList, xArr, cmap='hsv')
    plt.title('Optimum X')
    ax.set_xlabel('Sigma', labelpad = 20)
    ax.set_ylabel('Kappa', labelpad = 20)
    ax.set_zlabel('X', labelpad = 20)

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('X Colour Guide', labelpad = 20)

    ## Display plot
    plt.show()


def restore4_contour():
    ## Opening the file
    f = open("archive4.txt", "r")
    ## Changing font size
    plt.rcParams.update({'font.size': 20})

    ## Initialising lists that will be holding data values
    aMatrix = []
    xMatrix = []
    counter = 0

    ## Ignore first line
    f.readline()

    ## Loop through a specific range of sigma - currently we are going through
    ## sigma from 0.05 to 15 in steps of 0.05 (since we start at 1 and divide by
    ## 300/15 to get 0.05 - and end at 300 and divide by 300/15 to get 15)
    for u in range(1, 301):
        ## Read the next line and parse accordingly
        line = f.readline().split("|")
        aMatrix.append([])
        xMatrix.append([])

        ## Loop through a specific range of kappa - currently we are going through
        ## kappa from 0.05 to 15 in steps of 0.05 (since we start at 1 and divide by
        ## 300/15 to get 0.05 - and end at 300 and divide by 300/15 to get 15)
        for k in range(1, 301):
            ## Add data to lists
            tempA, tempX = line[k].split(", ")
            aMatrix[counter].append(float(tempA))
            xMatrix[counter].append(float(tempX))

        ## To keep track of where we are in the list
        counter += 1

    ## Changing data values into plottable collection of data
    aArr = np.matrix(aMatrix)
    xArr = np.matrix(xMatrix)
    ## Setting up coordinate system with sigma and kappa
    oList = np.asarray([round(x * 15 / 300, 4) for x in range(1, 301)])
    kList = np.asarray([round(x * 15 / 300, 4) for x in range(1, 301)])

    oList, kList = np.meshgrid(oList, kList)

    ## Setting up plot
    fig = plt.figure()
    ax = plt.axes()

    ## Plotting the contour for optimum alpha with appropriate labelling and
    ## titleing
    graph = ax.contourf(oList, kList, aArr, cmap='hsv')
    plt.title('Optimum Alpha in Radians')
    ax.set_xlabel('Sigma', labelpad = 20)
    ax.set_ylabel('Kappa', labelpad = 20)

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('Alpha Colour Guide (Rad)', labelpad = 20)

    ## Display plot
    plt.show()

    ## Setting up the plot
    fig = plt.figure()
    ax = plt.axes()

    ## Plotting the contour for optimum x with appropriate labelling and
    ## titleing
    graph = ax.contourf(oList, kList, xArr, cmap='hsv')
    plt.title('Optimum X')
    ax.set_xlabel('Sigma', labelpad = 20)
    ax.set_ylabel('Kappa', labelpad = 20)

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('X Colour Guide', labelpad = 20)

    ## Changes font size
    plt.rcParams.update({'font.size': 20})

    ## Display plot
    plt.show()


def thetaGrapher3D(o=2, k=2):
    ## Changes font size
    plt.rcParams.update({'font.size': 20})

    ## Setting up plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ## Setting z-axis to the bending angle
    Z = theta(X, A, o, k)

    ## Setting title
    plt.title(f"Bending Angle in Radians at σ={o} and κ={k}")

    ## Plotting 3D surface with appropriate labelling
    graph = ax.plot_surface(X, A, Z, cmap='hsv')
    ax.set_xlabel('X', labelpad = 20)
    ax.set_ylabel('Alpha (Rad)', labelpad = 20)
    ax.set_zlabel('Theta (Rad)', labelpad = 20)

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('Bending Angle Colour Guide (Rad)', labelpad = 20)

    ## Display plot
    plt.show()

def thetaGrapherContour(o=2, k=2):
    ## Changes font size
    plt.rcParams.update({'font.size': 16})

    ## Setting up plot
    fig = plt.figure()
    ax = plt.axes()

    ## Setting z-axis to the bending angle
    Z = theta(X, A, o, k)

    ## Setting title
    plt.title(f"Bending Angle in Radians at σ={o} and κ={k}")

    ## Plotting contour with appropriate labelling
    graph = ax.contourf(X, A, Z, cmap='hsv')
    ax.set_xlabel('X', labelpad = 20)
    ax.set_ylabel('Alpha (Rad)', labelpad = 20)

    ## Display colour guide for z-axis values on the side
    cbar = plt.colorbar(graph)
    cbar.ax.set_ylabel('Bending Angle Colour Guide (Rad)', labelpad = 20)

    ## Plotting where theta is maximum
    tempA, tempX = np.where(Z == np.amax(Z))
    plt.plot([xMax * tempX[0] / samples], [6.28 * tempA[0] / samples], marker='o', markersize=5 , color="black")

    ## Display plot
    plt.show()



if __name__ == "__main__":
    #optimiseAXArchive()
    #restore1_3D()
    #restore2_3D()
    #restore3_3D()
    #thetaGrapher3D(o=2, k=1)
    #thetaGrapherContour(o=2, k=1)
    #restore1_contour()
    restore4_3D()
    restore4_contour()
