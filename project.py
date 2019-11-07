'''
    Note that this code uses the simpler final equation - equation that made
    many assumptions before optimisation purposes.
'''

'''Various Python Packages that help with graphing and solving'''
import matplotlib.pyplot as plt ## For graphing
import numpy as np ## For processing data and forming coordinate systems
import pylab ## For graph aesthetics
from matplotlib.widgets import Slider ## Was meant to be for interactive graphs
from mpl_toolkits.mplot3d import Axes3D ## For 3D Graphing
from scipy.optimize import fsolve ## Used for solving differential systems


'''Constants'''
u = 5 ## Mu - Not really a constant
L = 0.05 ## Length
R = 0.001 ## Radius


'''Equations for project translated into Python'''
## Bending Angle
def theta(x, a, u=5):
    return u * (L / R) * ((u / 2) / (1 + (a * x) / (2 * np.pi)) + 1) * Q(x, a)

## Component of equation above
def Q(x, a):
    return (np.pi * x * np.sin(a / 2)) / F(x, a)

## Component of equation above
def F(x, a):
    return (np.pi ** 2) * (1 + (a * x) / (2 * np.pi)) * (1 + (x * (a + np.sin(a)) / (2 * np.pi))) + 2 * (x ** 2) * (np.sin(a / 2) ** 2)

## Partial derivative of bending angle in terms of alpha
def thetaDiffA(x, a, u=5):
    return (1 + (a * x) / (2 * np.pi)) * (u / 2 + 1 + (a * x) / (2 * np.pi)) * QDiffA(x, a) - (u / (4 * np.pi)) * x * Q(x, a)

## Partial derivative of bending angle in terms of x
def thetaDiffX(x, a, u=5):
    return (1 + (a * x) / (2 * np.pi)) * (u / 2 + 1 + (a * x) / (2 * np.pi)) * QDiffX(x, a) - (u / (4 * np.pi)) * a * Q(x, a)

## Partial derivative of Q in terms of alpha
def QDiffA(x, a):
    return (np.pi * 0.5 * x * np.cos(a / 2) * F(x, a) - np.pi * x * np.sin(a / 2) * FDiffA(x, a)) / (F(x, a) ** 2)

## Partial derivative of Q in terms of x
def QDiffX(x, a):
    return (np.pi * np.sin(a / 2) * F(x, a) - np.pi * x * np.sin(a / 2) * FDiffX(x, a)) / (F(x, a) ** 2)

## Partial derivative of F in terms of alpha
def FDiffA(x, a):
    return np.pi * x * (1 + np.cos(a) * 0.5) + ((x ** 2) / 4) * (2 * a + 5 * np.sin(a) + a * np.cos(a))

## Partial derivative of F in terms of x
def FDiffX(x, a):
    return np.pi * a + np.pi * np.sin(a) * 0.5 + (a * x / 2) * (a + np.sin(a)) + 4 * x * (np.sin(a / 2) ** 2)

## Equation obtained when setting partial derivative of bending angle in terms of
## alpha to 0
def system1(x, a, u=2):
    return (u / (4 * np.pi)) * x * Q(x, a) - (1 + (x * a) / (2 * np.pi)) * (u / 2 + 1 + (x * a) / (2 * np.pi)) * QDiffA(x, a)

## Equation obtained when setting partial derivative of bending angle in terms of
## x to 0
def system2(x, a, u=2):
    return (u / (4 * np.pi)) * a * Q(x, a) - (1 + (x * a) / (2 * np.pi)) * (u / 2 + 1 + (x * a) / (2 * np.pi)) * QDiffX(x, a)

## Used to calculate the solution to the above two equations (as a system of
## of differential equations)
def equations(p, u=2):
    x, a = p
    return (system1(x, a, u), system2(x, a, u))

'''Values'''
samples = 1000 ## How many times we sample x and alpha
xMax = 5 ## Maximum value for x
xlist = np.linspace(0, xMax, samples) ## Take x from 0 to xMax
alist = np.linspace(0, 6.28, samples) ## Take alpha from 0 to 2 * pi
X, A = np.meshgrid(xlist, alist) ## Make the coordinate system for x and alpha

Z = theta(X, A) ## Z is the z-axis, and we initially set it to the bending angle


'''Contour Map of Bending Angle'''
def thetaGrapher(u=5):
    ## Set z-axis to bending angle
    Z = theta(X, A, u)

    ## Set the x axis and y axis limits (miinimum and maximum value on axis)
    pylab.xlim([-1,xMax + 1])
    pylab.ylim([-1,8])

    ## Provide a title for the contour plot
    plt.title('Theta')

    ## Set x axis label for the contour plot
    plt.xlabel('X')

    ## Set y axis label for the contour plot
    plt.ylabel('A')

    ## Create contour plot
    cs = plt.contourf(X, A, Z, cmap=plt.cm.get_cmap('hsv'))

    ## Stuff that makes it look nice - not important
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()

    ## Display a colour guide on the side
    cbar = plt.colorbar(cs)
    cbar.ax.set_ylabel('Colour Guide')

    ## Display the contour plot
    plt.show()


def thetaGrapherWithMax(u=5):
    ## Set z-axis to bending angle
    Z = theta(X, A, u)

    ## Set the x axis and y axis limits (miinimum and maximum value on axis)
    pylab.xlim([-1,xMax + 1])
    pylab.ylim([-1,8])

    ## Provide a title for the contour plot
    plt.title('Theta')

    ## Set x axis label for the contour plot
    plt.xlabel('X')

    ## Set y axis label for the contour plot
    plt.ylabel('A')

    ## Create contour plot
    cs = plt.contourf(X, A, Z, cmap=plt.cm.get_cmap('hsv'))

    ## Stuff that makes it look nice - not important
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()

    ## Display a colour guide on the side
    cbar = plt.colorbar(cs)
    cbar.ax.set_ylabel('Colour Guide')

    ## Find where the bending angle is maximum and plot it
    tempA, tempX = np.where(Z == np.amax(Z))
    plt.plot([xMax * tempX[0] / samples], [6.28 * tempA[0] / samples], marker='o', markersize=5 , color="black")
    print(6.28 * tempA[0] / samples)
    print(xMax * tempX[0] / samples)

    # Display the contour plot with maximum point
    plt.show()

'''Contour Map of the Partial Derivative of Theta with respect to A'''
def thetaDiffAGrapher(u=5, levelsList=None):
    ## Set z-axis to partial derivative of the bending angle in terms of alpha
    Z = thetaDiffA(X, A, u)

    ## Set the x axis and y axis limits (minimum and maximum value on axis)
    pylab.xlim([-1,xMax + 1])
    pylab.ylim([-1,8])

    ## Provide a title for the contour plot
    plt.title('ThetaDiffA')

    ## Set x axis label for the contour plot
    plt.xlabel('X')

    ## Set y axis label for the contour plot
    plt.ylabel('A')

    ## Create contour plot - option to provide your own "levelsList" which are
    ## the contour levels inidicated in the colour guide
    if not levelsList:
        cs = plt.contourf(X, A, Z, cmap=plt.cm.get_cmap('hsv'))
    else:
        cs = plt.contourf(X, A, Z, levels=levelsList, cmap=plt.cm.get_cmap('hsv'))

    ## Stuff that makes it look nice - still not that important but can change
    ## how graph looks if a levelsList is provided
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()

    ## Display a colour guide for values of z-axis on the side
    cbar = plt.colorbar(cs)
    cbar.ax.set_ylabel('Colour Guide')

    ## Display the contour plot
    plt.show()



'''Contour Map of the Partial Derivative of Theta with respect to X'''
def thetaDiffXGrapher(u=5, levelsList=None):
    ## Set z-axis to partial derivative of bending angle in terms of x
    Z = thetaDiffX(X, A, u)

    ## Set the x axis and y axis limits (minimum and maximum value on axis)
    pylab.xlim([-1,xMax + 1])
    pylab.ylim([-1,8])

    ## Provide a title for the contour plot
    plt.title('ThetaDiffX')

    ## Set x axis label for the contour plot
    plt.xlabel('X')

    # Set y axis label for the contour plot
    plt.ylabel('A')

    ## Create contour plot - option to provide your own "levelsList" which are
    ## the contour levels inidicated in the colour guide
    if not levelsList:
        cs = plt.contourf(X, A, Z, cmap=plt.cm.get_cmap('hsv'))
    else:
        cs = plt.contourf(X, A, Z, levels=levelsList, cmap=plt.cm.get_cmap('hsv'))

    ## Stuff that makes it look nice - still not that important but can change
    ## how graph looks if a levelsList is provided
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()

    ## Display a colour guide for values on z-axis on the side
    cbar = plt.colorbar(cs)
    cbar.ax.set_ylabel('Colour Guide')

    ## Display the contour plot
    plt.show()



'''Dodgy way of showing how each of the above change with Mu'''
def effectOfU():
    ## Asks user for input in their terminal

    ## Asks for the values of mu user wants to check
    ## Must be separated by spaces
    uList = input("Input multiple values of mu (separate by space): ").split(" ")

    ## If values are not given it will not execute
    if uList == [""]:
        print("Nothing entered try again :)")
        return None

    ## Turns values given into numbers
    try:
        uList = [float(x) for x in uList]
    except:
        print("Remember u is a number :)")

    ## Does the same for the "levelsList" for the contour plots
    ## Must be separated by spaces
    levelsList = input("Input the levels you want for the partial derivative contour: ").split(" ")

    ## Tries to parse the levelsList provided
    ## If not provided, the program uses the default levelsList - will still
    ## execute
    if levelsList != [""]:
        try:
            levelsList = [float(x) for x in levelsList]
            levelsList.sort()
        except:
            print("Remember levels are numbers :)")
            levelsList = None
    else:
        levelsList = None

    ## The graphs of each are subplots on the same figure so we have a specific
    ## number of rows and columns depending on the number of values of mu given.
    ## Counter is used to make each graph one by one
    counter = 0;

    rows = int(len(uList) / 3)
    rows = rows if len(uList) % 3 == 0 else rows + 1

    ## Formats the subplot sizes
    if rows == 1:
        plt.subplots(figsize=(15, 5))
    else:
        plt.subplots(figsize=(15, 8))

    ## Goes through each value of mu and makes the corresponding bending angle
    ## contour plot
    for uTemp in uList:
        ## Change subplot we are drawing on
        counter += 1
        plt.subplot(rows, 3, counter)

        ## Set z-axis to the bending angle
        Z = theta(X, A, uTemp)

        #@ Set the x axis and y axis limits (minimum and maximum values on axes)
        pylab.xlim([-1,xMax + 1])
        pylab.ylim([-1,8])

        ## Provide a title for the contour plot
        plt.title('Theta at u = ' + str(uTemp))

        ## Set x axis label for the contour plot
        plt.xlabel('X')

        ## Set y axis label for the contour plot
        plt.ylabel('A')

        ## Create contour plot
        cs = plt.contourf(X, A, Z, cmap=plt.cm.get_cmap('hsv'))

        ## Stuff that makes it look nice - not important
        cs.cmap.set_over('red')
        cs.cmap.set_under('blue')
        cs.changed()

        ## Display colour guide for z-axis values on the side
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('Colour Guide')


    ## Display the plot
    plt.show()

    ## Re-formatting the subplots
    if rows == 1:
        plt.subplots(figsize=(15, 5))
    else:
        plt.subplots(figsize=(15, 8))

    ## Resetting Counter
    counter = 0

    ## Goes through each value of mu and makes the corresponding partial
    ## derivative of bending angle in terms of alpha contour plot
    for uTemp in uList:
        ## Change subplot we are drawing on
        counter += 1
        plt.subplot(rows, 3, counter)

        ## Set z-axis to partial derivative of bending angle in terms of alpha
        Z = thetaDiffA(X, A, uTemp)

        ## Set the x axis and y axis limits (minimum and maximum values on axes)
        pylab.xlim([-1,xMax + 1])
        pylab.ylim([-1,8])

        ## Provide a title for the contour plot
        plt.title('ThetaDiffA at u = ' + str(uTemp))

        ## Set x axis label for the contour plot
        plt.xlabel('X')

        ## Set y axis label for the contour plot
        plt.ylabel('A')

        ## Create contour plot - option to provide your own "levelsList" which
        ## are the contour levels inidicated in the colour guide
        if not levelsList:
            cs = plt.contourf(X, A, Z, cmap=plt.cm.get_cmap('hsv'))
        else:
            cs = plt.contourf(X, A, Z, levels=levelsList, cmap=plt.cm.get_cmap('hsv'))

        ## Stuff that makes it look nice - still not that important but can
        ## change how graph looks if a levelsList is provided
        cs.cmap.set_over('red')
        cs.cmap.set_under('blue')
        cs.changed()

        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('Colour Guide')

    ## Display plot
    plt.show()

    ## Reformmatting the subplots
    if rows == 1:
        plt.subplots(figsize=(15, 5))
    else:
        plt.subplots(figsize=(15, 8))

    ## Resetting the counter
    counter = 0

    ## Goes through each value of mu and makes the corresponding partial
    ## derivative of bending angle in terms of x contour plot
    for uTemp in uList:
        ## Change subplot we are drawing on
        counter += 1
        plt.subplot(rows, 3, counter)

        Z = thetaDiffX(X, A, uTemp)

        ## Set the x axis and y axis limits (minimum and maximum values on axes)
        pylab.xlim([-1,xMax + 1])
        pylab.ylim([-1,8])

        ## Provide a title for the contour plot
        plt.title('ThetaDiffX at u = ' + str(uTemp))

        ## Set x axis label for the contour plot
        plt.xlabel('X')

        ## Set y axis label for the contour plot
        plt.ylabel('A')

        ## Create contour plot - option to provide your own "levelsList" which
        ## are the contour levels inidicated in the colour guide
        if not levelsList:
            cs = plt.contourf(X, A, Z, cmap=plt.cm.get_cmap('hsv'))
        else:
            cs = plt.contourf(X, A, Z, levels=levelsList, cmap=plt.cm.get_cmap('hsv'))

        ## Stuff that makes it look nice - still not that important but can
        ## change how graph looks if a levelsList is provided
        cs.cmap.set_over('red')
        cs.cmap.set_under('blue')
        cs.changed()

        ## Display colour guide for z-axis values on the side
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('Colour Guide')


    plt.show()


'''Overlaying the Partial Derivatives of Theta'''
def thetaDiffXAndAGrapher(u=5, levelsList=None, show=True):
    ## Only display if we want to display (value of show can be True or False)

    ## Formatting the plot size
    if show:
        plt.subplots(figsize=(10, 7))

    ## Set z-axis to partial derivative of bending angle in terms of alpha
    Z = thetaDiffA(X, A, u)

    ## Set the x axis and y axis limits (minimum and maximum values on axes)
    pylab.xlim([-1,xMax + 1])
    pylab.ylim([-1,8])

    ## Provide a title for the contour plot
    plt.title('ThetaDiffXANDA')

    ## Set x axis label for the contour plot
    plt.xlabel('X')

    ## Set y axis label for the contour plot
    plt.ylabel('A')

    ## Create contour plot - option to provide your own "levelsList" which are
    ## the contour levels inidicated in the colour guide
    if not levelsList:
        cs = plt.contourf(X, A, Z, cmap=plt.cm.get_cmap('hsv'))
    else:
        cs = plt.contourf(X, A, Z, levels=levelsList, cmap=plt.cm.get_cmap('hsv'))

    ## Stuff that makes it look nice - still not that important but can change
    ## how graph looks if a levelsList is provided
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()

    ## Display colour guide for z-axis values on the side
    cbar1 = plt.colorbar(cs)
    cbar1.ax.set_ylabel('Colour Guide for Diff A')

    ## Set z-axis to partial derivative of bending angle in terms of x
    Z = thetaDiffX(X, A, u)

    ## Create contour plot - option to provide your own "levelsList" which are
    ## the contour levels inidicated in the colour guide
    if not levelsList:
        cs = plt.contourf(X, A, Z, cmap=plt.cm.get_cmap('spring'))
    else:
        cs = plt.contourf(X, A, Z, levels=levelsList, cmap=plt.cm.get_cmap('spring'))

    ## Stuff that makes it look nice - still not that important but can change
    ## how graph looks if a levelsList is provided
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()

    ## Display colour guide for z-axis values on the side
    cbar2 = plt.colorbar(cs)
    cbar2.ax.set_ylabel('Colour Guide for Diff X')

    # Display the contour plot - only if we want to
    if show:
        plt.show()


'''Overlaying the Partial Derivatives of Theta and Theta itself'''
def theWholeSummary(upBound=0.95, numOfLevels=10, levelsList=[-0.001, 0, 0.001], u=5, show=True):
    ## Formatting the plot size
    fig, ax = plt.subplots(figsize=(10, 7))

    ## Setting z-axis to the bending angle
    Z = theta(X, A, u)

    ## Set the x axis and y axis limits (minimum and maximum values on axes)
    pylab.xlim([-1,xMax + 1])
    pylab.ylim([-1,8])

    ## Provide a title for the contour plot
    plt.title('ThetaDiffXANDA')

    ## Set x axis label for the contour plot
    plt.xlabel('X')

    ## Set y axis label for the contour plot
    plt.ylabel('A')

    # Create contour plot
    cs = plt.contourf(X, A, Z, levels=np.linspace(upBound * max, max, numOfLevels), cmap=plt.cm.get_cmap('binary'))

    ## Stuff that makes it look nice - still not that important but can change
    ## how graph looks if a levelsList is provided
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()

    ## Display colour guide for z-axis values on the side
    cbar3 = plt.colorbar(cs)
    cbar3.ax.set_ylabel('Colour Guide for Theta')

    ## Also display the partial derivatives
    thetaDiffXAndAGrapher(levelsList=levelsList, show=False, u=u)

    ## Set title to whole graph
    plt.title("The Whole Summary")

    ## Display the plot
    if show:
        plt.show()

'''Graph of the bending angle in 3D'''
def thetaGrapher3D(u=5):
    ## Set up the plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ## Set the z-axis to the bending angle
    Z = theta(X, A, u)

    ## Plot the 3D surface with appropriate axes labelling and titleing
    ax.plot_surface(X, A, Z, cmap='hsv')
    ax.set_xlabel('x')
    ax.set_ylabel('a')
    ax.set_zlabel('theta')

    ## Display the plot
    plt.show()

'''Finding the optimum graphs of x and alpha by computing various graphs of
   theta, and finding the maximum of each one'''
def findOptimumAXUsingTheta():
    ## Initialising lists that will be holding data values
    aList = []
    xList = []

    tempA = 0
    tempX = 0

    ## Loop through a specific range of mu - currently we are going through mu
    ## from 0.1 to 10 in steps of 0.1 (since we start at 1 and divide by 10 to
    ## get 0.1 - and end at 100 and divide by 10 to get 10)
    for u in range(1, 101):
        ## Keep track of progress by printing to terminal
        print(u)
        u /= 10
        u = round(u, 4)

        ## Get the graph of bending angle
        Z = theta(X, A, u)

        ## Find the maximum of this graph - returns the list index not the
        ## actual values of alpha and x
        tempA, tempX = np.where(Z == np.amax(Z))

        ## Add the maximum to the lists holding data values - with a
        ## transformation to the actual values of alpha and x
        aList.append(6.28 * tempA[0] / samples)
        xList.append(xMax * tempX[0] / samples)


    ## Changing data values into plottable collection of data
    aArr = np.asarray(aList)
    xArr = np.asarray(xList)
    ## Setting up axis for mu
    u = np.asarray([round(x / 10, 4) for x in range(1, 101)])

    ## Plot alpha in terms of mu with appropriate labelling and titleing
    plt.plot(u, aArr)
    plt.title("Optimum alpha using graph of theta")
    plt.xlabel('u')
    plt.ylabel('A')
    ## Display plot
    plt.show()

    ## Plot x in terms of mu with appropriate labelling and titleing
    plt.plot(u, xArr)
    plt.title("Optimum x using graph of theta")
    plt.xlabel('u')
    plt.ylabel('X')
    ## Display plot
    plt.show()

'''Finding the optimum graphs of x and alpha by solving the differential system
   in the function called "equations" whilst making "guesses" - these guesses
   are used as the starting point for the solving of the system (the way the
   function solves the system is by taking an iterative Jacobian Approximation,
   so the guesses are where we start the approximation from)'''
def findOptimumAXUsingSystem():
    ## Initialising lists that will be holding data values
    aList = []
    xList = []

    tempA = 0
    tempX = 0

    ## Initial guesses of what the solution is
    guessA = 2
    guessX = 2

    ## Loop through a specific range of mu - currently we are going through mu
    ## from 0.01 to 10 in steps of 0.01 (since we start at 1 and divide by 100
    ## to get 0.01 - and end at 1000 and divide by 100 to get 10)
    for u in range(0, 1001):
        ## Keep track of progress by printing to terminal
        print(u)
        u /= 100
        u = round(u, 4)

        ## Since the solving of the system is just an iterative approximation,
        ## we repeat the approximation 10 times, taking our previous result as
        ## the guess of the next set of iterations (each time fsolve() is called
        ## 1000 iterations are completed)
        for i in range(10):

            tempX, tempA = fsolve(equations, (guessX, guessA), args=(u), maxfev=1000)

            guessX = tempX
            guessA = tempA

        ## Add the data to the lists
        aList.append(tempA)
        xList.append(tempX)


    ## Changing data values into plottable collection of data
    aArr = np.asarray(aList)
    xArr = np.asarray(xList)
    ## Setting up axis for mu
    u = np.asarray([round(x / 100, 4) for x in range(0, 1001)])

    ## Plot alpha in terms of mu with appropriate labelling and titleing
    plt.plot(u, aArr)
    plt.title("Optimum alpha using system")
    plt.xlabel('u')
    plt.ylabel('A')
    ## Display plot
    plt.show()

    ## Plot x in terms of mu with appropriate labelling and titleing
    plt.plot(u, xArr)
    plt.title("Optimum x using system")
    plt.xlabel('u')
    plt.ylabel('X')
    ## Display plot
    plt.show()



def findOptimumAXUsingSystemWithoutGuessing():
    ## Initialising lists that will be holding data values
    aList = []
    xList = []

    tempA = 0
    tempX = 0

    ## Initial guesses of what the solution is
    guessA = 1
    guessX = 1

    ## Loop through a specific range of mu - currently we are going through mu
    ## from 0.1 to 10 in steps of 0.1 (since we start at 1 and divide by 10
    ## to get 0.1 - and end at 100 and divide by 10 to get 10)
    for u in range(0, 101):
        ## Keep track of progress by printing to terminal
        u /= 10
        u = round(u, 4)

        ## Solve system - initial approximation is always (1, 1)
        tempX, tempA = fsolve(equations, (guessX, guessA), args=(u))

        ## Add the data to the lists
        aList.append(tempA)
        xList.append(tempX)


    ## Changing data values into plottable collection of data
    aArr = np.asarray(aList)
    xArr = np.asarray(xList)
    ## Setting up axis for mu
    u = np.asarray([round(x / 10, 4) for x in range(0, 101)])

    ## Plot alpha in terms of mu with appropriate labelling and titleing
    plt.plot(u, aArr)
    plt.title("Optimum alpha using system")
    plt.xlabel('u')
    plt.ylabel('A')
    ## Display plot
    plt.show()

    ## Plot x in terms of mu with appropriate labelling and titleing
    plt.plot(u, xArr)
    plt.title("Optimum x using system")
    plt.xlabel('u')
    plt.ylabel('X')
    ## Displa plot
    plt.show()

'''Plotting points of optimum x and alpha using both methods above to compare'''
def showMaxUsingThetaVsSystem(uList=[round(x/10, 2) for x in range(0, 101)]):
    ## Initial guesses for system approach
    previousSystemGuessX = 1
    previousSystemGuessA = 1

    ## Plotting the points obtained by solving the differential system
    for u in uList:
        ## Idea of repeating the approximation with the previous result being
        ## the guess of the new result
        for i in range(5):
            tempX, tempA = fsolve(equations, (previousSystemGuessX, previousSystemGuessA), args=(u))
            previousSystemGuessX = tempX
            previousSystemGuessA = tempA

        ## Plot the point
        plt.plot(tempX, tempA, marker='p', markersize=5 , color="red", label=str(u))

    ## Plotting the points obtained by computing the graphs of bending angle
    for u in uList:
        ## Get graph of bending angle
        Z = theta(X, A, u)

        ## Find where it is maximum
        tempA, tempX = np.where(Z == np.amax(Z))
        tempX = xMax * tempX[0] / samples
        tempA = 6.28 * tempA[0] / samples

        ## Plot it
        plt.plot(tempX, tempA, marker='o', markersize=5 , color="black")


    ## Apply appropriate labelling and titleing
    plt.title("Comparing Optimum A/X based on theta vs based on system")
    plt.xlabel('X')
    plt.ylabel('A')
    ## Display plot
    plt.show()



## The part of the program that actually executes, the functions that don't
## start with hashtags are called and executed
if __name__ == "__main__":
    #effectOfU()
    #theWholeSummary()
    #thetaGrapher3D()
    #thetaGrapherWithMax(2)
    #findOptimumAXUsingTheta()
    #findOptimumAXUsingSystem()
    #showMaxUsingThetaVsSystem()
