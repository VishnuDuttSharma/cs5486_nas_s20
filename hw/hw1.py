"__author__: Vishnu Dutt Sharma"
"__date__: Mar 30, 2020"
'''
Description: This program contains implementations of Steepest Descent and Newton's method. It tries to optimize Booth objective function. 
It produces following plots 
(1) Steepest descent for 4 fixed values of alpha
(2) Newton's menthod for 4 fixed values of alpha
(3) Steepst descent for 3 different strategies for alpha, mentioned in HW1
(4) Newton's Method for 3 different strategies for alpha, mentioned in HW1
(5) Newton's Method for alpha 1
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import markers
np.random.seed(1)

#########################################
###### Function Definitions #############
#########################################

# Defining the range
domain_x1 = np.arange(-10, 10, 0.01)
domain_x2 = np.arange(-10, 10, 0.01)

xx, yy = np.meshgrid(domain_x1, domain_x2, indexing='ij')

domain = np.array([xx,yy])

# Defining the analytical forms for Booth objective function
def booth_func(point):
    """
    Booth objective function
    `point`: a vector (x1, x2)
    Returns a scaler
    """
    x1, x2 = point
    return (x1  + 2 * x2 - 7)**2 + (2* x1 + x2 - 5)**2

def grad_booth_func(point):
    """
    First Gradient of Booth objective function
    `point`: a vector: (x1, x2)
    Returns a vector (1D)
    """
    x1, x2 = point
    return np.array([10*x1 + 8*x2 - 34, 8*x1 + 10*x2 - 38]).T

def hessian_booth_func(point):
    """
    Second Gradient/Hessian of Booth objective function
    `point`: a vector: (x2, x2)
    Returns a 2D matrix
    """
    x1, x2 = point
    return np.array([[10, 8], [8, 10]])


#############################################
######## Simple Optimization Methods ########
#############################################
def steepest_descent(initial_guess, alpha_func, alpha_val=1):
    """
    Steepest Descent Algorithm
    `inital_guess`: x_0
    `alpha_func`: Option to choose value of alpha in each iterations
            "const": alpha remains contant in all iterations. Uses the values set with `alpha_val` argument
            "norm": Uses norm of the gradient of f(x_k) as alpha
            default: Uses 1/(k+1) as alpha
    `alpha_val`: Value of alpha to be used in "const" mode
    Returns:
        integer: 1: Algorithm converged, 0: Algorithm did not converge
        integer: number of steps that algorithm took to reach the point
        np.array: numpy array containing x_k at each step of iteration
    """
    
    # Save x_0 as inital value
    x_arr = [initial_guess]
    
    
    tolerance = 1e-10 # Minimum value of error to stop iterations
    error = -100 # Inital value of error (temporary)
    
    cnt = 0 # Number of iterations
    
    # Run the iterations
    while cnt <= 100:
        
        # Find p_k
        x_k = x_arr[-1]
        grad_x_k = grad_booth_func(x_k)
        p_k = -grad_x_k/np.linalg.norm(grad_x_k) #as per the definition in class
        
        # Find x_k+1 as per the user argument
        if alpha_func == "const":
            x_k_1 = x_k + alpha_val*p_k
        elif alpha_func == "norm":
            x_k_1 = x_k + np.linalg.norm(grad_x_k)*p_k
        else:
            x_k_1 = x_k + (1./(cnt+1))*p_k
        
        # Add updated value of x_k to the list
        x_arr.append(x_k_1)
        
        # Find error
        error = booth_func(x_k) - booth_func(x_k_1)
        
        # If the updated value of x_k goes out of domain, return the array, and status as failure
        if min(x_k_1) < -10 or max(x_k_1) > 10:
            print('Not converged')
            print('Stopping due to going out of domain')
            return 0, len(x_arr)-1, x_arr #in number of steps, exclude inital step. No verification step is involved 
        
        # If error is within tolerance (convergence), return the array, and status as success
        if np.abs(error) < tolerance:
            print('Converged')
            return 1, len(x_arr)-2, x_arr #in number of steps, exclude initial (0th) and last (k+1) step. Last step excluded as it is used only for verification of convergence 
        
        
        # Update number of iterations
        cnt += 1
    
    # If 100 iterations reached, return the array, and status as failure
    print('Not converged')
    print('Stopping as max number iterations reached')
    return 0, 100, x_arr

def newtons_method(initial_guess, alpha_func, alpha_val=1):
    """
    Newton's Method Optimization Algorithm
    `inital_guess`: x_0
    `alpha_func`: Option to choose value of alpha in each iterations
            "const": alpha remains contant in all iterations. Uses the values set with `alpha_val` argument
            "norm": Uses norm of the gradient of f(x_k) as alpha
            default: Uses 1/(k+1) as alpha
    `alpha_val`: Value of alpha to be used in "const" mode
    Returns:
        integer: 1: Algorithm converged, 0: Algorithm did not converge
        integer: number of steps that algorithm took to reach the point
        np.array: numpy array containing x_k at each step of iteration
    """
    
    # Save x_0 as inital value
    x_arr = [initial_guess]
    
    tolerance = 1e-10 # Minimum value of error to stop iterations
    error = -100 # Inital value of error (temporary)
    
    cnt = 0 # Number of iterations
    
    # Run the iterations
    while cnt <= 100:
        x_k = x_arr[-1]
        
        # Find p_k
        grad_pk = grad_booth_func(x_k)
        p_k = np.linalg.solve(hessian_booth_func(x_k), -grad_pk)
        
        # Find x_k+1 as per the user argument
        if alpha_func == "const":
            x_k_1 = x_k + alpha_val*p_k
        elif alpha_func == "norm":
            x_k_1 = x_k + np.linalg.norm(grad_pk)*p_k
        else:
            x_k_1 = x_k + (1./(cnt+1))*p_k
        
        # Add updated value of x_k to the list
        x_arr.append(x_k_1)
        
        # Find error
        error = booth_func(x_k) - booth_func(x_k_1)
        
        # If the updated value of x_k goes out of domain, return the array, and status as failure
        if min(x_k_1) < -10 or max(x_k_1) > 10:
            print('Not converged')
            print('Stopping due to going out of domain')
            return 0, len(x_arr)-1, x_arr #in number of steps, exclude inital step. No verification step is involved 
        
        # If error is within tolerance (convergence), return the array, and status as success
        if error < tolerance:
            print('Converged')
            return 1, len(x_arr)-2, x_arr #in number of steps, exclude initial (0th) and last (k+1) step. Last step excluded as it is used only for verification of convergence 
        
        
        # Update number of iterations
        cnt += 1
    
    # If 100 iterations reached, return the array, and status as failure
    print('Not converged')
    print('Stopping as max number iterations reached')
    return 0, 100, x_arr


################################################
############### Generating Plots ###############
################################################
if __name__== "__main__":
    
    alpha_range = np.array([1e-4, 1e-3, 1e-2, 1e-1])
    status_list = ['No', 'Yes']
    
    ### Steepest Descent for different values of alphas
    print('Steepest Descent for different alphas')
    figs, axes = plt.subplots(2,2, figsize=(12, 10), num="Steepest Descent for different alphas")

    for itr, alpha in enumerate(alpha_range):
        print('Alpha:', alpha)
        status, num_itr, x_arr = steepest_descent(np.array([0,0]), 'const', alpha)

        axes[itr//2, itr%2].contour(domain_x1, domain_x2, booth_func(domain), 50)

        it_array = np.array(x_arr)
        axes[itr//2, itr%2].set_title(r'$\alpha:$'+str(alpha)+' Converged:'+status_list[status])
        axes[itr//2, itr%2].plot(it_array.T[0], it_array.T[1], "x-")

        axes[itr//2, itr%2].plot(1,3, "g*", alpha=0.5, markersize=14)
        
        
        print('#Iterations:', num_itr)
        print('Error:', booth_func(x_arr[-2])-booth_func(x_arr[-1]))
        print('')

    plt.show()
    print('#'*15)
    
    
    # Newton's Method for different values of alpha
    print("Newton's Method for different alpha")
    figs, axes = plt.subplots(2,2, figsize=(12, 10), num="Newton's Method for different alphas")

    for itr, alpha in enumerate(alpha_range):
        print('Alpha:', alpha)
        status, num_itr, x_arr = newtons_method(np.array([0,0]), 'const', alpha)

        axes[itr//2, itr%2].contour(domain_x1, domain_x2, booth_func(domain), 50)

        it_array = np.array(x_arr)
        axes[itr//2, itr%2].set_title(r'$\alpha:$'+str(alpha)+' Converged:'+status_list[status])
        axes[itr//2, itr%2].plot(it_array.T[0], it_array.T[1], "x-")

        axes[itr//2, itr%2].plot(1,3, "g*", alpha=0.5, markersize=14)
        
        
        print('#Iterations:', num_itr)
        print('Error:', booth_func(x_arr[-2])-booth_func(x_arr[-1]))
        print('')

    plt.show()
    print('#'*15)
    
    ###### Steepest Descent for different methods of choosing alpha
    alpha_range = np.array(["const", "norm", "count"])
    
    print('Steepest Descent for different methods of choosing alpha')
    figs, axes = plt.subplots(1,3, figsize=(12, 3), num="Steepest Descent for different methods of choosing alpha")

    for itr, alpha in enumerate(alpha_range):
        print('Mode:', alpha)
        status, num_itr, x_arr = steepest_descent(np.array([0,0]), alpha_func=alpha, alpha_val=1.)

        axes[itr].contour(domain_x1, domain_x2, booth_func(domain), 50)
        it_array = np.array(x_arr)
        axes[itr].set_title(r'$\alpha:$'+str(alpha)+', Converged:'+status_list[status])

        axes[itr].plot(it_array.T[0], it_array.T[1], "x-")

        axes[itr].plot(1,3, "g*", alpha=0.5, markersize=14)
        
        print('#Iterations:', num_itr)
        print('Error:', booth_func(x_arr[-2]) - booth_func(x_arr[-1]))
        print('')
    
    plt.show()
    print('#'*15)
    
    
    ###### Newton's method for different methods of choosing alpha
    print("Newton's method for different methods of choosing alpha")
        
    figs, axes = plt.subplots(1,3, figsize=(12, 3), num="Newton's Method for different methods of choosing alpha")

    for itr, alpha in enumerate(alpha_range):
        print('Mode:', alpha)
        status, num_itr, x_arr = newtons_method(np.array([0,0]), alpha_func=alpha, alpha_val=1.)

        axes[itr].contour(domain_x1, domain_x2, booth_func(domain), 50)

        it_array = np.array(x_arr)
        axes[itr].set_title(r'$\alpha:$'+str(alpha)+', Converged:'+status_list[status])
        axes[itr].plot(it_array.T[0], it_array.T[1], "x-")

        axes[itr].plot(1,3, "g*", alpha=0.5, markersize=14)
        
        print('#Iterations:', num_itr)
        print('Error:', booth_func(x_arr[-2]) - booth_func(x_arr[-1]))
        print('')
    
    plt.show()
    print('#'*15)
    
    ###### Newton's method for alpha = 1
    print("Newton's method for alpha = 1")
    status, num_itr, x_arr = newtons_method(np.array([0,0]), alpha_func='const', alpha_val=1.)
    
    fig = plt.figure("Newton's Method for alpha=1")
    plt.contour(domain_x1, domain_x2, booth_func(domain), 50)

    it_array = np.array(x_arr)
    plt.title(r'$\alpha:$'+str(alpha)+', Converged:'+status_list[status])
    plt.plot(it_array.T[0], it_array.T[1], "x-")

    plt.plot(1,3, "g*", alpha=0.5, markersize=14)
    
    print('#Iterations:', num_itr)
    print('Error:', booth_func(x_arr[-2]) - booth_func(x_arr[-1]))
    print('')
    
    plt.show()
    print('#'*15)
    
