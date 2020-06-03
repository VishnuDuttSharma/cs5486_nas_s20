"__author__: Vishnu Dutt Sharma"
"__date__: Mar 30, 2020"
'''
Description: This program contains implementations of Steepest Descent and Newton's method in 
conjunction with the zoom and bracket methods. It tries to optimize Booth objective function. 
It produces plots for each of the above mentioned methods (2 plots total).
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


################################################
######## Optimization Methods with zoom ########
################################################
def steepest_descent_with_zoom(initial_guess):
    """
    Steepest Descent Algorithm with Zoom Line Search Method
    `inital_guess`: x_0
    Returns:
        integer: 1: Algorithm converged, 0: Algorithm did not converge
        integer: number of steps that algorithm took to reach the point
        np.array: numpy array containing x_k at each step of iteration
    """
    # Save x_0 as inital value
    x_arr = [initial_guess]
    
    tolerance = 1e-10 # Minimum value of error to stop iterations
    error = -100 # Inital value of error (temporary)
    alpha_max = 5 # Max value of alpha
    
    c1 = 1e-4  # c1: Constant for Armijo condition
    c2 = 0.9  # c2: Constant for curvature condition
    
    
    cnt = 0 # Number of iterations
    
    # Run the iterations
    while cnt <= 100:
        
        # Find p_k
        x_k = x_arr[-1]
        grad_x_k = -grad_booth_func(x_k)
        p_k = grad_x_k/np.linalg.norm(grad_x_k) # as per definition in class
        
        # Getting alpha using zoom line search method
        alpha_val = line_search(x_k, p_k, alpha_max, c1, c2)
        
        # Calculating new x_k (x_k+1)
        x_k_1 = x_k + alpha_val*p_k
        
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
    return 0, x_arr

def newtons_method_with_zoom(initial_guess):
    """
    Newton's method for optmization with Zoom Line Search Method
    `inital_guess`: x_0
    Returns:
        integer: 1: Algorithm converged, 0: Algorithm did not converge
        integer: number of steps that algorithm took to reach the point
        np.array: numpy array containing x_k at each step of iteration
    """
    # Save x_0 as inital value
    x_arr = [initial_guess]
    
    tolerance = 1e-10 # Minimum value of error to stop iterations
    error = -100 # Inital value of error (temporary)
    alpha_max = 5 # Max value of alpha
    
    c1 = 1e-4  # c1: Constant for Armijo condition
    c2 = 0.9  # c2: Constant for curvature condition
    
    
    cnt = 0 # Number of iterations
    
    # Run the iterations
    while cnt <= 100:
        
        # Find p_k
        x_k = x_arr[-1]
        
        grad_pk = grad_booth_func(x_k)
        p_k = np.linalg.solve(hessian_booth_func(x_k), -grad_pk)
        
        
        # Getting alpha using zoom line search method
        alpha_val = line_search(x_k, p_k, alpha_max, c1, c2)
        
        # Calculating new x_k (x_k+1)
        x_k_1 = x_k + alpha_val*p_k
        
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
    return 0, x_arr


##############################################################
######## Line search and zoom method for step length #########
##############################################################
def line_search(x_k, p_k, alpha_max, c1, c2):
    """
    Line search method based on Algorithm 3.5, for finding good step length(alpha)
    `x_k`: x_k, current estimate of the minimizer
    `p_k`: p_k, current estimate of the direction to optimizer
    `alpha_max`: max value of alpha
    `c1`:  Constant for Armijo condition
    `c2`:  Constant for curvature condition
    Returns: 
        float: a good value of step length (alpha) for given x_k and p_k
    """
    
    a_0 = 0 # Inital value of alpha (alpha_0)
    
    phi_zero = booth_func(x_k + a_0*p_k) # Function value at alpha_0
    grad_phi_zero = grad_booth_func(x_k + a_0*p_k) @ p_k # Directional Gradient of function at alpha_0 in direction of p_k
    
    # Add intial value to array
    a_arr = [a_0]
    
    # Set a_i to a random value in range [0, alpha_max]
    a_i = np.random.random_sample()*alpha_max # current alpha
    
    # Add current alpha(a_i) to array 
    a_arr.append(a_i)
    
    # Counter/number of iterations
    i = 1
    
    phi_a_i_1 = phi_zero # value of function at previous alpha (i.e. at alpha_{i-1})
    a_i_1 = a_0 # previous value of alpha (alpha_{i-1})
    
    # Repeat the loop till a stop condition is met
    while True:
        # Find function value at next x_k (i.e. at x_{k+1}
        phi_a_i = booth_func(x_k + a_i*p_k)
        
        # First Wolfe condition/Armijo Condition. Use zoom to find good alhpa between previous and current alpha
        if phi_a_i > (phi_zero + c1*a_i*grad_phi_zero) or (phi_a_i >= phi_a_i_1 and i > 1):
            return zoom_func(x_k, p_k, a_i_1, a_i, c1, c2)
        
        # Find directional gradient at x_{k+1} 
        grad_phi_a_i = grad_booth_func(x_k + a_i*p_k) @ p_k
        
        # Second Wolfe condition/Curvature condition
        if (np.abs(grad_phi_a_i) <= -c2*grad_phi_zero):
            return a_i
        
        # If direction gradient is positive, Use zoom to find good alhpa between current alpha and alpha_max
        if grad_phi_a_i >= 0:
            return zoom_func(x_k, p_k, a_i, alpha_max, c1, c2)
        
        # Save current alpha and functional value as previous values
        a_i_1 = a_i
        phi_a_i_1 = phi_a_i
        
        # Add current alpha(a_i) to array 
        a_i = (alpha_max - a_i)*np.random.random_sample() + a_i
        
        # Update number of iterations
        i += 1
        

def zoom_func(x_k, p_k, a_lo, a_hi, c1, c2):
    """
    Zoom method based on Algorithm 3.6, for finding good step length(alpha)
    `x_k`: x_k, current estimate of the minimizer
    `p_k`: p_k, current estimate of the direction to optimizer
    `alpha_max`: max value of alpha
    `c1`:  Constant for Armijo condition
    `c2`:  Constant for curvature condition
    Returns: 
        float: a good value of step length (alpha) for given x_k and p_k
    """
    
    j = 0 # Number of iterations
    
    # Repeat till a good value fo alpha is found
    while True:
    
    # Using bisection to find the value in between alpha_low and alpha_high
        a_j = (a_lo + a_hi)/2
        
        phi_a_j = booth_func(x_k + a_j * p_k) # Function value at current alpha, x_k, p_k
        phi_0 = booth_func( x_k + 0 * p_k ) # Function values at alpha=0, x_k, p_k
        grad_phi_0 = grad_booth_func( x_k + 0 * p_k) @ p_k # Direction gradient at alpha=0

        phi_a_lo = booth_func(x_k + a_lo * p_k) # Function value at alpha_low, x_k, p_k
        
        # If Armijo condition fails or if function value at current alpha is greater than alpha_low, update alpha_high
        if (phi_a_j > phi_0+c1*a_j*grad_phi_0) or (phi_a_j >= phi_a_lo):
            a_hi = a_j
        
        else:
            grad_phi_a_j = grad_booth_func(x_k + a_j * p_k) @ p_k # Directional gradient at a_j, x_k, p_k
            
            # If Second Wolfe condition/Curvature condition is met then return current alpha
            if np.abs(grad_phi_a_j) <= -c2*grad_phi_0:
                return a_j
            
            # if directional gradient and difference between alpha_high and alpha_low have same sign, update alpha_high to alpha_low
            if grad_phi_a_j*(a_hi - a_lo) >= 0:
                a_hi = a_lo
            
            # Update alpha_low to current alpha 
            a_lo = a_j
        
        # Update the number of iterations
        j += 1
        

################################################
############### Generating Plots ###############
################################################
if __name__== "__main__":
    status_list = ['No', 'Yes']
    ###### Steepest Descent with Zoom Line Search
    print('Steepest Descent with Zoom Line Search')
    status, num_itr, x_arr = steepest_descent_with_zoom(np.array([0,0]))
    
    fig = plt.figure("Steepest Descent with Zoom")
    plt.contour(domain_x1, domain_x2, booth_func(domain), 50)
    
    it_array = np.array(x_arr)
    
    plt.title(r'Converged:'+status_list[status])
    plt.plot(it_array.T[0], it_array.T[1], "x-")

    plt.plot(1,3, "g*", alpha=0.5, markersize=14)

    print('#Iterations:', num_itr)
    print('Error:', booth_func(x_arr[-2]) - booth_func(x_arr[-1]))
    
    plt.show()
    print('#'*15)
    
    ###### Newton's Method with Zoom Line Search
    print("Newton's Method with Zoom Line Search")
    status, num_itr, x_arr = newtons_method_with_zoom(np.array([0,0]))
    
    fig = plt.figure("Newton's Method with Zoom")
    plt.contour(domain_x1, domain_x2, booth_func(domain), 50)
    
    it_array = np.array(x_arr)
    
    plt.title(r'Converged:'+status_list[status])
    plt.plot(it_array.T[0], it_array.T[1], "x-")

    plt.plot(1,3, "g*", alpha=0.5, markersize=14)

    print('#Iterations:', num_itr)
    print('Error:', booth_func(x_arr[-2]) - booth_func(x_arr[-1]))
    
    plt.show()
    print('#'*15)
    
    