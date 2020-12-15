import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import math
import os
import time
from numba import jit
from constants import *

os.remove('data.txt')

First = True
Fuel = True

def animate(i):
    D = np.loadtxt('data.txt', dtype=float, delimiter='\t')
    ax1.clear()
    ax1.plot(D[:,0], D[:,-1], 'g-')
    ax1.plot(D[-1,0], D[-1,-1], 'ko')

@jit(nopython=True)
def get_g (Height):
    R = 6357000 #m
    return(9.80665*(1-((2*Height))/R))
@jit(nopython=True)
def get_density_s(Height): 
    return (1.315090656*np.exp((-0.0001233931202*Height)))

def get_density(Height): 
    if Height >=0 and Height < 11000: 
        return(1.2250) #deg - K && kg/m^3
    elif Height >= 11000 and Height < 20000:
        return(0.36391)
    elif Height >= 20000 and Height < 32000:
        return(0.08803)
    elif Height >= 32000 and Height < 47000:
        return(0.01322)
    elif Height >= 47000 and Height < 51000:
        return(0.00143)
    elif Height >= 51000 and Height < 71000:
        return(0.00086)
    elif Height >= 71000:
        return(0.000064)

@jit(nopython=True)
def f1(t, x1, x2):  # rate of change of x1 (velocity) wrt time
    reentry_fuel = 40500 + 7776
    A = 43.00840343
    Cd = 0.74
    
    if x1 >= 0:
        current_prop_mass = prop_mass - mfr*t
        M_t =  second_stage_mass + empty_mass + current_prop_mass

        if current_prop_mass < (reentry_fuel):
            T = 0.
        else:
            T = mfr*v_ex

        f = (T/M_t) - get_g(x2) - (get_density_s(x2)/(2*M_t))*Cd*A*(x1**2)

    else:
        T_ = 0.
        M_t = empty_mass + reentry_fuel
        if t > 570 and t < 630:
            print('Re-entry')
            T_ = mfr*v_ex/3.  
            reentry_fuel = reentry_fuel - mfr*(t-570)/3.
            M_t = empty_mass + reentry_fuel
        
        if t >= 630:
            M_t = empty_mass

        if abs(x1) < 150 and t>610:
            print('Parachute')
            T_ = 0
            Cd = 1.75
            A = 5000 # Orion Spacecraft #20000 by calculations

            M_t = empty_mass

        f = (T_/M_t) - get_g(x2) + (get_density_s(x2)/(2*M_t))*Cd*A*(x1**2)

    return f

@jit(nopython=True)
def f2(t, x1):  # rate of change of x2 (position) wrt to time

    f = x1
    return f

if __name__ == '__main__':

    x1 = x1_0
    x2 = x2_0
    t = 0.

    landed = False
    Posn = []
    Vel = []
    Time = []
    Delta_t = []
    
    time.sleep(2)
    iterations = 0

    # Note: Second Index of k is for variable number i.e. 1 for x1 and 2 for x2

    while not landed:

        #time.sleep(0.00001)
        f = open("data.txt", 'a')

        k1_1 = f1(t, x1, x2)
        k1_2 = f2(t, x1)

        k2_1 = f1(t + 0.25*delt, x1 + 0.25*k1_1, x2 + 0.25*k1_2)
        k2_2 = f2(t + 0.25*delt, x1 + 0.25*k1_1)

        k3_1 = f1(t + (3/8)*delt, x1 + (3/32)*k1_1 + (9/32)* k2_1, x2 + (3/32)*k1_2 + (9/32)*k2_2)
        k3_2 = f2(t + (3/8)*delt, x1 + (3/32)*k1_1 + (9/32)*k2_1)

        k4_1 = f1(t + (12/13)*delt, x1 + ((1932/2197)*k1_1) - ((7200/2197)*k2_1) + (7296/2197)*k3_1, x2 + ((1932/2197)*k1_2) - ((7200/2197)*k2_2) + (7296/2197)*k3_2)
        k4_2 = f2(t + (12/13)*delt, x1 + ((1932/2197)*k1_1) - ((7200/2197)*k2_1) + (7296/2197)*k3_1)

        k5_1 = f1(t + delt, x1 + ((439/216) * k1_1) - (8*k2_1) + ((3680/513) * k3_1) - ((845/4104)*k4_1), x2 + ((439/216) * k1_2) - (8*k2_2) + ((3680/513) * k3_2) - ((845/4104)*k4_2))
        k5_2 = f2(t + delt, x1 + ((439/216) * k1_1) - (8*k2_1) + ((3680/513) * k3_1) - ((845/4104)*k4_1))

        k6_1 = f1(t + 0.5*delt, x1 - ((8/27) * k1_1) + (2 * k2_1) - ((3544/2565) * k3_1) + ((1859/4104) * k4_1) - ((11/40)*k5_1), x2 - ((8/27) * k1_2) + (2 * k2_2) - ((3544/2565) * k3_2) + ((1859/4104) * k4_2) - ((11/40)*k5_2))
        k6_2 = f2(t + 0.5*delt, x1 - ((8/27) * k1_1) + (2 * k2_1) - ((3544/2565) * k3_1) + ((1859/4104) * k4_1) - ((11/40)*k5_1))

        x1_ = x1 + (((25/216)*k1_1 + ((1408/2565)*k3_1) + ((2197/4104)*k4_1) - (1/5)*k5_1))*delt
        x1_bar = x1 + (((16/135)*k1_1) + ((6656/12825)*k3_1) + ((28561/56430)*k4_1) - ((9/50)*k5_1) + ((2/55)*k6_1))*delt

        x2_ = x2 + (((25/216)*k1_2 + ((1408/2565)*k3_2) + ((2197/4104)*k4_2) - (1/5)*k5_2))*delt
        x2_bar = x2 + (((16/135)*k1_2) + ((6656/12825)*k3_2) + ((28561/56430)*k4_2) - ((9/50)*k5_2) + ((2/55)*k6_2))*delt

        t = t + delt
        iterations += 1
        error = abs(x1 - x1_bar)

        delt = delt*(tol/error)**0.2

        delt = min(0.05, delt)
        delt = max(1e-6, delt)

        x1 = x1_
        x2 = x2_

        Posn.append(x2)
        Vel.append(x1)
        Time.append(t)
        Delta_t.append(delt)

        f.write(str(t))
        f.write('\t')
        f.write(str(x1))
        f.write('\t')
        f.write(str(x2/1000.))
        f.write('\n')

        #print(t, x2, x1)
        print("delta t = ", delt)
        print(t, x1, x2)
        #print(delt)
        # plt.plot(t, x2)
        # plt.show()

        if x2 < 0:
            landed = True

        if iterations >= 1000000:
            landed = True

        #plt.show()

        f.close()


    Posn_ = np.array(Posn)
    Posn_ *= 0.001
    print(iterations)
    
    plt.plot(Time, Delta_t)
    plt.title(r'$Delta_t \ v/s \ Time$')
    plt.ylabel(r'$Position\ (km) \longrightarrow$')
    plt.xlabel(r'$Time \longrightarrow$')
    plt.yscale('log')
    plt.grid()
    plt.show()
    plt.plot(Time, Vel, 'g-')
    plt.title(r'$Velocity \ (m/s) \ v/s \ Time$')
    #plt.legend(loc='best')
    plt.ylabel(r'$Velocity \longrightarrow$')
    plt.xlabel(r'$Time \longrightarrow$')
    #plt.savefig('problem_2_a_%.2f_%.2f.png'%(total_time, dt), dpi=400)
    plt.grid()
    plt.show()


