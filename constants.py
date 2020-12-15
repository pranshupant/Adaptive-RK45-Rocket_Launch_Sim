x1_0 = 0.
x2_0 = 0.
delt = 0.1

v_ex = 3241.22 # Will empty the tanks in 162s.
mfr = 2430.0  # For all Engines combined

T = mfr*v_ex
T_ = 0.


empty_mass = 25600
prop_mass = 395700
payload = 22800
second_stage_mass = 96570.+ 22800. # Dead Mass to be dropped at stage separation

M_l = empty_mass + prop_mass + second_stage_mass # Launch Mass

dead_mass = 96570

reentry_fuel = 40500 + 7776

A = 43.00840343
Cd = 0.74
g = 9.81
ro_0 = 1.225
R = 287
Temp = 300
tol = 1e-2