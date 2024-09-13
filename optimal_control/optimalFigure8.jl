#=
Generate a time-optimal figure 8 trajectory and control for quadrotors
Runtime: a few minutes

Copyright 2024 Till Blaha (Delft University of Technology)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=#

using OptimalControl # model definition language
using NLPModelsIpopt # solver
using JLD # import/export the solutions
using Plots # to plot the solution overview


#%% Trajectory parameters

#=
1. The track is made up of gates, seen below for the top loop
2. We model the right hand loop (symmetric track means both lobes are the same)
3. starting and finishing point is at "o" with some initial speed and attitude
4. we assume we must keep in between two circles centered at "."
            x 3
            |
            |
            x
4
x-------x   .   x------x 2

            x 1
            |
            o
            |
            x
  ^ x
  |
  .--> y
=#

gatesize = 1.5  # meters, dimension of square gate
d = 0.5gatesize 
Ri = d   # inner circle
Ro = 3d  # outer circle
Ravg = 0.5(Ri + Ro)

px0 = -Ravg # start point. origin is the center of the circles

GEE = 9.80665


#%% Model parameters
#https://github.com/tudelft/optimal_quad_control_RL/blob/icra2025/randomization.py
# 5inch drone:
pars5 = Dict(
"k_w"=>2.49e-06, "k_x"=>4.85e-05, "k_y"=>7.28e-05, # drag model
"k_p1"=>6.55e-05, "k_p2"=>6.61e-05, "k_p3"=>6.36e-05, "k_p4"=>6.67e-05, # omega^2 effectiveness on roll
"k_q1"=>5.28e-05, "k_q2"=>5.86e-05, "k_q3"=>5.05e-05, "k_q4"=>5.89e-05, # pitch
"k_r1"=>1.07e-02, "k_r2"=>1.07e-02, "k_r3"=>1.07e-02, "k_r4"=>1.07e-02, "k_r5"=>1.97e-03, "k_r6"=>1.97e-03, "k_r7"=>1.97e-03, "k_r8"=>1.97e-03, # omega and omega_dot on yaw
"w_min"=>238.49, "w_max"=>3295.50, "k"=>0.95, "tau"=>0.04 # motor min/max omega, nonlinearity and time constant
)

# COORDINATE SYSTEM IS FRD/NED (Forward-Right-Down body and North-East-Down "inertial")



#%% Model 2

# states (13): position, velocity, attitude, body rates, body accelerations
# controls: 4 normalized motor thrusts

# effectivenss matrix: (fz, wxdot, wydot, wzdot)_steady_state = G * u
Gvv = [
    pars5["w_max"]^2 * [ -pars5["k_w"] , -pars5["k_w"] , -pars5["k_w"],  -pars5["k_w"]  ],
    pars5["w_max"]^2 * [ -pars5["k_p1"], -pars5["k_p2"],  pars5["k_p3"],  pars5["k_p4"] ],
    pars5["w_max"]^2 * [ -pars5["k_q1"],  pars5["k_q2"], -pars5["k_q3"],  pars5["k_q4"] ],
    pars5["w_max"]   * [ -pars5["k_r1"],  pars5["k_r2"],  pars5["k_r3"], -pars5["k_r4"] ],
]
G = reduce(vcat, transpose.(Gvv))
rate_limit = 2000 * pi/180 # 2000 deg/s gyroscope sensing limit

ocp1 = @def begin
# see: https://control-toolbox.org/OptimalControl.jl/stable/tutorial-double-integrator-time.html

    tf ∈ R,                                                         variable
    t ∈ [ 0, tf ],                                                  time
    x = (px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, fzlag, wxdotlag, wydotlag, wzdotlag) ∈ R¹⁷, state
    u = (u1, u2, u3, u4) ∈ R⁴,                                      control

    tf ≥ 0

    # control bounds, assume indepent
    0 ≤ u1(t) ≤ 1 # normalized thrust
    0 ≤ u2(t) ≤ 1 # normalized thrust
    0 ≤ u3(t) ≤ 1 # normalized thrust
    0 ≤ u4(t) ≤ 1 # normalized thrust

    # boundary conditions
    px(0) == px0
    px(tf) == px0
    py(0) == 0
    py(tf) == 0

    vx(0) >= 0 # begin going north-wards
    vx(tf) <= 0 # end going south-wards
    vy(0) >= 0 # go to right at the start
    vy(tf) >= 0 # go to the right at the end
    vz(0) == 0 # no vertical speed at boundaries to make sure it's repeatable
    vz(tf) == 0

    pz(0) - pz(tf) == 0
    vx(0) + vx(tf) == 0
    vy(0) - vy(tf) == 0 # cyclic boundary conditions to ensure continuity between right and (unmodelled) left lobe
    qx(0) + qx(tf) == 0
    qy(0) - qy(tf) == 0 # same holds for rotation
    qz(0) + qz(tf) == 0
    #wx(0) - wx(tf) == 0 # and for rates
    #wy(0) - wy(tf) == 0
    #wz(0) - wz(tf) == 0

    # path constraints
    Ri^2 ≤ (px(t)^2 + py(t)^2) ≤ Ro^2,              (1) # stay within gates in top view
    -d ≤ pz(t) ≤ d,                                 (2) # stay within gates vertically
    #-20 ≤ vx(t) ≤ 20,                               (3)
    #-20 ≤ vy(t) ≤ 20,                               (4)
    # -2 ≤ vz(t) ≤ 2,                                (5)
    ( qw(t)^2 + qx(t)^2 + qy(t)^2 + qz(t)^2 ) == 1, (6) # keep quaternion making sense
    -1 ≤ qw(t) ≤ 1,                                 (7)
    -1 ≤ qx(t) ≤ 1,                                 (8)
    -1 ≤ qy(t) ≤ 1,                                 (9)
    -1 ≤ qz(t) ≤ 1,                                 (10)


    # useful aliases
    fz    = G[1,1] * u1(t)  +  G[1,2] * u2(t)  +  G[1,3] * u3(t)  +   G[1,4] * u4(t)
    wxdot = G[2,1] * u1(t)  +  G[2,2] * u2(t)  +  G[2,3] * u3(t)  +   G[2,4] * u4(t)
    wydot = G[3,1] * u1(t)  +  G[3,2] * u2(t)  +  G[3,3] * u3(t)  +   G[3,4] * u4(t)
    wzdot = G[4,1] * u1(t)  +  G[4,2] * u2(t)  +  G[4,3] * u3(t)  +   G[4,4] * u4(t)

    # Dynamics
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    # https://ahrs.readthedocs.io/en/latest/filters/angular.html#quaternion-derivative
    vxdot = 2(qx(t)*qz(t) + qy(t)*qw(t))*fzlag(t)           # neglect drag
    vydot = 2(qy(t)*qz(t) - qx(t)*qw(t))*fzlag(t)
    vzdot = (1 - 2*qx(t)^2 - 2*qy(t)*qy(t))*fzlag(t) + GEE

    ẋ(t) == [
        vx(t),                                                    # xdot_Inertial = v_Inertial
        vy(t),
        vz(t),
        vxdot,
        vydot,
        vzdot,
        0.5(              - wx(t)*qx(t) - wy(t)*qy(t) - wz(t)*qz(t) ),   # quaternion derivative qdot = 0.5 * (0 w) * q
        0.5( +wx(t)*qw(t)               + wz(t)*qy(t) - wy(t)*qz(t) ),
        0.5( +wy(t)*qw(t) - wz(t)*qx(t)               + wx(t)*qz(t) ),
        0.5( +wz(t)*qw(t) + wy(t)*qx(t) - wx(t)*qy(t)               ),
        wxdotlag(t),
        wydotlag(t),
        wzdotlag(t),
        (fz - fzlag(t)) / pars5["tau"],       # first order model for body accelerations due to motor dynamics
        (wxdot - wxdotlag(t)) / pars5["tau"],
        (wydot - wydotlag(t)) / pars5["tau"],
        (wzdot - wzdotlag(t)) / pars5["tau"],
    ]

    # objective
    ( tf
      + ∫( 0.001*( u1(t)^2 + u2(t)^2 + u3(t)^3 + u4(t)^2 ) + 0.0001*wz(t)^2 ) # to dampen numerical noise on the inputs
    ) → min

end


#%% Initial guess: guess a circle of radius Ravg = 0.5(Ri + Ro)
fz_max = sum(G[1, :])
omega_max = sqrt( sqrt(fz_max^2 - GEE^2) / Ravg )  # a = w^2 r, and a_max = sqrt(fz_max^2 - GEE^2)
omega_init = 1*omega_max
tf_init = 2pi / omega_init

x_init(t) = [
    -Ravg*cos(omega_init*t), Ri*sin(omega_init*t), 0, # init guess: circle of Radius around center of all gates
    +Ravg*sin(omega_init*t), Ri*cos(omega_init*t), 0,
    1, 0, 0, 0, # just upright
    0, 0, 0,
    -GEE, 0, 0, 0, # hover
]

control_init = inv(G) * [-GEE, 0, 0, 0]; # hover

#%% solve and save
sol = solve(ocp1;
    init=(
        state = x_init,
        control = control_init,
        variable = tf_init,
    ),
    grid_size=100,
    max_iter=5000, # default was 1000
    display=true,
)

tf = sol.variable

println("Final Time $tf, Final Objective $(sol.objective)")

# export data
save("optimalFigure8.jld",
        "state", sol.state.x.(range(0, tf, length=251)),
        "control", sol.control.u.(range(0, tf, length=251)),
        "tf", tf,
        "objective", sol.objective)

# auto-plot
plot1 = plot(sol, fmt=:pdf)
savefig(plot1, "optimalFigure8.pdf")
