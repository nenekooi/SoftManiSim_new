import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def moving_rod(q, u_0, f):
    # Length of the rod
    l = 100e-3

    # Physical parameters of the rod
    E = 70e9
    r1 = (1.5) * 1e-3
    r2 = (1) * 1e-3
    J = np.pi / 2 * (r1 ** 4 - r2 ** 4)
    I = 0.5 * J
    G = 10e9

    # Pre-curvature of the rod
    Ux = 0
    Uy = 0
    Uz = 0

    q_0 = np.array([0, 0])
    B = q[0] + q_0[0]  # Length of the rod before template

    # Initial angles
    alpha = q[1] + q_0[1]

    # Initial conditions
    r0 = np.array([0, 0, 0])
    R0 = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ]).reshape(9)
    du0 = np.eye(3).reshape(9)
    s_span = np.linspace(0, l + B, 100)

    y0 = np.zeros(33)
    y0[0:3] = u_0
    y0[3:15] = np.concatenate([r0, R0])
    y0[15:24] = du0

    sol = odeint(ode, y0, s_span, args=(Ux, Uy, Uz, E * I, G * J, f, u_0))
    shape = sol[:, 3:6]
    r = shape
    u = sol[-1, 0:3]
    dU0 = sol[-1, 15:24].reshape((3, 3))

    return r, u, dU0


def ode(y, s, Ux, Uy, Uz, EI, GJ, f, u_0):
    dydt = np.zeros(33)

    # Extract the state variables
    ux, uy, uz = y[0:3]
    U = np.array([ux, uy, uz])


    # Stiffness matrix
    K = np.array([[0.0022, 0, 0], [0, 0.0022, 0], [0, 0, 0.001]])
    u_hat = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    e3 = np.array([0, 0, 1])
    e3_hat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    Us = np.array([Ux, Uy, Uz])

    R = y[6:15].reshape((3, 3))
    B = y[15:24].reshape((3, 3))
    C = y[24:33].reshape((3, 3))

    # ODEs
    D = np.dot(R.T, f)
    C_hat = np.array([
        [0, -D[2], D[1]],
        [D[2], 0, -D[0]],
        [-D[1], D[0], 0]
    ])
    dC = np.dot(C_hat, B) - np.dot(u_hat, C)

    A = np.dot(K, U - Us)
    A_hat = np.array([
        [0, -A[2], A[1]],
        [A[2], 0, -A[0]],
        [-A[1], A[0], 0]
    ])
    dB = np.linalg.inv(K).dot(-A_hat.dot(B) + u_hat.dot(K).dot(B) + e3_hat.dot(C))

    du = -np.linalg.inv(K).dot((u_hat).dot(K).dot(U - Us) + e3_hat.dot(np.dot(R.T, f)))
    dr = np.dot(R, e3)
    dR = np.dot(R, u_hat)

    dydt[0:3] = du
    dydt[3:6] = dr
    dydt[6:15] = dR.T.reshape(9)
    dydt[15:24] = dB.T.reshape(9)
    dydt[24:33] = dC.T.reshape(9)

    return dydt


# Main loop
dt = 0.01  # Sampling time in seconds
t_total = 30  # Simulation time in seconds
t = np.arange(0, t_total + dt, dt)

# Initializing simulation parameters
u_0 = np.array([0, 0, 0])  # Initial Guess for rod's curvature at its base
P = np.eye(3)
r_end = []  # Cartesian coordinates of the rod backbone
u_end = []  # Instantaneous curvature of the rod's tip

# Set up the 3D plot
fig = plt.figure( dpi=300)  
ax = fig.add_subplot(111, projection='3d')

for i in range(len(t)):
    V = 0
    q = np.array([0, 0])  # Rods insertion and rotation
    f = np.array([0.1, 0., 0])  # External Force

    # Solving rod's model
    r, u, B = moving_rod(q, u_0, f)

    if i % 15 == 0:
        # Plotting rod's backbone in 3D
        ax.plot(r[:, 0], r[:, 1], r[:, 2], 'k', linewidth=1)

    A = V * np.eye(3)
    Q = 30 * np.eye(3)
    R = 120 * np.eye(3)

    # Observer equations
    dP = -A.T @ P - P @ A - P @ B @ np.linalg.inv(R) @ B.T @ P + Q
    du0 = -np.linalg.inv(R) @ B.T @ P @ u
    P = P + dt * dP
    u_0 = u_0 + dt * du0

    # Recording rod's tip position
    r_end.append(r[-1, :])

r_end = np.array(r_end)

# Final plot
ax.plot(r[:, 0], r[:, 1], r[:, 2], 'b', linewidth=3)
ax.plot(r_end[:, 0], r_end[:, 1], r_end[:, 2], 'r', linewidth=3)
ax.legend(['Rod Backbone', 'Rod Tip Trajectory'], loc='upper left')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_xlim([-0.05, 0.05])
ax.set_ylim([-0.05, 0.05])
ax.set_zlim([-0.0, 0.15])
ax.grid(True)
plt.show()