import Sofa
import Sofa.Core
import Sofa.Simulation
import numpy as np
from scipy.integrate import solve_ivp

class ODE():
    def __init__(self) -> None:
        self.l  = 0
        self.uy = 0
        self.ux = 0
        self.dp = 0
        self.err = np.array((0,0,0))
        self.errp = np.array((0,0,0))

        self.simCableLength  = 0
        # initial length of robot
        self.l0 = 2000e-3
        # cables offset
        self.d  = 7.5e-3
        # ode step time
        self.ds     = 0.001 #0.0005  
        r0 = np.array([0,0,0]).reshape(3,1)  
        R0 = np.eye(3,3)
        R0 = np.reshape(R0,(9,1))
        y0 = np.concatenate((r0, R0), axis=0)
        self.states = np.squeeze(np.asarray(y0))
        self.y0 = np.copy(self.states)


    def updateAction(self,action=np.array([0,0,0])):
        self.l  = self.l0 + action[0]        
        self.uy = (action[1]) /  (self.l * self.d)
        self.ux = (action[2]) / -(self.l * self.d)
    
      
    def _reset_y0(self):
        r0 = np.array([0,0,0]).reshape(3,1)  
        R0 = np.eye(3,3)
        R0 = np.reshape(R0,(9,1))
        y0 = np.concatenate((r0, R0), axis=0)
        self.l0 = 2000e-3       
        self.states = np.squeeze(np.asarray(y0))
        self.y0 = np.copy(self.states)
        


    def odeFunction(self,s,y):
        dydt  = np.zeros(12)
        # % 12 elements are r (3) and R (9), respectively
        e3    = np.array([0,0,1]).reshape(3,1)              
        u_hat = np.array([[0,0,self.uy], [0, 0, -self.ux],[-self.uy, self.ux, 0]])
        r     = y[0:3].reshape(3,1)
        R     = np.array( [y[3:6],y[6:9],y[9:12]]).reshape(3,3)
        # % odes
        dR  = R @ u_hat
        dr  = R @ e3
        dRR = dR.T
        dydt[0:3]  = dr.T
        dydt[3:6]  = dRR[:,0]
        dydt[6:9]  = dRR[:,1]
        dydt[9:12] = dRR[:,2]
        return dydt.T


    def odeStepFull(self):        
        cableLength          = (0,self.l)
        # METHODS = {'RK23': RK23,
        #         'RK45': RK45,
        #         'DOP853': DOP853,
        #         'Radau': Radau,
        #         'BDF': BDF,
        #         'LSODA': LSODA}
        t_eval               = np.linspace(0, self.l, int(self.l/self.ds))
        sol                  = solve_ivp(self.odeFunction,cableLength,self.y0,method='RK45',t_eval=t_eval)
        self.states          = np.squeeze(np.asarray(sol.y[:,-1]))
        return sol.y


class RobotController(Sofa.Core.Controller):
    """ A controller to update robot state in SOFA using the ODE class """
    def __init__(self, node, mechanicalObject, body,nBody, visual,nShape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node = node
        # self.mechanicalObject = mechanicalObject
        self._body = body
        self._visual = visual
        
        self._ode = ODE()
        self._action = np.array([0.001,0,0.0])
        self._ode.updateAction()
        self.timeSinceLastUpdate = 0.0
        self._number_of_sphere_body = nBody
        self._number_of_sphere_visual = nShape
        self._number_of_segment = 2
        print("-----------------")
       
    def onAnimateBeginEvent(self, event):
        dt = self.node.getRoot().dt.value
        self.timeSinceLastUpdate += dt
        t = self.timeSinceLastUpdate
        
        
        cable_1   = 0.01*np.sin(0.5*np.pi*t)
        cable_2   = 0.005*np.cos(1*np.pi*t)
        cable_3   = 0.01*np.sin(0.4*np.pi*t)
        cable_4   = 0.003*np.cos(1*np.pi*t)
        cable_5   = 0.01*np.sin(0.4*np.pi*t)
        cable_6   = 0.003*np.cos(1*np.pi*t)
        
        action=np.array([0, cable_1, cable_2, 
                         0, cable_3, cable_4,
                         0, cable_5, cable_6])
        
        if (np.shape(action)[0]<self._number_of_segment*3):
            action = np.concatenate((np.zeros((self._number_of_segment*3)-np.shape(action)[0]),action),axis=0) 
        
        self._ode._reset_y0()
        sol = None
        for n in range(self._number_of_segment):
            # self._ode._update_l0(self,l0)
            self._ode.updateAction(action[n*3:(n+1)*3])
            sol_n = self._ode.odeStepFull()
            self._ode.y0 = sol_n[:,-1]        
            
            if sol is None:
                sol = np.copy(sol_n)
            else:                
                sol = np.concatenate((sol,sol_n),axis=1)
        
        
        idx_body = np.linspace(0, sol.shape[1] - 1, self._number_of_sphere_body, dtype=int)
        idx_shape = np.linspace(0, sol.shape[1] - 1, self._number_of_sphere_visual, dtype=int)

        for i in range(self._number_of_sphere_body):
            positions = self._body[i].mstate.position.value
            
            # Create a new array for updated positions
            new_positions = np.copy(positions)
            _idx = idx_body[i]
            new_positions[0][:3] = sol[:3,_idx]*5  # Increment the first point's X coordinate
            self._body[i].mstate.position.value = new_positions
        
        for i in range(self._number_of_sphere_visual):
            positions = self._visual[i].getObject('MechanicalObject').position.value
            # Create a new array for updated positions
            new_positions = np.copy(positions)
            _idx = idx_shape[i]
            new_positions[0][:3] = sol[:3,_idx]*5  # Increment the first point's X coordinate
            self._visual[i].getObject('MechanicalObject').position.value = new_positions
            
    
    def onKeypressedEvent(self, event):
        key = event['key']
        # Example of action based on key press
        if key == 'Z':  # Increase some parameter
            self._action[0]+= 0.005
            self._action[1]+= 0.002
        elif key == 'X':  # Decrease the same parameter
            self._action[0]-= 0.005
            self._action[1]-= 0.002
            print(f"Key pressed: {key}, action z: {self._action[0]}")
        
            
      
def createScene(rootNode):
    pluginList = ['Cosserat',
              'Sofa.Component.AnimationLoop',
              'Sofa.Component.LinearSolver.Iterative', 'Sofa.Component.LinearSolver.Direct', 'Sofa.Component.StateContainer',
              'Sofa.Component.SolidMechanics.FEM.Elastic', 'Sofa.Component.SolidMechanics.Spring', 'Sofa.Component.MechanicalLoad', 'Sofa.Component.Mass',
              'Sofa.Component.Topology.Container.Dynamic', 'Sofa.Component.Topology.Container.Grid',
              'Sofa.Component.ODESolver.Backward',
              'Sofa.Component.Mapping.Linear', 'Sofa.Component.Mapping.NonLinear', 'Sofa.Component.Topology.Mapping',
              'Sofa.Component.Collision.Geometry', 'Sofa.Component.Collision.Detection.Algorithm','Sofa.Component.Collision.Detection.Intersection','Sofa.Component.Collision.Response.Contact',
              'Sofa.Component.Constraint.Lagrangian.Correction','Sofa.Component.Constraint.Lagrangian.Solver',
              'Sofa.Component.Engine.Select',
              'Sofa.GL.Component.Rendering3D', 'Sofa.Component.Setting', 'Sofa.Component.SceneUtility', 'Sofa.Component.IO.Mesh', 'Sofa.Component.Visual',
              'SofaPython3', 'Sofa.GL.Component.Rendering2D', 'Sofa.GL.Component.Rendering3D','Sofa.GL.Component.Shader'
              ]
    
    
    rootNode.addObject('RequiredPlugin', pluginName=[pluginList] )
    rootNode.addObject("VisualGrid", nbSubdiv=10, size=1000)
    rootNode.addObject('VisualStyle', displayFlags='showVisual showWireframe')
    rootNode.addObject('Camera', position="-35 0 280", lookAt="0 0 0")

    
    # Collision pipeline
    rootNode.addObject('DefaultPipeline')
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('GenericConstraintSolver', tolerance="1e-6", maxIterations="1000")
    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('RuleBasedContactManager', responseParams="mu="+str(0.0), name='Response', response='FrictionContactConstraint')
    rootNode.addObject('LocalMinDistance', alarmDistance=10, contactDistance=5, angleCone=0.01)

    rootNode.gravity=[0.0,0,0.0]
    rootNode.dt = 0.01  # Simulation time step
    
    confignode = rootNode.addChild("Config")
    confignode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")

    # Create a solver
    rootNode.addObject('RungeKutta4Solver', name="odeExplicitSolver")
    rootNode.addObject('CGLinearSolver', iterations="100", tolerance="1e-5", threshold="1e-5")
    

    body = []
    nbody = 70
    visual = []
    
    for i in range(nbody):
        
        body.append(rootNode.addChild(f"sphere{i}"))
        body[-1].addObject('MechanicalObject', name=f"mstate", template="Rigid3", translation2=[i, 0., 0.], rotation2=[0., 0., 0.], showObjectScale=50)

        #### Visualization subnode for the sphere
        visual.append(body[-1].addChild("VisualModel"))
        visual[-1].loader = visual[-1].addObject('MeshOBJLoader', name=f"loader{i}", filename="mesh/sphere.obj")
        # visual[-1].loader = visual[-1].addObject('MeshOBJLoader', name=f"loader{i}", filename="mesh/cube.obj")
        
        if i == nbody-1:
            visual[-1].addObject('OglModel', name=f"model{i}", src=f"@loader{i}", scale3d=[0.022*5]*3, color=[0., 0, 0.75], updateNormals=False)
        else:
            visual[-1].addObject('OglModel', name=f"model{i}", src=f"@loader{i}", scale3d=[0.018*5]*3, color=[0.5, .0, 0.6], updateNormals=False)
        visual[-1].addObject('RigidMapping')
        
        if i == nbody-1:
            #### Collision subnode for the sphere
            collision = body[-1].addChild('collision')
            collision.addObject('MeshOBJLoader', name="loader", filename="mesh/sphere.obj", triangulate="true", scale3d=[0.022*5]*3)
            collision.addObject('MeshTopology', src="@loader")
            collision.addObject('MechanicalObject')
            collision.addObject('TriangleCollisionModel')
            collision.addObject('LineCollisionModel')
            collision.addObject('PointCollisionModel')
            collision.addObject('RigidMapping')
        
    
    visual_arr = []
    nShape = 2
    
    for i in range (nShape):
        visual_arr.append(rootNode.addChild(f"VNode{i}"))
        visual_arr[-1].addObject("MechanicalObject", template="Rigid3d", position=f"{i} 0 0   0 0 0 1", showObject="1")
    
    controller = RobotController(body[-1], body[-1].getObject('MechanicalObject'), body,nbody,visual_arr,nShape)
    body[-1].addObject(controller)


    return rootNode
