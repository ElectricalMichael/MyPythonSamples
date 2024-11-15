# -*- coding: utf-8 -*-
"""
GPU-accelerated differentiable Finite Element Method (Galerkin Method) solver implementation for computing
the floating potential of an arbitary 2D Mesh placed within an electrostatic field defined by boundary conditions
(Dirichlet and a Robin boundary condition) and a source potential.

This file implements a GPU-accelerated Finite Element Method (FEM) solver for solving a two-dimensional partial differential equation of the Poisson type.
The computation process is implemented in a fully differentiable manner in a Tensorflow Gradienttape to exploit Auto Differentiation capabilities.
The solver outputs the absolute voltage potential for all defined nodes in the mesh. A gradient can be computed for each of these voltage values w.r.t.
any input parameter to the network, e.g., the voltage potential on boundaries or the source potential.

To show-case the power of this differentiable implementation, a gradient-descent-based optimization scheme that optimizes the value of the mesh's Floating Potential 
to a certain target value by minimizing the the mean squared error.

Optimization Level: The implementation supports Graph-based execution, which is already significantly faster than eager-mode execution. XLA acceleration
is not yet supported due to the linear system solver not supporting dynamic shapes in XLA mode.

Challenge: The challenging part was to port a very sequential mathematical algorithm (filling a big matrix one element at the time) to tensorflow instructions
that natively operate on batches of data. Moreover, solving the linear systems even in graph mode does not work straight forward as the shapes of the matrices
and vectors change multiple times dynamically in the process, which is not supported by graph mode. The trick here was to allocate all intermediate-needed 
matrices and vectors with their individual shapes upfront and discarding unused elements at the end.

Created on 01.06.2024.

@author: Michael Petry

Simple use-case: Optimization of a potential difference (source) within the scenario to achieve a specific Floating Potential value.
Optimization performed via iterative Gradient descent.
"""

import meshtools as mt # Low-level library for handling mesh creation and modification in python
import meshpy.triangle as triangle
import tensorflow as tf
import numpy as np
import math;
import matplotlib.pyplot as plt


########################################
####### Definition of the region #######
########################################
# Edge points of the region
length = 0.8
length_edge = 0.4
regionsize = 5
rp1 = [-regionsize,regionsize]  # left, top
rp2 = [-regionsize,-regionsize] # left, bottom
rp3 = [regionsize,-regionsize]  # right, bottom
rp4 = [regionsize,regionsize]   # right, top

# Add potential source "hole" in the region
rph_mcircle = [2.5,2.5] # hole position
rph_circle_radius = 0.3 # hole radius

# Add floating potential hole in the region
floating_length = 0.2 # segment length of the floating potential circle
rph_floating_mcircle = [1.5,2] # center of the floating potential circle
rph_floating_circle_radius = 0.3 # radius of the floating potential circle

# Create line segments with meshtools
m1,v1 = mt.LineSegments(rp1, rp2, edge_length=length)
m2,v2 = mt.LineSegments(rp2, rp3, edge_length=length)
m3,v3 = mt.LineSegments(rp3, rp4, edge_length=length)
m4,v4 = mt.LineSegments(rp4, rp1, edge_length=length)
p,v = mt.AddMultipleSegments(m1,m2,m3,m4)  # combine line segments and generate points and vertices

# Integrate holes into the region
mh1, vh1 = mt.CircleSegments(rph_mcircle, rph_circle_radius, edge_length=length_edge) # Create potential source hole segment
mh2, vh2 = mt.CircleSegments(rph_floating_mcircle, rph_floating_circle_radius, edge_length=floating_length) # Create floating potential hole segment
p,v = mt.AddCurves(p, v, mh1, vh1) # combine potential source hole into region
p,v = mt.AddCurves(p, v, mh2, vh2) # combine floating potential hole into region

poi,tri,BouE,li_BE,bou_elem,CuE,li_CE = mt.DoTriMesh(p,v,edge_length=length, holes=[rph_mcircle, rph_floating_mcircle], show=False) # Triangular mesh generation


##############################################
####### Region Information Exctraction #######
##############################################
# Extract points for
# - Dirichlet Boundary outline
# - Potential source outline
# - Floating Point outline

# Prepare query for boundary points of above components
Ps = [rp1,rp3,rp1,rph_mcircle,rph_mcircle, rph_floating_mcircle, rph_floating_mcircle]
bseg = mt.RetrieveSegments(poi,BouE,li_BE,Ps,['Nodes','Nodes','Nodes', 'Nodes'])

# Plot extracted groups for visual verification
plt.triplot(poi[:, 0], poi[:, 1], tri)
mt.PlotBoundary(poi,bseg[0],'Nodes', 'bo')
mt.PlotBoundary(poi,bseg[1],'Nodes', 'ro')
mt.PlotBoundary(poi,bseg[2],'Nodes', 'yo')
mt.PlotBoundary(poi,bseg[3],'Nodes', 'go')
plt.show()

# Extract boundary points for Potential Source, Outer Mesh Boundary, and Floating Potential
gamma1 = np.append(np.array(bseg[1]), np.array(bseg[0]))    # potential source
gamma2 = np.array(bseg[2])                                  # outer mesh boundary
gammafp = np.array(bseg[3], dtype=np.int32)                 # floating potential

# Create variable aliases for convenience
p = poi
t = tri

###############################################
################# FEM-SOLVER ##################
###############################################

######### Part 1 #########
@tf.function#(jit_compile=True)  
def FEM_PART1(gamma1, gamma2, gammafp, p, t, K, D, VoltVar):
# Only Graph, no XLA acceleration. Linear system solver does not support dynamic shapes in XLA mode
# and message passing between XLA-Graph boundary is not supported for tensorlists.

    # Coefficient functions of the 2D partial differentiable equation
    def alpha1(x, y):
        return 1

    def alpha2(x, y):
        return alpha1(x,y)

    def beta(x, y):
        return 0

    def f(x, y):
        return 0
        

    ## Integrate Dirichlet boundary conditions    
    dD = tf.zeros(len(gamma1) + len(gamma2), dtype=tf.int32)
    PhiR = tf.zeros(shape=dD.shape)
    
    indices1 = tf.range(len(gamma1), dtype=tf.int32)
    dD = tf.tensor_scatter_nd_update(dD, tf.expand_dims(indices1, axis=1), gamma1)
    PhiR = tf.tensor_scatter_nd_update(PhiR, tf.expand_dims(indices1, axis=1), tf.zeros_like(indices1, dtype=tf.float32))

    indices2 = tf.range(len(gamma1), len(gamma1) + len(gamma2), dtype=tf.int32)
    dD = tf.tensor_scatter_nd_update(dD, tf.expand_dims(indices2, axis=1), gamma2)
    PhiR = tf.tensor_scatter_nd_update(PhiR, tf.expand_dims(indices2, axis=1), tf.fill([len(gamma2)], VoltVar))
    
    # Remove multiple same points in dD and PhiR
    dD = tf.sort(dD)
    unique_vals, _ = tf.unique(dD)
    rind = tf.map_fn(
        lambda x: tf.cast(tf.argmax(tf.cast(tf.equal(dD, x), tf.int64)), tf.int32), 
        unique_vals)
    uniquedD = tf.zeros(len(rind), dtype=tf.int32)
    uniquePhiR = tf.zeros(len(rind), dtype=tf.float32)

    # Gather the unique values from dD and PhiR using rind
    uniquedD = tf.tensor_scatter_nd_update(uniquedD, tf.expand_dims(tf.range(len(rind)), axis=1), tf.gather(dD, rind))
    uniquePhiR = tf.tensor_scatter_nd_update(uniquePhiR, tf.expand_dims(tf.range(len(rind)), axis=1), tf.gather(PhiR, rind))

    # Update dD and PhiR
    dD = uniquedD
    PhiR = uniquePhiR

    # Fill matrix K and vector D according to the Galerkin method
    for idx in tf.range(t.shape[0]):
        e = tf.gather(t, idx)
        p1 = p[e[0]]
        x1 = p1[0]
        y1 = p1[1]
        p2 = p[e[1]]
        x2 = p2[0]
        y2 = p2[1]
        p3 = p[e[2]]
        x3 = p3[0]
        y3 = p3[1]
        xm = (x1+x2+x3)/3
        ym = (y1+y2+y3)/3
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1
        TwoDeltaE = c2*b1 - c1*b2
        
        alph_1 = alpha1(xm,ym)
        alph_2 = alpha2(xm,ym)
        bet = beta(xm,ym)
        fxm = f(xm, ym)

        K_local = alph_1/(2*TwoDeltaE) * tf.stack([[b1*b1, b1*b2, b1*b3],[b2*b1, b2*b2, b2*b3],[b3*b1, b3*b2, b3*b3]], axis=0)
        K_local += alph_2/(2*TwoDeltaE) * tf.stack([[c1*c1, c1*c2, c1*c3],[c2*c1, c2*c2, c2*c3],[c3*c1, c3*c2, c3*c3]], axis=0)
        K_local += bet*TwoDeltaE/24 * tf.stack([[2.0, 1, 1], [1, 2, 1], [1, 1, 2]], axis=0)
        
        D_local = fxm * TwoDeltaE/6 * tf.constant([1, 1, 1], dtype=tf.float32)
        
        K11 = K_local[0, 0]
        K22 = K_local[1, 1]
        K33 = K_local[2, 2]
        K12 = K_local[0, 1]
        K13 = K_local[0, 2]
        K23 = K_local[1, 2]

        i1 = e[0]
        i2 = e[1]
        i3 = e[2]
        
        indices = [[i1, i1],[i1, i2],[i1, i3],[i2, i1],[i2, i2],[i2, i3],[i3, i1],[i3, i2],[i3, i3]]
        values = [K11, K12, K13, K12, K22, K23, K13, K23, K33]
        K = tf.tensor_scatter_nd_add(K, indices, values)
        
        D = tf.tensor_scatter_nd_add(D, tf.expand_dims([i1, i2, i3], axis=1), [D_local[0],  D_local[1],  D_local[2]])

    # Matrix K and Vector D is fully filled at this point. Now K and D need to be reduced by integrating the Dirichlet conditions.
    kshapebefore = K.shape[0]   
    num_rows = tf.shape(K)[0]
    all_indices = tf.range(num_rows)
    keep_mask = tf.reduce_all(tf.not_equal(tf.expand_dims(all_indices, axis=1), tf.expand_dims(dD, axis=0)), axis=1)

    # Filter out redundant rows in D
    K = tf.boolean_mask(K, keep_mask)
    num_elements = tf.shape(D)[0]
    all_indices = tf.range(num_elements)
    keep_mask = tf.reduce_all(tf.not_equal(tf.expand_dims(all_indices, axis=1), tf.expand_dims(dD, axis=0)), axis=1)
    D = tf.boolean_mask(D, keep_mask) # Apply the mask to filter out the elements in dD

    kshapeafter = kshapebefore - len(dD)
    
    selected_columns = tf.gather(K, dD, axis=1)
    updates = tf.reduce_sum(tf.expand_dims(PhiR, axis=0) * selected_columns, axis=1)
    indices = tf.expand_dims(tf.range(tf.size(D)), axis=1)
    D = tf.tensor_scatter_nd_add(D, indices, -updates)
    K = tf.boolean_mask(K, keep_mask, axis=1)

    # Integrate the floating potential condition in the matrix K and vector D. Reduce both elements.
    if len(gammafp) > 0:
        gammafp = tf.sort(gammafp)
        fp_newindexes = tf.zeros(len(gammafp), dtype=tf.int32)
        sorted_dirichlet_indexes = tf.sort(dD)
        for ind in range(tf.size(gammafp)):
            pind = tf.gather(gammafp, ind)
            minusoffset = 0
            for dirichletind in sorted_dirichlet_indexes:
                if dirichletind <= pind:
                    minusoffset += 1
                else:
                    break
            fp_newindex = pind - minusoffset
            fp_newindexes = tf.tensor_scatter_nd_update(fp_newindexes, [[ind]], [fp_newindex])
            
        firstIndex = fp_newindexes[0]
        for i in tf.range(1, len(fp_newindexes)):
            K = tf.tensor_scatter_nd_add(K, indices=tf.stack([tf.range(kshapeafter), tf.fill([kshapeafter], firstIndex)], axis=1),
                                          updates=tf.gather(K, fp_newindexes[i], axis=1))
            D = tf.tensor_scatter_nd_add(D, indices=tf.expand_dims([firstIndex], axis=1), updates=[tf.gather(D, fp_newindexes[i], axis=0)])

        all_indices = tf.range(kshapeafter)
        keep_mask = tf.reduce_all(tf.not_equal(tf.expand_dims(all_indices, axis=1), tf.expand_dims(fp_newindexes[1:], axis=0)), axis=1)

        # Apply masks to modify matrix and vector.
        K = tf.boolean_mask(K, keep_mask, axis=1)
        K = tf.boolean_mask(K, keep_mask, axis=0)
        D = tf.boolean_mask(D, keep_mask, axis=0)

    D_solve = tf.expand_dims(D, axis=-1)
    # Return the final matrix K, vector D, and floating point indices to extract the floating potential from the solution
    return K, D_solve, fp_newindexes # Alternatively use tensorlist, but: currently buggy inside TF implementation.


######### Part 2 #########
## System solver part
# I implemented a seperate function in order to separate the linear system solving step from the matrix building part with
# the goal of using the super-fast XLA acceleration for the matrix filling part and graph-mode for part 2, however,
# the XLA-Graph boundary does not support message passing of a tensors between both domains yet, hence, this approach does not work yet.
@tf.function() # Graph acceleration, no XLA
def FEM_PART2(K, D_solve):
    PHI = tf.linalg.solve(K, D_solve)
    return PHI

###############################################
############## FEM-Solver Done ################
###############################################

#### Call the FEM solver for our defined scenario ####

# Declare Inputs / Variables for FEM Program ######
num = len(p)
K = tf.Variable(shape=(num,num), initial_value=tf.zeros((num,num))) # Allocate Matrix K as TF Variable to allow graph execution
D =  tf.Variable(shape=(num), initial_value=tf.zeros((num)))        # Allocate Vector D as TF Variable to allow graph execution
gm1 = tf.constant(gamma1, dtype=tf.int32)           # Convert boundary gamma1 (potential source) to TF constant
gm2 = tf.constant(gamma2, dtype=tf.int32)           # Convert boundary gamma2 (Dirichlet) to TF constant
gmfp = tf.constant(gammafp, dtype=tf.int32)         # Convert boundary gammafp (Floating Potential) to TF constant
ptf = tf.constant(p, dtype=tf.float32)              # Convert mesh points to TF constant
ttf =  tf.constant(t, dtype=tf.int32)               # Convert mesh triangles tuples to TF constant

# Definition of target voltage for Floating Potential
PHI_TARGET = tf.constant([10.])     

# Wrap our FEM program in a gradient tape to calculate the gradient of the loss w.r.t. the source potential voltage
@tf.function(jit_compile=False)
def calcLossGradient(Volt):
    with tf.GradientTape() as tape: # Gradient Tape is TF's auto differentiation. Every **TENSORFLOW-OPERATION** (no numpy) is tracked here.    
        # Step 1: Calculate K-Matrix and D-Vector
        K_new, D_new, fp_newindexes = FEM_PART1(gm1, 
            gm2, 
            gmfp, 
            ptf, 
            ttf, K, D, Volt) 
        # Step 2: Solve Linear System
        retPHI = FEM_PART2(K_new, D_new) # Solve LinAlg
        # Step 3: Extract Floating Potential from Solution
        PhiFloating = retPHI[fp_newindexes[0]]
        # Step 4: Calculate Loss
        loss = tf.reduce_mean(tf.abs(PHI_TARGET - PhiFloating)**2)
        # Step 5: Calculate Gradient of Loss w.r.t. Source Potential Voltage (Volt)
        gradients = tape.gradient(loss, [Volt])
        # Step 6: Return Floating Potential, Loss, and Gradient to outer loop for optimization
        return PhiFloating, loss, gradients



#### Optimization Loop ####
# very primitive manual optimization loop, can easily utilize ADAM or other ML optimizers
Voltage = tf.Variable(5.0, dtype=tf.float32) # Definition of initial source potential voltage as TF variable
print("This program will optimize the source potential voltage to achieve a floating potential of", PHI_TARGET.numpy(), "V. Initial source potential voltage is", Voltage.numpy(), "V.")
for i in range(100): # Loop 100 times in the optimization process
    phi, loss, grad = calcLossGradient(Voltage)
    Voltage.assign(Voltage - grad[0].numpy()*0.1) # Update Voltage iteratively with a constant learning rate, primitive approach
    print("PHI-Float: ", phi[0].numpy(), "Loss: ", loss.numpy(), "Grad: ", grad[0].numpy(), "Source-Voltage: ", Voltage.numpy()) # Display current values
