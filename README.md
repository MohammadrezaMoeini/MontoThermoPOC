# MecGenAI: Generative AI for Mechanical Engineering.
This is an open source genetative AI interperter for FE simulation and optimization.
The objective of this project is to build a platform to read and analyze the finite element method results.




# Quick Start
the first example is to run the 2D strucutre.


# Examples:
## Example 01:
This is a very simple example of linear elastic finite element problem. The structure
can be simply modeled by truss element in 2D. The FE model includes only 4 nodes
and 5 element. We want to minimize the nodal displacement at node 1 and the optimization
variable is the force angle (theta).

I need a FIGURE XXXX

## Example 2:
This example is slightly more complicated truss structure. The FE model includes
11 nodes and 19 element. The force is a vertical concentrated acted force on node 7.
The goal is to minimize the nodal displacement at node 2 by changing the x-position
of nodes {2, 3, 4, 8, 9, 10}.

I need a FIGURE XXXX

## Example 3:
In this example we want to as AI, what is the best approach to optimize this truss
structure. The objective is still to minimize the nodal displacement at node 2. Moreover, the optimization variable is also x-coordinate of the nodal position of nodes {2, 3, 4, 8, 9, 10}.
But here, only one node's position can be changed. Wen want to find out which node
is the best one to be optimized. We (as the user) ask this question from AI.

## Example 4:
This


I need a FIGURE XXXX
