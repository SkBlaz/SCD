In order to compile, type

make
----------------------------------------

To run the program, type:

./benchmark (-sup) (-inf)
----------------------------------------



Use the option -sup (-inf) if you want to produce a benchmark whose distribution of the ratio of external degree/total degree is superiorly (inferiorly) bounded by the mixing parameter. In other words, if you use one of these options, the mixing parameter is not the average ratio of external degree/total degree (as it used to be) but the maximum (or the minimum) of that distribution. When using one of these options, what the program essentially does is to approximate the external degree always by excess (or by defect) and if necessary to modify the degree distribution. Nevertheless, this last possibility occurs for a few nodes and numerical simulations show that it does not affect the degree distribution appreciably.



The program needs some parameters which can be set editing a file called "parameters.dat".

This file is supposed to contain a list of 6 numbers:

number of nodes
average degree
maximum degree
exponent for the degree distribution
exponent for the community size distribution
mixing parameter, i.e. the average ratio of external degree/total degree for each node.

Optionally, two more numbers will be considered: the minimum and the maximum of the community size range (open parameters.dat for an example). Otherwise, these numbers will be automatically set by the program close to the minimum and maximum degree of the nodes.

For instance, if you want to produce a kind of Girvan-Newman benchmark, you can type:


----------------------------------------

128 	# number of nodes
16	# average degree
16	# maximum degree
1	# exponent for the degree distribution (but, in this case, it does not matter)
1	# exponent for the community size distribution (but, in this case, it does not matter)
0.2	# mixing parameter
32	# minimum for the community sizes
32	# maximum for the community sizes


----------------------------------------


Please note that the community size distribution can be modified by the program to satisfy several constraints (a warning will be displayed).

The program will produce three files:

1) network.dat contains the list of edges (nodes are labelled from 1 to the number of nodes; the edges are ordered and repeated twice, i.e. source-target and target-source).

2) community.dat contains a list of the nodes and their membership (memberships are labelled by integer numbers >=1).

3) statistics.dat contains the degree distribution (in logarithmic bins), the community size distribution, and the distribution of the mixing parameter.



----------------------------------------

Notices: 

-It is sometimes useful to have a graph with no community structure. If this is the case, set the mixing parameter > 1.
-In the file bench_seed.dat you can edit the seed which generates the random numbers. After reading, the program will increase this number by 1 (this is done to generate different networks running the program again and again). If the file is erased, it will be produced by the program again.


