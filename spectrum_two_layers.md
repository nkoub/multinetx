
#### Import standard libraries


    import numpy as np
    import matplotlib.pylab as plt
    %matplotlib inline

#### Import the package MultiNetX


    import multinetx as mx

#### Create two Erd"os- R'enyi networks with N nodes for each layer


    N = 20
    g1 = mx.barabasi_albert_graph(N,2,seed=231)
    g2 = mx.barabasi_albert_graph(N,3,seed=231)

#### Create an 2Nx2N lil sparse matrix for interconnecting the two layers


    adj_block = mx.lil_matrix(np.zeros((N*2,N*2)))

#### Define the type of interconnection between the layers (here we use identity matrices thus connecting one-to-one the nodes between layers)


    adj_block[:N,N:] = np.identity(N)    # L_12
    adj_block += adj_block.T

#### Create an instance of the MultilayerGraph class


    mg = mx.MultilayerGraph(list_of_layers=[g1,g2],
                            inter_adjacency_matrix=adj_block)

#### Inter-layer weights (intra-layer weight equals one)


    step = 0.01
    total_steps = 10000
    eigval_all = np.zeros((total_steps,mg.number_of_nodes()))

#### Loop for scanning inter-layer weight with ::step for ::total_steps


    for n in range(total_steps):
        inter_diff = step * n
        diffusion_constants = mg.set_edges_weights(intra_layer_edges_weight=1.0,
                                            inter_layer_edges_weight=inter_diff)
        # Laplacian spectrum        
        eigval_all[n] = mx.laplacian_spectrum(mg,weight="weight")

#### Plot the eigenvalues as a function of inter-layer coupling


    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both',which='major',labelsize=14)
    ax.set_title('Spectrum of ' + mg.name,fontsize=10)
    ax.set_xlabel('$D_x$',fontsize=14)
    ax.set_ylabel('$\lambda_2$,...,$\lambda_{'+\
                    '{}'.format(N*mg.get_number_of_layers())+\
                    '}$',fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(bottom=step,top=total_steps)
    ax.set_xlim(step,step*total_steps)
    ax.plot([step*n for n in range(total_steps)],eigval_all,linewidth=1)
    plt.show()


![png](spectrum_two_layers_files/spectrum_two_layers_17_0.png)



    mg.get_number_of_layers()




    2




    
