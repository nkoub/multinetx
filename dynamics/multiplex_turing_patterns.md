
#### Import libraries


    import numpy as np
    import matplotlib.pylab as plt
    
    import multinetx as mx
    
    from scipy.integrate import ode

#### The Mimura-Murray ecological model


    def mimura_murray_duplex(t, y0, G):
        f = y0.copy()    
        u = y0[:G.N] # activator
        v = y0[G.N:] # inhibitor    
        sum_lapl_u = G.diff[0] * u * G.lapl[0:G.N,0:G.N]   
        sum_lapl_v = G.diff[1] * v * G.lapl[G.N:2*G.N,G.N:2*G.N]    
        # activator
        f[:G.N] = ( (G.a + G.b * u - u * u) / G.c - v) * u + sum_lapl_u
        # inhibitor   
        f[G.N:] = (u - 1.0 - G.d * v) * v + sum_lapl_v
        return f 

#### Define the integrate method


    def integrate(G, dt=0.5, tmax=200, method='dopri5', store_solution=True):
        '''This function integrates the MM model'''
        ## Setup the integrator 
        t0 = 0.0
        sol = [G.species]
        solver = ode(G.rhs).set_f_params(G)
        int_pars = dict(atol=1E-8, rtol=1E-8, first_step=1E-2,
                        max_step=2E1, nsteps=2E4)
        solver.set_integrator(method,**int_pars)
        solver.set_initial_value(G.species,t0)
        ## Integrate the system    
        while solver.successful() and solver.t < tmax:
            solver.integrate(solver.t+dt)
            sol.append(solver.y)
        return np.array(sol)

#### Create the activator-inhibitor multiplex


    G = mx.MultilayerGraph()


    N = 350
    G.add_layer(mx.barabasi_albert_graph(n=N, m=5,   seed=812))   # activators
    G.add_layer(mx.barabasi_albert_graph(n=N, m=200, seed=812))   # inhibitors

#### Laplacian matrices of the multiplex


    G.lapl = (-1.0) * mx.laplacian_matrix(G,weight=None)

#### Right-hand-side of the Mimura-Murray model


    G.rhs = mimura_murray_duplex

#### Define the parameters of the model (They correspond to the uniform steady state)


    G.a = 35.0
    G.b = 16.0
    G.c = 9.0
    G.d = 0.4


    G.N = G.get_number_of_nodes_in_layer() 
    G.diff = [0.12, 0.12]  

#### Initial conditions and perturbation


    activators = np.empty(G.N,dtype=np.double)
    inhibitors = np.empty(G.N,dtype=np.double)
    activators[:] = 5.0
    inhibitors[:] = 10.0
    activators[10] += 10E-5
    G.species = np.concatenate((activators,inhibitors),axis=0)

#### Integrate the system


    sol = integrate(G,dt=0.5,tmax=200,method='dopri5')

#### Sort the solution according to decreasing degree of the activator layer


    deg_act = G.get_layer(0).degree().values()
    sdeg_act = np.argsort(deg_act)[::-1]
    sdeg_sol = np.append(sdeg_act,G.N+sdeg_act)
    ssol = sol[:,sdeg_sol]


    def plot_sol(ax, insol, NN, t=0):
        ax.plot(insol[t],':',color='green',lw=1.1,alpha=1)
        sc = ax.scatter(np.arange(NN),insol[t],
                        c=insol[t],s=25,marker='o',lw=1.2,
                        vmin=min(insol[t]),vmax=max(insol[t]),
                        cmap=plt.cm.YlOrBr)
        ax.set_xlim(-5,NN)


    %matplotlib inline

#### Development of Turing pattern (activator layer is shown)


    sol_act = ssol[:,:G.N]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.set_ylabel(r'$u_i$')
    ax1.set_title(r'$t=0$')
    ax1.set_ylim(0,10)
    ax1.set_xticks([])
    plot_sol(ax1,sol_act,G.N,t=0)
    
    ax2 = fig.add_subplot(222)
    ax2.set_ylabel(r'$v_i$')
    ax2.set_title(r'$t=18$')
    ax2.set_ylim(0,10)
    ax2.set_xticks([])
    plot_sol(ax2,sol_act,G.N,t=18)
    
    ax3 = fig.add_subplot(223)
    ax3.set_xlabel(r'$i$')
    ax3.set_ylabel(r'$v_i$')
    ax3.set_title(r'$t=21$')
    ax3.set_ylim(0,10)
    plot_sol(ax3,sol_act,G.N,t=21)
    
    ax4 = fig.add_subplot(224)
    ax4.set_xlabel(r'$i$')
    ax4.set_ylabel(r'$v_i$')
    ax4.set_title(r'$t=400$')
    ax4.set_ylim(0,10)
    plot_sol(ax4,sol_act,G.N,t=400)
    
    plt.show()


![png](multiplex_turing_patterns_files/multiplex_turing_patterns_25_0.png)


#### Turing pattern shown in activator and inhibitor layer


    sol_act = ssol[:,:G.N]
    sol_inh = ssol[:,G.N:]
    tsnap = 400
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel(r'$u_i$')
    ax1.set_ylim(0,10)
    plot_sol(ax1,sol_act,G.N,t=tsnap)
    
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel(r'$i$')
    ax2.set_ylabel(r'$v_i$')
    ax2.set_ylim(7.5,11.1)
    plot_sol(ax2,sol_inh,G.N,t=tsnap)
    
    plt.show()


![png](multiplex_turing_patterns_files/multiplex_turing_patterns_27_0.png)



    
