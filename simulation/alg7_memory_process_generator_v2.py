# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 20:10:31 2021

@author: Mike
"""

def Process_with_memory(D = ["a","b","c","d","e"], 
                                    mode = ["min_entropy","max_entropy","med_entropy"][2], 
                                    num_traces=2, 
                                    sample_len=50,
                                    K=2):
    import numpy as np
    import pandas as pd
    
    ##### Part 1: Generate the transition probabilities
    
    # event-log container
    Theta = []
    
    
    # Including absorption state
    D_abs = D.copy()
    D_abs.append("!")
    
    
    # Generate the model
    from simulation.alg2_initial_probabilities import GenerateInitialProb
    from simulation.homc_helpers import create_homc

    
    #generate initial probabilities
    probabilities = GenerateInitialProb(D_abs, p0_type=mode)    
    P0 = {}
    
    for i in range(0,len(D_abs)):
        P0.update({D_abs[i]:probabilities[i]})
    

    #create the markov chain
    HOMC = create_homc(D_abs, P0, h=K)

    
    Phi = HOMC
    
    ##### Part 2: Draw from the distributions
    for trace in list(range(0,num_traces)):
                
        #Trace placeholder
        sigma = []
        
        #Continue drawing until there is an absorption event when length = x
        while "!" not in set(sigma):
            
            #Sample trace from model
            sigma = HOMC.sample(sample_len)
            
            #Remove all extra occurrences of absorption state
            sigma = sigma[:sigma.index('!')+1]
            
        #recode the name of the termination event
        sigma = [w.replace('!', 'END') for w in sigma]
    
        #Update the event-log
        Theta.append(sigma)
        
    return Theta, HOMC




# Theta, HOMC = Process_with_memory_pomegranate(D = ["a","b","c","d","e"], 
#                                     mode = ["min_entropy","max_entropy","med_entropy"][2], 
#                                     num_traces=2, 
#                                     sample_len=50,
#                                     K=2)





# def Process_with_memory(D = ["a","b","c","d","e"], 
#                         mode = ["min_entropy","max_entropy","med_entropy"][2], 
#                         num_traces=2, 
#                         K=2,
#                         settings={"med_ent_e_steps":5,
#                                           "med_ent_n_transitions":5,
#                                           "med_ent_max_trials":100}):
#     import numpy as np
#     import pandas as pd
    
#     from simulation.alg2_initial_probabilities import GenerateInitialProb
#     from simulation.alg3_transition_matrix_min_entropy import Generate_transition_matrix_min_ent
#     from simulation.alg4_transition_matrix_max_entropy import Generate_transition_matrix_max_ent
#     from simulation.alg5_transition_matrix_med_entropy import Generate_transition_matrix_med_ent
    
#     def TransMatWrapper(D,mode):
#         if mode =="min_entropy":
#             P = Generate_transition_matrix_min_ent(D, P0)
            
#         if mode =="max_entropy":
#             P = Generate_transition_matrix_max_ent(D)
            
#         if mode =="med_entropy":
#             P = Generate_transition_matrix_med_ent(D,
#                                                #e_steps=settings["med_ent_e_steps"],
#                                                n_tranitions=settings["med_ent_n_transitions"],
#                                                limit_trials=settings["med_ent_max_trials"])
#         return P
    
    
#     ##### Part 1: Generate the transition probabilities
    
#     #event-log container
#     Theta = []
    
#     repetitions = num_traces #10
        
    
#     #Including absorption state
#     D_abs = D.copy()
#     D_abs.append("END")
    
#     #Amount of memory K
#     #K=10
    
#     #Transition matrices
#     Phi = []
    
#     #Absorption matrix
#     A = np.zeros((len(D)+1,len(D)+1))
#     A[:,len(D)] = 1
#     A = pd.DataFrame(A,columns=D_abs)
    
#     #generate initial probabilities
#     P0 = GenerateInitialProb(D, p0_type=mode)
    
#     #Append to Phi
#     Phi.append(P0)
    
#     #min entropy exception
#     if mode =="min_entropy":
#         P = TransMatWrapper(D,mode)
    
#     for i in range(0,K):
        
#         if i < K-1:
       
#             #Generate a transition matrix
#             P_i = TransMatWrapper(D,mode)
            
#             if mode =="min_entropy":
#                 #Append to Phi
#                 Phi.append(P)
                
#             if mode !="min_entropy":
#                 #Append to Phi
#                 Phi.append(P_i)
#         else:
#             #Append to Phi
#             Phi.append(A)
        
    
#     ##### Part 2: Draw from the distributions
        
#     for trace in list(range(0,repetitions)):
        
#         #Timestep counter
#         t = 0
        
#         #Trace placeholder
#         sigma = []
    
        
#         #sample from initial distribution
#         e_t = np.random.choice(D, #len(D), #
#                                    size=1, replace=False, p=P0)[0]
        
#         sigma.append(e_t)
        
#         #If current event is not absorption
#         while e_t != "END":
#             t = t+1
            
#             #sample from distribution P_k
#             P = Phi[t]
            
#             #Add index so that states can be looked up
#             P.index = D_abs
            
#             #get conditional probability (e_t'th row of P)
#             p_t = P.loc[P.index==e_t]
            
#             e_t = np.random.choice(D_abs, size=1, replace=False, p=p_t.values[0])[0]
            
#             sigma.append(e_t)
    
#         #print(trace,": ",sigma)
        
#         Theta.append(sigma)

#     return Theta, Phi


# Theta, Phi = Process_with_memory(D = ["a","b","c","d","e"], 
#                         mode = ["min_entropy","max_entropy","med_entropy"][2], 
#                         num_traces=2, 
#                         K=10)