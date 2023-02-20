#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


from dataprep.helperfunctions import *
from dataprep.memory_helperfunctions import prepare_data_f_memory


# In[4]:


from simulation.simulation_pipeline import *
from simulation.simulation_helpers import *


# In[5]:


from experiment.DoE import *


# # Make a design table

# In[6]:


run_settings = {"process_entropy":["min_entropy"], #,"med_entropy","max_entropy"
                "number_of_traces":[100],
                "statespace_size":[5],
                "process_type":["memory"],        #,"memory"        
                "process_memory":[3],
                
                "Fact1":[100],
                "Fact2":[100],
                "Fact3":[100]}


# Generate a full factorial:
df=build_full_fact(run_settings)#[0:2]

# Recode the string factor levels (recoding from natural number to string)
df = fix_label_values(df, run_settings, variables = ["process_entropy",
                                                     "process_type"])

# Important variables
df["RUN"] = df.index + 1
df["Done"] = 0
df["Failure"] = 0

#change types
df.statespace_size = df.statespace_size.astype(int)
df


# In[7]:


# Loop over the experiments


# In[8]:


results = []

for run in df.index:
    #print(run)
    #print(df.loc[run])
    
    """
    Settings from experiments
    """
    curr_settings = df.loc[run]
    
    """
    settings for simulation
    """
    
    SIM_SETTINGS = {"save_eventlog":1, #0 = no, 1 = yes...
                
                "statespace_size":make_D(int(curr_settings["statespace_size"])),

                "number_of_traces":int(curr_settings["number_of_traces"]),  

                "process_entropy":curr_settings["process_entropy"],

                "process_type":curr_settings["process_type"],                

                "process_memory":int(curr_settings["process_memory"]),                
                
                                    #desired max number of steps:
                "process_settings":{"med_ent_e_steps":5,
                                    # desired max number of possible transitions in P. 
                                    # NOTE: This can maximally be the number of states, and should be higher than 2
                                    "med_ent_n_transitions":3,
                                    #max number of trials to find matrix with desired max steps
                                    "med_ent_max_trials":5},

                #lambda parameter of inter-arrival times
                "time_settings":{"inter_arrival_time":1.5, 
                                #lambda parameter of process noise
                                "process_stability_scale":0.1,
                                #probability of getting an agent
                                "resource_availability_p":0.5,                          
                                #waiting time in days, when no agent is available      
                                "resource_availability_n":3,
                                #waiting time in days, when no agent is available
                                "resource_availability_m":0.041, 
                                #variation between activity durations
                                "activity_duration_lambda_range":0.5,

                                #time-unit for a full week: days = 7, hrs = 24*7, etc.
                                "Deterministic_offset_W":make_workweek(["weekdays","all-week"][1]),

                                "Deterministic_offset_u":7},

                "run":0}

    # generate the log
    log = Generate_eventlog(SIM_SETTINGS)
    print(len(log))
    
    
    """
    Prepare data for modelling
    """
    input_data = prepare_data_f_memory(log)
    
    """
    Train a model
    """
    # X: 
    input_data["x_train"]
    input_data["x_test"]
    
    # Y:
    input_data["y_test"]
    input_data["y_test"]
    
    """
    Evaluate the model
    """
    
    
    """
    Store the results
    """
    curr_settings["RES_num_events"] = len(log)
    
    curr_settings = pd.DataFrame(curr_settings.T)
    
    results.append(curr_settings)


# # Inspect example data

# In[9]:


log


# In[10]:


input_data


# In[ ]:





# In[ ]:




