# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:18:15 2019

@author: mje

Module with functions used to compute or recompute Morris sensitivity.

"""

import numpy
import matplotlib.pyplot as plt

from SALib.analyze import morris
from SALib.sample.morris import sample
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, sample_histograms
import pickle
import datetime
#from GUI import flux_project

def find_directory_name(file_path):
    """Function finding the directory of a file from its path
    
    This function takes the full path string of a file as input parameters, and 
    returns the path of its directory.
    
    Parameters
    ----------
    file_path: str
        Path of the file
    
    Returns
    -------
    str
        Path of the corresponding directory    
    """
    #creation of a list of the strings separated by / in the file's full path   
    file_path_split=file_path.split('/')    
    #initialization of the variable containing the path of the project directory     
    file_directory="/" 
    #the last cell of the list file_path_split, which corresponds to the name of
    #the file and its extension, is not kept
    directory_split=file_path_split[0:len(file_path_split)-1]
    #one string containing the path of the file directory is recreated by 
    #inserting '/' between each cell of directory_split
    file_directory=file_directory.join(directory_split) 
    #one / is added at the end of the project directory path
    file_directory="%s/" %file_directory 

    return file_directory

def same_values_in_arrays(array1,array2):
    """Function checking if two arrays contain the same values with no regard on their order
    
    Returns a boolean indicating if the two arrays given as input parameters 
    contain the same values, without accounting for their orders in their array. 
    
    Parameters
    ----------
    array1: numpy.ndarray
        First array   
    array2: numpy.ndarray
        Second array
        
    Returns
    -------
    bool
        True if the two arrays contain the same values, False otherwise
    
    """
    
    #To contain the same values, the two arrays must have the same length and 
    #their intersection must also have the same length as them
    if array1.shape[0]==array2.shape[0]:
        #calculation of the intersection
        intersection=numpy.intersect1d(array1,array2)
        if intersection.shape[0]==array1.shape[0]:
            return True
        else:
            #In this case the two arrays have the same length but their intersection
            #is smaller so they do not contain the same values
            return False
    else:
        #In this case the two arrays have different lengths so they do not contain the
        #same values
        #NB: The values of the arrays are considered unique
        return False

def detect_same_rows(matrix):
    """Function to detect the same rows in a 2-dimensional matrix
    
    This function analyses the rows of a 2D matrix and returns the indices of
    the rows appearing several times
    
    Parameters
    ----------
    matrix: numpy.ndarray
        Matrix to analyze
    
    Returns
    -------
    
    dict
        dictionary containing the indices of the rows appearing several times
        in the matrix. The keys are the indices of the first appearance of the 
        rows appearing several times, and the corresponding values are lists 
        containing the indices of their other appearances.
    
    """
    
    dict_similar_indices={}
    #Each row of the matrix is considered
    for row_index in range(matrix.shape[0]):
        #If the row is already indentified in the dict as similar to another one
        #it is not considered. Otherwise a search for similar rows is done
        already_identified=0
        for similar_indices in dict_similar_indices.values():
            if row_index in similar_indices:
                already_identified=1
                break
        if already_identified==0:
            list_similar_rows_indices=[]
            for row_index_check in range(row_index+1,matrix.shape[0]):
                if numpy.all(numpy.equal(matrix[row_index],matrix[row_index_check])):
                    list_similar_rows_indices.append(row_index_check)
            
            #If some rows are the same as the one tested, their index are stored
            #in the dict
            if list_similar_rows_indices!=[]:
                dict_similar_indices["%s"%row_index]=list_similar_rows_indices
    
    return dict_similar_indices
            
class dummy_opt_parameters:
    def __init__(self):
        self.a=1
    def define_points_to_modify(self):
        
        return []

def Morris_sensitivity(sens_params_names, bnds, sampling_method,
                       functions_list,
                       function_related_params_dicts_list=None,
                       groups_params=None,
                       results_storage_path=None,
                       plot_results=False,
                       function_display_captions=None):
    """
    Function to evaluate the global sensitivity of a design to several parameters.
    
    This function uses the Morris method to evaluate the sensitivity of a function
    to several parameters in their variation range.
    Some sensitivity plots are generated, if requested.
    
    Parameters
    ----------
    sens_params_names: list of strings 
        List containing the names of the parameters.
        
    bnds: tuple 
        tuple of tuples containing the (min,max) bounds for each parameter
        
    sampling_method: dict
        Dictionnary containing all the necessary data to run a classical Morris sampling or
        an improved sampling maximizing the distance between trajectories (Campolongo 2007),
        either with brut force trajectories selection or with faster iterative trajectories
        selection (Ruano2012).
        It should have follwing keys :
            "method": str
                "classical" for a random Morris sampling
                "Compolongo" or "Ruano" to perform Compolongo or Ruano improvements
            "ntraj_final": int
                number of trajectories to use for the sensitivity analysis
            "ntraj_to_gen": int
                For Ruano or Compolongo methods, number of trajectories to generate
                from which a number of optimal trajectories equal to "ntraj_final" 
                are finally taken
    
    functions_list: list of functions
        List of the Python functions of interest defined in the same file. They must
        necessarily have two arguments, variables_values and index. The other arguments
        can be filled in using the function_related_params_dicts_list
    
    function_related_params_dicts_list: list of dicts
        List of dictionaries containing all the information necessary only for the evaluation
        of the functions of interest (the **kwargs). If this sensitivity method is used for other 
        applications than electric motors sensitivity, this dictionnary is not 
        required anymore, while the other parameters are necessary.
        
    groups_params: list of strings
        List containing the name of the group each parameter belongs to. This variable
        is defined in **kwargs. If not defined, each parameter is considered individually
        (one factor at a time - OAT). If defined, each parameter must belong to one
        group and one groupe only.
        
    results_storage_path: string
        Path where the Morris sample and results files must be saved

    plot_results: True/False, False by default
        To plot results when calling function or not
        
    function_display_captions: list of strings
        List of functions captions to display on the plots
    
    Returns
    -------
    Reurns tuple  (Morris_sample,Morris_sensitivity_results)
    
    Morris_sample: numpy.ndarray
        Array of all the normalized samples used for the Morris sensitivity
        analysis, of size (n_parameters, n_trajectories)
        
    Morris_sensitivity_results: list of dictionaries 
        list containing, for each function of interest, a dictionary containing
        the Morris statistical data related to the output function
    
    """   
    
    #### Check of inputs
    
    #There must be one set of boundary values per parameter
    if len(sens_params_names)!=len(bnds):
      raise TypeError('The number of parameter names and parameter boundary values mismatch')
  
    #If groups are defined, there must be one membership group for each parameter
    if groups_params!=None:
        if len(groups_params)!=len(bnds):
            raise TypeError("The number of parameter names and parameter group memberships mismatch")

    #If no additional parameters, to be passed as **kwargs arguments of the 
    #functions contained in functions_list, are given, the dedicated list 
    #function_related_params_dicts_list is filled with empty dicts for all
    #functions
    if function_related_params_dicts_list == None:
        function_related_params_dicts_list = [{} for index_func in range(len(functions_list))]
    
    #else if a list of additional parameters is given, a dictionary must be specified for each
    #function
    else:            
        if len(function_related_params_dicts_list) != len(functions_list):
            raise TypeError("a **kwargs dictionary must be defined for each function of interest")
        
    #if no function captions to display on plots are given, generic captions are built
    if function_display_captions == None:
        function_display_captions=["Objective #{}".format(index_function+1) for index_function in range(len(functions_list))]

    #If some captions are given, they must be given in a list of the same size
    #as the number of functions (one caption per function)
    else:
        if len(function_display_captions)!=len(functions_list):
            raise TypeError("Mismatch between specified captions and functions")

    #### Calculation of the normalization coefficients

    #list storing, for each parameter, the size of the allowed range between the bounds
    size_ranges = [0 for i in range(len(sens_params_names))] 
    #List of the normalization multiplicative coefficients
    multiplicative_coeffs_normalization = [0 for i in range(len(sens_params_names))] 
    #List of normalization offsets
    additive_coeffs_normalization = [0 for i in range(len(sens_params_names))]
    
    #Calculation of the size of the range between the bounds for each parameter   
    for index_param in range(0,len(sens_params_names)):
        size_ranges[index_param] = bnds[index_param][1] - bnds[index_param][0]

    #Calculation of the values of the normalization coefficients
    for index_param in range(0,len(sens_params_names)):
        #The offset is the lower bound
        additive_coeffs_normalization[index_param] = bnds[index_param][0]
        #The multiplicative coefficient is the size of the allowed range
        multiplicative_coeffs_normalization[index_param] = size_ranges[index_param]
        
    
    #Definition of the boundary values used for the sensitivity analysis. The
    #parameters values being normalized, the bounds are [0.0,1.0] for all 
    #parameters
    bounds_normalized = [[0.0,1.0] for i in range(len(sens_params_names))]
    
    #Definition of the "problem" dictionary as specified in the SALib library
    problem={
            'num_vars': len(sens_params_names),
            'names': [sens_params_names[i] for i in range(len(sens_params_names))],
            'groups': groups_params,
            'bounds': bounds_normalized    
            }    
    
    #If 'classical' Morris sampling method is used, the number of trajectories
    #to generate, i.e. the N argument of the "sample" function, is the number of 
    #trajectories required
    if sampling_method["method"] == "classical":
        ntraj_to_generate = sampling_method["ntraj_final"]
        ntraj_to_select = None #No selection is done, all trajectories are selected

    #If an improved sampling maximizing the distance between trajectories is selected,
    # the final trajectories are selected from a larger amount of generated trajectories        
    elif sampling_method["method"] in ["Compolongo","Ruano"]:
        ntraj_to_generate = sampling_method["ntraj_to_gen"]
        ntraj_to_select = sampling_method["ntraj_final"]
    
    #If the Ruano method is used, the corresponding option must be activated in
    #the function "sample"
    if sampling_method["method"] == "Ruano":
        local_opt = True
    else:
        local_opt = False
    
    Morris_sample = sample(problem, N=ntraj_to_generate, num_levels=sampling_method["num_levels"],
                      optimal_trajectories=ntraj_to_select,local_optimization=local_opt)  
    
    ################## Denormalization of the samples ##################
    
    Morris_sample_denormalized=numpy.copy(Morris_sample)    
    
    for index_sample in range(Morris_sample.shape[0]):
        for index_parameter in range(Morris_sample.shape[1]):
            Morris_sample_denormalized[index_sample][index_parameter]=(
                    Morris_sample[index_sample][index_parameter]*
                    multiplicative_coeffs_normalization[index_parameter]+
                    additive_coeffs_normalization[index_parameter])
            
    ##### Determination of the samples appearing several times ##########
    same_samples=detect_same_rows(Morris_sample)
    
    #Initialization of the array containing the function values for each sample contained in Morris_sample
    
    function_values_list=[numpy.zeros(Morris_sample.shape[0]) for index_func in range(len(functions_list))]

    ######### Determination of the function values for each sample ########    
    

    #for each row of the Morris_sample array, the function of interest must be evaluated
    for index_sample in range(Morris_sample.shape[0]):

        ### check if the same sample was already evaluated in another trajectory ####
        sample_already_calculated=0 #variable set to 1 if the sample appears in the values of same_samples
        for (first_sample_indices_str,same_samples_lists) in same_samples.items():
            if index_sample in same_samples_lists:
                sample_already_calculated=1
                first_sample_index=int(first_sample_indices_str)
                break
 
        for index_function in range(len(function_values_list)):        
            #If this is the first appearance of the sample, it is evaluated normally
            if sample_already_calculated==0:
                function_values_list[index_function][index_sample]=functions_list[index_function](
                        Morris_sample_denormalized[index_sample],
                        index=index_sample,**function_related_params_dicts_list[index_function])
            #if the function of interest was already evaluated for the same sample
            #in another trajectory, its values is copied
            else:
                function_values_list[index_function][index_sample]=function_values_list[index_function][first_sample_index]

    date_str=str(datetime.datetime.now())
    date_str=date_str.replace(' ','_')
    date_str=date_str.replace('-','_')
    date_str=date_str.replace(':','_')
    date_str=date_str.replace('.','_')

    if results_storage_path!=None:
        with open("%s_Morris_sample%s.pickle"%(results_storage_path,date_str),"wb") as f:
            pickle.dump(Morris_sample,f)
            f.close()
        with open("%s_functions_values%s.pickle"%(results_storage_path,date_str),"wb") as f:
            pickle.dump(function_values_list,f)
            f.close()        

    Morris_sensitivity_results=[] #List containing the output from morris_analyze for each function

#    for index_function in range(len(function_values_list)):       
#        Morris_sensitivity_results.append(morris.analyze(problem, Morris_sample, function_values_list[index_function],
#                                                         conf_level=0.95,print_to_console=True,
#                                                         num_levels=sampling_method["num_levels"],
#                                                         num_resamples=100))
#        
#    
#        # Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
#        # e.g. dict['mu_star'] contains the mu* value for each parameter, in the
#        # same order as the parameter file
#    
#        fig, (ax1, ax2) = plt.subplots(1, 2)
#        horizontal_bar_plot(ax1, Morris_sensitivity_results[index_function], {}, 
#                            sortby='mu_star', unit=r"%s"%function_display_captions[index_function])
#        covariance_plot(ax2, Morris_sensitivity_results[index_function], {},
#                        unit=r"%s"%function_display_captions[index_function])
#
#        fig2 = plt.figure()
#        sample_histograms(fig2, Morris_sample,problem, {'color': 'y'})
#        plt.show()

    for index_function in range(len(function_values_list)):       
        Morris_sensitivity_results.append(morris.analyze(problem, Morris_sample, function_values_list[index_function],
                                                         conf_level=0.95,print_to_console=True,
                                                         num_levels=sampling_method["num_levels"],
                                                         num_resamples=100))
          
        # Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
        # e.g. dict['mu_star'] contains the mu* value for each parameter, in the
        # same order as the parameter file
    
    if plot_results:
   
        for index_function in range(len(function_values_list)):       
    
            fig, (ax1, ax2) = plt.subplots(1, 2)
            horizontal_bar_plot(ax1, Morris_sensitivity_results[index_function], {}, 
                                sortby='mu_star', unit=r"%s"%function_display_captions[index_function])
            covariance_plot(ax2, Morris_sensitivity_results[index_function], {},
                            unit=r"%s"%function_display_captions[index_function])

            #for now this kind of plot is only working when the number of samples
            #is under 10. Therefore, this plot is not printed if the number of samples
            #is larger than 9
            if len(sens_params_names)<9:
                fig2 = plt.figure()
                sample_histograms(fig2, Morris_sample,problem, {'color': 'y'})
                plt.show()

    
    if results_storage_path!=None:
        with open("%s_results%s.pickle"%(results_storage_path,date_str),"wb") as f:
            pickle.dump(Morris_sensitivity_results,f)
            f.close()

    return (Morris_sample,Morris_sensitivity_results)

def recompute_Morris_sensitivity(sens_params_names,
                                 n_levels,
                                 samples_array,
                                 function_values_list,
                                 groups_params=None,
                                 function_display_captions=None,
                                 plot_results=False):
    """Function to reperform a Morris sensitivity analysis when the samples were evaluated previously
    
    This function takes as input a normalized Morris sample and the corresponding 
    values of the functions of interest, and performs the sensitivity analysis 
    using these data. 
    
    Parameters
    ----------
    sens_params_names: list of strings 
        List containing the names of the sensitivity parameters.
    n_levels: int
        Number of grid levels of the imported results
    samples_array: numpy.ndarray
        Array containing all the samples previously calculated for a Morris
        sensitivity analysis
    function_values_list: list of ndarrays
        List containing arrays in which the values of each function of interest
        is contained for each sample contained in samples
    groups_params: list of strings
        List containing the name of the group each parameter belongs to. This variable
        is defined in **kwargs. If not defined, each parameter is considered individually
        (one factor at a time - OAT). If defined, each parameter must belong to one
        group and one groupe only.
    function_display_captions: list of strings
        List of functions captions to display on the plots
    plot_results: bool, False by default
        Variable to indicate if "old" plots should be plotted or not when function
        is called.
    
    Returns
    -------
    numpy.ndarray
        Array of all the normalized samples used for the Morris sensitivity
        analysis
    list of dictionaries 
        list containing, for each function of interest, a dictionary containing
        the Morris statistical data related to the output function    
    """   
    
    #If groups are defined, there must be one membership group for each parameter
    if groups_params!=None:
        if len(groups_params)!=len(sens_params_names):
            print("Error: the number of parameter names and parameter group memberships are not matching")
    
    #if no function captions to display on plots are given, generic captions are built
    if function_display_captions==None:
        function_display_captions=["function%s"%(index_function+1) for index_function in range(len(function_values_list))]

    #If some captions are given, they must be given in a list of the same size
    #as the number of functions (one caption per function)
    else:
        if len(function_display_captions)!=len(function_values_list):
            print("Error: a caption must be given for each function")
            return
    
    #Calculation of the number of trajectories
    n_traj=samples_array.shape[0]/(len(sens_params_names)+1)
    #The number of trajectories must be an integer, otherwise samples_array and 
    #sens_params_names are not matching
    if int(n_traj)!=n_traj:
        print("Error: the number of samples and the number of sensitivity parameters are not matching")
        return
    
    #Check that samples contained in samples_array contain a value for each
    #parameter contained in sens_params_names
    if samples_array.shape[1]!=len(sens_params_names):
        print("Error: the number of values contained in the samples is not matching",
              " the number of parameters names")
    # check that function_values_list contains arrays of the same size, and
    #matching the size of samples_array
    for index_function in range(len(function_values_list)):
        if function_values_list[index_function].shape[0]!=samples_array.shape[0]:
            print("Error: the results must match the samples")
            return

    bounds_normalized=[[0.0,1.0] for i in range(len(sens_params_names))]
    
    #Definition of the "problem" dictionary as specified in the SALib library
    problem={
            'num_vars': len(sens_params_names),
            'names': [sens_params_names[i] for i in range(len(sens_params_names))],
            'groups': groups_params,
            'bounds': bounds_normalized    
            }  

    Morris_sensitivity_results=[] #List containing the output from morris_analyze for each function

    for index_function in range(len(function_values_list)):

        Morris_sensitivity_results.append(morris.analyze(problem, samples_array, function_values_list[index_function],
                                                         conf_level=0.95,print_to_console=False,
                                                         num_levels=n_levels,
                                                         num_resamples=100))

        # Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
        # e.g. dict['mu_star'] contains the mu* value for each parameter, in the
        # same order as the parameter file
        #NOTES PARPINOU:
            #1# Avec version salib 1.4.5, si print_to_console=True, ça bug...
            
            #2#☻  If modified within the 'morris.analyze', there also can be
            # a new key ["elementary_effects"], with all
            # computed elementary effects, array of size (n_params, n_trajectories)
                #but we need to have admin rights to do so
        
            #3#♥ Another way of doing this, by copying functions from morris.py
        # if "elementary_effects" not in Morris_sensitivity_results[index_function].keys():
        #         delta = morris._compute_delta(n_levels)
        #         num_vars = problem['num_vars']
        #         groups = morris._check_groups(problem)
        #         if not groups:
        #             number_of_groups = num_vars
        #         else:
        #             groups, unique_group_names = morris.compute_groups_matrix(groups)
        #             number_of_groups = len(set(unique_group_names))
                
        #         X = samples_array
        #         Y = function_values_list[index_function]
        #         num_trajectories = int(Y.size / (number_of_groups + 1))
        #         trajectory_size = int(Y.size / num_trajectories)
            
        #         elementary_effects = morris._compute_elementary_effects(X, Y, trajectory_size, delta)
            
        #         Morris_sensitivity_results[index_function]["elementary_effects"] = elementary_effects
        if "elementary_effects" not in Morris_sensitivity_results[index_function].keys():
            delta = morris._compute_delta(n_levels)
            num_vars = problem['num_vars']
        
            # Supposons qu'il n'y a pas de groupes
            number_of_groups = num_vars
        
            X = samples_array
            Y = function_values_list[index_function]
            num_trajectories = int(Y.size / (number_of_groups + 1))
            trajectory_size = int(Y.size / num_trajectories)
        
            elementary_effects = morris._compute_elementary_effects(X, Y, trajectory_size, delta)
        
            Morris_sensitivity_results[index_function]["elementary_effects"] = elementary_effects


    if plot_results:
        #    for index_function in range(len(function_values_list)-1,0,-1): 
        for index_function in range(len(function_values_list)):       
    
            fig, (ax1, ax2) = plt.subplots(1, 2)
            horizontal_bar_plot(ax1, Morris_sensitivity_results[index_function], {}, 
                                sortby='mu_star', unit=r"%s"%function_display_captions[index_function])
            covariance_plot(ax2, Morris_sensitivity_results[index_function], {},
                            unit=r"%s"%function_display_captions[index_function])
    
            #for now this kind of plot is only working when the number of samples
            #is under 10. Therefore, this plot is not printed if the number of samples
            #is larger than 9
            if len(sens_params_names)<9:
                fig2 = plt.figure()
                sample_histograms(fig2, samples_array,problem, {'color': 'y'})
                plt.show()

    return Morris_sensitivity_results

def Morris_sensitivity_gen_sample_only(sens_params_names, bnds, sampling_method,groups_params=None):

    """
    Function to generate Morris trajectories for given parameters variation ranges
    
    Parameters
    ----------
    sens_params_names: list of strings 
        List containing the names of the sensitivity parameters.
    
    bnds: tuple 
        tuple of tuples containing the (min,max) bounds for each sensitivity
        parameter
    
    sampling_method: dict
        Dictionnary containing all the necessary data to run a classical Morris sampling or
        an improved sampling maximizing the distance between trajectories (Campolongo 2007),
        either with brut force trajectories selection or with faster iterative trajectories
        selection (Ruano2012).
        It should have follwing keys :
            "method": str
                "classical" for a random Morris sampling
                "Compolongo" or "Ruano" to perform Compolongo or Ruano improvements
            "ntraj_final": int
                number of trajectories to use for the sensitivity analysis
            "ntraj_to_gen": int
                For Ruano or Compolongo methods, number of trajectories to generate
                from which a number of optimal trajectories equal to "ntraj_final" 
                are finally taken
    
    groups_params: list of strings
        List containing the name of the group each parameter belongs to. This variable
        is defined in **kwargs. If not defined, each parameter is considered individually
        (one factor at a time - OAT). If defined, each parameter must belong to one
        group and one groupe only.
    
    Returns
    -------
    Morris_sample: numpy.ndarray
        Array of all the normalized samples used for the Morris sensitivity
        analysis
    
    Morris_sample_denormalized: numpy.ndarray
        Array of all the denormalized samples used for the Morris sensitivity
        analysis   
    """

    #There must be one set of boundary values per parameter
    if len(sens_params_names)!=len(bnds):
      print('Error: the number of parameter names and parameter boundary values are not matching')
      return
  
    #If groups are defined, there must be one membership group for each parameter
    if groups_params!=None:
        if len(groups_params)!=len(bnds):
            print("Error: the number of parameter names and parameter group memberships are not matching")


    ##### Calculation of the normalization coefficients #####

    #list storing, for each parameter, the size of the allowed range between the bounds
    size_ranges=[0 for i in range(len(sens_params_names))] 
    #List of the normalization multiplicative coefficients
    multiplicative_coeffs_normalization=[0 for i in range(len(sens_params_names))] 
    #List of normalization offsets
    additive_coeffs_normalization=[0 for i in range(len(sens_params_names))]
    
    #Calculation of the size of the range between the bounds for each parameter   
    for index_param in range(0,len(sens_params_names)):
        size_ranges[index_param]=bnds[index_param][1]-bnds[index_param][0]

    #Calculation of the values of the normalization coefficients
    for index_param in range(0,len(sens_params_names)):
        #The offset is the lower bound
        additive_coeffs_normalization[index_param]=bnds[index_param][0]
        #The multiplicative coefficient is the size of the allowed range
        multiplicative_coeffs_normalization[index_param]=size_ranges[index_param]
        
    
    #Definition of the boundary values used for the sensitivity analysis. The
    #parameters values being normalized, the bounds are [0.0,1.0] for all 
    #parameters
    bounds_normalized=[[0.0,1.0] for i in range(len(sens_params_names))]
    
    #Definition of the "problem" dictionary as specified in the SALib library
    problem={
            'num_vars':len(sens_params_names),
            'names': [sens_params_names[i] for i in range(len(sens_params_names))],
            'groups':groups_params,
            'bounds': bounds_normalized
            }    
    
    #If there classical Morris sampling method is used, the number of trajectories
    #to generate, i.e. the N argument of the "sample" function, is the number of 
    #trajectories required
    if sampling_method["method"]=="classical":
        ntraj_to_generate=sampling_method["ntraj_final"]
        ntraj_to_select=None #No selection is done, all trajectories are selected

    #If an improved sampling maximizing the distance between trajectories is selected,
    # the final trajectories are selected from a larger amount of generated trajectories        
    elif sampling_method["method"] in ["Compolongo","Ruano"]:
        ntraj_to_generate=sampling_method["ntraj_to_gen"]
        ntraj_to_select=sampling_method["ntraj_final"]
    
    #If the Ruano method is used, the corresponding option must be activated in
    #the function "sample"
    if sampling_method["method"]=="Ruano":
        local_opt=True
    else:
        local_opt=False
    
    Morris_sample=sample(problem, N=ntraj_to_generate, num_levels=sampling_method["num_levels"],
                      optimal_trajectories=ntraj_to_select,local_optimization=local_opt)

    #for now this kind of plot is only working when the number of samples
    #is under 10. Therefore, this plot is not printed if the number of samples
    #is larger than 110
    if len(sens_params_names)<10:
        fig2 = plt.figure()
        sample_histograms(fig2, Morris_sample,problem, {'color': 'y'})
        plt.show()
    ################## Denormalization of the samples ##################
    
    Morris_sample_denormalized=numpy.copy(Morris_sample)    
    
    for index_sample in range(Morris_sample.shape[0]):
        for index_parameter in range(Morris_sample.shape[1]):
            Morris_sample_denormalized[index_sample][index_parameter]=(
                    Morris_sample[index_sample][index_parameter]*
                    multiplicative_coeffs_normalization[index_parameter]+
                    additive_coeffs_normalization[index_parameter])

    return Morris_sample,Morris_sample_denormalized

def classify_results_sensitivity(results_sensitivity,
                                 list_of_categories=None,
                                 criteria_names = None,
                                 ratio = 5/2):
    """
    Fonction pour classifier l'effets des paramètres sur les les objectifs évalués,
    après traitement par les fonctions propres la méthode Morris.

    Parameters
    ----------
    results_sensitivity : list of dict
        Liste de sortie typiquement obtenue en sortie de la fonction "recompute_Morris_sensitivity()"
        C'est une liste, de dictionnaires, 1 dictionnaire / Critère évalué
        avec notamment les clefs suivantes attendues :
            "elemantary_effects": array de taille (n_params, n_traj)
            "mu_star": moyenne absolue array de taille (n_params)
            "names": noms des paramètres, liste de taille (n_params)
    categories : list of dict, None par défaut
        Liste de dictionnaires, 1 dictionnaire / critère, pour éventuellement
        renseigner les catégories à considérer pour le classement des effets
        moyens absolus des paramètres.
        Exemple:
            categories = [{"low:" [0, 1],
                           "medium": [1, 10],
                           "strong": [10, 100],
                           "vstrong": [100, 1000]}]
            Si non renseigné, le code considère par défaut 3 catégories,
            "low", "medium", "strong", d'après la valeur maxi des effets moyens absolus.
            
            /!\ Il est important que les bornes renseignées ENGLOBENT toutes les
            valeurs prises par les effets élémentaires...
            
            NB, quand l'intervalle [a, b] est spécifié pour les bornes,
            l'algo cherche valeurs incluses dans [a, b[.

                criteria_names : List of str
        Optionnel, liste des labels de critères éavlués.
    ratio: float
        Limite au delà de laquelle l'effet du paramètre est jugé "++""--" ou "+-"
        C'est le ratio du nombre d'effets élémentaires positifs vs négatifs.
        5/2 par défaut ; si 5 effets positifs contre 2 négatifs, alors '++', sinon '+-'

    Returns
    -------
    df: DataFrame pandas
        tableau de classification des paramètres vs objectifs

    """
    import pandas as pd
    import numpy as np
    import copy
    
    if list_of_categories is None:
        list_of_categories_was_None = True
        list_of_categories = [] # init
    else:
        list_of_categories_was_None = False

        
    dicos = []
    for index, d in enumerate(results_sensitivity): # pour chaque critère évalué
        
        # Catégories utilisées
        if list_of_categories_was_None:
            max_abs_effect = np.max(np.abs(d["mu_star"]))
            list_of_categories.append({"low": [0, max_abs_effect/3],
                                       "medium": [max_abs_effect/3, 2*max_abs_effect/3],
                                       "strong": [2*max_abs_effect/3, max_abs_effect]})
        
        categories = list_of_categories[index]
        
        # On ca faire le tri, en mettant infos dans un dictionnaire
        dico = {}
        for param_index, param in enumerate(d["names"]): # Pour chaque paramètre
            dico[param] = {}
            for c, categorie in enumerate(categories.keys()):
                if d["mu_star"][param_index] >= categories[categorie][0]:
                    if d["mu_star"][param_index] < categories[categorie][1]:
                        dico[param]["effect"] = categorie
                        break
                    elif c == len(categories.keys()):
                        raise TypeError("L'intervalle renseigné pour la classification n'englobe pas toutes les valeurs d'effets élémentaires calculés.",
                                        "Il faut augmenter la valeur maxi...")
                    # else:
                    #     raise TypeError(param, categorie, d["mu_star"][param_index], categories[categorie][0], categories[categorie][1])
            # Reste à juger du sens de l'effet, plutôt "++" ou "--" ?
            # En comptant les effets élémentaires positifs et négatifs...
            n_pos = 0
            n_neg = 0
            for k, e in enumerate(d["elementary_effects"][param_index]):
                # Pour chaque effet élémentaire calculé du paramètre
                if e >= 0:
                    n_pos += 1
                else:
                    n_neg += 1
                # Si il y a plus de 4/5 d'effet dans un sens, c'est "++"" ou "--"
                # Sinon c'est "+-"
                if n_neg == 0:
                    dico[param]["direction"] = "++"
                elif n_pos/n_neg > ratio:
                    dico[param]["direction"] = "++"
                elif n_pos == 0:
                    dico[param]["direction"] = "--"
                elif n_neg/n_pos > ratio:
                    dico[param]["direction"] = "--"
                else:
                    dico[param]["direction"] = "+-"
            # Fin du tri sur 1 paramètre
        dicos.append(copy.deepcopy(dico))
        # Fin du tri sur tous les paramètres pour 1 critère
    # Fin du tri sur tous les critères
    
    if criteria_names is None:
        criteria_names = ["Objective {}".format(k) for k in range(len(results_sensitivity))]
    
    categories_names_by_objective = [list(categories.keys()) for categories in list_of_categories]
    all_categories_names = []
    for a in categories_names_by_objective:
        all_categories_names += a
    
    # On va construire 1 DataFrame / Objectif, avant de merger à la fin
    # ça permet de mettre différentes catégories par objectif
    dfs = []
    muxs = []
    datas = []
    
    for k, dico in enumerate(dicos): # Pour chaque critère
        # On prépare un multiindex
        muxs.append(pd.MultiIndex.from_product([[criteria_names[k]], categories_names_by_objective[k]]))
        datas.append([]) #init 1 liste / critère
        for param_index, param in enumerate(dicos[0].keys()): # pour chaque paramètre
            # On connait la catégorie du paramètre: dicos[param][effect]
            # On connait la direction de l'efftee: dicos[param][direction]
            # reste à remplir une liste avec autant de catégories que définies
            # On initialise des str vides, pour chaque catégorie
            datas[-1].append(['' for cat in list_of_categories[k].keys()])
            # On renseigne la direction de l'effet du paramètre, dans la bonne liste
            datas[-1][param_index][categories_names_by_objective[k].index(dico[param]["effect"])] = copy.deepcopy(dico[param]["direction"])
        # Fin de la boucle sur les paramètres
        # On créé un DataFrame
        dfs.append(pd.DataFrame(datas[-1], columns=muxs[-1], index=list(dicos[0].keys())))
    # Fin de la boucle sur les critères
    
    # On merge le dataframe pour le retrun
    df = dfs[0]
    for k in range(len(dfs)):
        if k > 0:
            df = df.join(dfs[k])
    
    return df