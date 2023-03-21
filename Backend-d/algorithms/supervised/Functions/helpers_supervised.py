def import_functions_from_folder(folder_name_list):
    import importlib,sys,os,inspect, collections
    print('foldernamelist:', folder_name_list)
    info = collections.namedtuple('info', 'description value')
    result = collections.namedtuple('result', 'score properties')
    #general+supervised
    sys.path.extend([r"Backend",r"Backend/algorithms",r"Backend/algorithms/supervised", r"Backend/algorithms/supervised/Functions", r"Backend/algorithms/supervised/Functions/Accountability",r"Backend/algorithms/supervised/Functions/Fairness",r"Backend/algorithms/supervised/Functions/Explainability",r"Backend/algorithms/supervised/Functions/Robustness"])
    #unsupervised
    sys.path.extend([r"Backend/algorithms/unsupervised", r"Backend/algorithms/unsupervised/Functions", r"Backend/algorithms/unsupervised/Functions/Accountability",r"Backend/algorithms/unsupervised/Functions/Fairness",r"Backend/algorithms/unsupervised/Functions/Explainability",r"Backend/algorithms/unsupervised/Functions/Robustness"])
    pillars_function_list=[info, result]
    functions  ={}
    for folder_name in folder_name_list:
        pillar_function_file_name=folder_name+'_supervised.py' 
        folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), folder_name))
        folder_name = folder_path.split('/')[-1]
        for file_name in os.listdir(folder_path):
           
            if file_name.endswith('.py') and file_name!=pillar_function_file_name and file_name!="__init__.py":
                module_name = file_name[:-3]
                sys.path.append(folder_path)
                module = importlib.import_module(f'{module_name}')
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and name.startswith('get_'):
                        function_name = name[4:].lower().replace('_', '')
                        print("FUNCTION NAME: ",function_name)
                        functions[function_name] = obj
        print('functions:', functions.length)
        pillars_function_list.append(functions)
        print("result:", pillars_function_list)
    return pillars_function_list

