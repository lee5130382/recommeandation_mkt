# +
# import glob
# from os.path import dirname, basename, isfile, join
# from imp import reload
# def reload_modules(path_func, var_dict):
#     """
#     Reload all module in <path_func> dirs
#     Args :
#         path_func (string) : function locate dir (ex: main/func/) 
#         var_dict (dictionary) : globals() or locals() in jupyter notebook (dictionary variable mapping)
    
#     Note : 
#         All packages in this folder will be reloaded. 
#         Can not direct to specific module.
    
#     """
#     # -- File py  get in paths
#     modules = glob.glob(join(dirname( path_func ), "*.py"))
    
#     # -- Name get by cut the <.py> string (by -3) 
#     __all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
    
#     # -- Modules reload (by globals())
#     for module in __all__ : 
#         reload(var_dict[ module ])
        
#     print("Reload Packges in {0} done.".format(path_func))
