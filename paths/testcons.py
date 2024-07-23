

######################################################
#  SConscript file for paths library
######################################################
import os
import fnmatch
from SCons.Script import Import, Return, Dir

Import('env', 'compiler', 'common_build_dir', 'paths_build_dir')
#Import("*")
######################################################
#  Retrieve list of files that need to be compiled
######################################################
srcs = []
for file in os.listdir(Dir('.').srcnode().abspath):
   if fnmatch.fnmatch(file, "*.f90"):
      srcs.append(file)

# For Nag handling
srcs.remove("PATHS_fileIOM_nw.f90")	  
if compiler=='nag':
	srcs.remove('PATHS_fileIOM.f90')
	srcs.append('PATHS_fileIOM_nw.f90')

############################################
#  construction environment is imported    #
############################################

env=env.Clone()
env['F90PATH'] = ['#' + common_build_dir]
env['FORTRANMODDIR'] = "#" + paths_build_dir

######################################################
#  Build the library
######################################################

paths_lib = env.StaticLibrary(target = "paths", source = srcs )

Return("paths_lib")
