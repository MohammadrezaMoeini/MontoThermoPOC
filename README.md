# MontoThermoPOC: 
This is a thermal simulator by MC. 




# Quick Start
First we create a Python virtual environment (on Mac/Linux):
    python3 -m venv ~/.MontoThermoPOC
Activate the venv:

For Zombie library we need to acticate this env. 
ource ~/.MontoThermoPOC312/bin/activate
    



# Pull latest zombie updates                                                                                                                                              
  git submodule update --remote solver/zombie 

# After cloning your repo on a new machine                                                                                                                                
  git clone --recurse-submodules <your-repo-url> 

Each time step we need to update the geometry
GCode reader: Start a simple example 
We need a function to get the GCode and time step and return the envlope up to that time step
And then we run MC for this output



