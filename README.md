# MontoThermoPOC: 
This is a thermal simulator by MC. 




# Quick Start
First we create a Python virtual environment (on Mac/Linux):
    python3 -m venv ~/.MontoThermoPOC
Activate the venv:

For Zombie library we need to acticate this env. 
ource ~/.MontoThermoPOC312/bin/activate
    

# Dependencies

## Slic3r (GCode generation)
Slic3r is used to generate GCode from STL files. It is not included in this repo.
Download the pre-built binary from: https://slic3r.org/download/
Extract it into a `slicers/` folder at the repo root (already in `.gitignore`).

## Zombies
### Pull latest zombie updates                                                                                                                                              
  git submodule update --remote solver/zombie 

### After cloning your repo on a new machine                                                                                                                                
  git clone --recurse-submodules <your-repo-url> 

# Branch sterategy
Create your feature branch as feature/myfeature
    git checkout -b feature/my-feature

and then push
    git push -u origin feature/my-feature

# Next step
Each time step we need to update the geometry
GCode reader: Start a simple example 
We need a function to get the GCode and time step and return the envlope up to that time step
And then we run MC for this output



