# MontoThermoPOC: 
This is a thermal simulator by MC. 




# Quick Start
First we create a Python virtual environment (on Mac/Linux):
    python3 -m venv ~/.MontoThermoPOC
Activate the venv:

For Zombie library we need to acticate this env. 
ource ~/.MontoThermoPOC312/bin/activate
    

# Dependencies

## GCode generator
For the slicer i used prusaslicer. To install that: 
    brew install --cask prusaslicer

Check the installation:
    /Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer --version

Use this library with this example:
    /Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer --export-gcode --output tests/20mmbox.gcode tests/20mmbox.stl

Note that the Slic3r library needs many dependencies. According to claude: 
"Slic3r-master is written in Perl, and it needs a module called local::lib that isn't installed on your machine".
So I decided to use PrusaSlicer.

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

after the pr is approved and merged. Delete your branch on your local as well as on remote:
    git branch -d feature/test_feature_branch
    git push origin --delete feature/test_feature_branch


# Testing
Every class and fuctions should have the corresponding unit test. Please do note create a PR wihtout adding the related 
unit test. 

## How to run the test
Example:
    pytest tests/test_slicer.py -v 

# Next step
Each time step we need to update the geometry
GCode reader: Start a simple example 
We need a function to get the GCode and time step and return the envlope up to that time step
And then we run MC for this output



