# Project Files
- ```./my_utils/formula_utils.py``` contains custom utility functions to parse CNF data in DIMACs text files and manipulate Boolean formulas
- ```./my_utils/graph_utils.py``` contains custom graph utility functions to convert Boolean formulas into the graph representations VCG, LCG, VCGm, and LCGm, as well as plotting functions
- ```./models.py``` contains the custom GCN message-passing implementation for heterogeneous graphs for each representation in VCG, LCG, VCGm, and LCGm
- ```i_train.py``` is a scipt that takes in command line arguments to train the model, type in ```python i_train.py -h``` at the terminal to see usage
- ```i_test.py``` is a script that takes in command line arguments to test the trained models, type in ```python i_test.py``` at the terminal to see usage

# Dependencies
The following can be run in a Colab cell to execute the scripts
```
import torch
!pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install git+https://github.com/pyg-team/pytorch_geometric.git
!pip install ipywidgets==7.7.1
```
# Dataset
The dataset zip file must be extracted to a directory ```./dataset``` before the scripts can be run.
