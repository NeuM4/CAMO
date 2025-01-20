import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from Acquistion_function.Discrete.DF_KG import discrete_fidelity_knowledgement_gradient as DMF_KG
from Acquistion_function.Discrete.DF_EI import expected_improvement as DMF_EI
from Acquistion_function.Discrete.DF_UCB import upper_confidence_bound as DMF_UCB