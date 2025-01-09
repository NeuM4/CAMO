import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from Acq.Discrete.CFKG import discrete_fidelity_knowledgement_gradient as DMF_KG
from Acq.Discrete.MF_EI import expected_improvement as DMF_EI
from Acq.Discrete.MF_UCB import upper_confidence_bound as DMF_UCB