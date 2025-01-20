import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from Acquistion_function.Continue.CF_KG import continuous_fidelity_knowledgement_gradient as CMF_KG
from Acquistion_function.Continue.CF_UCB import upper_confidence_bound_continuous as CMF_UCB