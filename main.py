from ep_utils.setups import wf_setup
from env.context import Context
from tf_agents.environments import utils

if __name__ == '__main__':
    test_wfs, test_times, test_scores, test_size = wf_setup(['Montage_25'])
    ttree, tdata, trun_times = test_wfs[0]
    environment = Context(5, [4, 8, 8, 16], trun_times, ttree, tdata)
    utils.validate_py_environment(environment, episodes=5)
