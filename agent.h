#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <torch/torch.h>
#include <vector>

#include "model.hpp"
#include "env.h"
using namespace std;
using namespace torch;
class Agent
{
    public:
        Agent();
        void hard_copy();//to hard copy the params of local and target networks
        void soft_copy();//to soft copy the params of local and target networks
        float get_reward(float &reward, float (&reward_type)[5], vector<float> &new_state, vector<float> &old_state);//calculate the reward
        void step(Env &env, float &done, float &reward, float (&reward_type)[5], float &total_reward);
        void learn(double &total_actor_loss, double &total_critic_loss);//sample a batch and learn from it.

        Actor a_local;
        Actor a_target;
        torch::optim::Adam actor_optimizer;

        Critic c_local;
        Critic c_target;
        torch::optim::Adam critic_optimizer;

        torch::Device device;

        ReplayBuffer Buffer;
        OUNoise noise;
        int id;


};