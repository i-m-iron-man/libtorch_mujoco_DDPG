#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <torch/torch.h>
#include <boost/circular_buffer.hpp>
#include <vector>

#include "agent.h"

using namespace std;
using namespace torch;
using Experience = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

/*learning hyper-params*/
const int64_t actor_state = 5;//input size of actor
const int64_t actor_layer_1 = 256;
const int64_t actor_layer_2 = 128;
const int64_t actor_layer_3 = 64;
const int64_t actor_layer_4 = 32;
const int64_t actor_out = 1;//output size of actor

// input to critic = actor_state , actor_out as critic is for action-value
const int64_t critic_layer_1 = 200;
const int64_t critic_layer_2 = 100;
const int64_t critic_layer_3 = 25;
const int64_t critic_out = 1;// output_size of critic
// note: architecture and activation f's of networks can be changed in "model.hpp"

const float actor_lr = 3e-4;      //learning rate for actor = 0.0001
const float critic_lr = 1e-3;     //learning rate of critic= 0.001
const float tau = 0.01;            //rate at which target nets are updated from local nets
const int batch_size = 256;        //IT IIZ WHAT IT IIZ!
float action_scale = 1.0;     // we are going to scale the output of action by this quantity
const float gamma = 0.99;         //forget/discout rate



Agent::Agent():
a_local(Actor(actor_state, actor_layer_1, actor_layer_2, actor_layer_3, actor_layer_4, actor_out)),
a_target(Actor(actor_state, actor_layer_1, actor_layer_2, actor_layer_3, actor_layer_4, actor_out)),
c_local(Critic(actor_state, actor_out, critic_layer_1, critic_layer_2, critic_layer_3, critic_out)),
c_target(Critic(actor_state, actor_out, critic_layer_1, critic_layer_2, critic_layer_3, critic_out)),
actor_optimizer(a_local->parameters(), actor_lr),
critic_optimizer(c_local->parameters(), critic_lr),
Buffer(ReplayBuffer()),
noise(OUNoise(actor_out)),
device(torch::kCPU)
{
    Agent::id = id;
    if (torch::cuda::is_available()) 
    {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }
    a_local->to(device);
    a_target->to(device);
    c_local->to(device);
    c_target->to(device);

    Agent::hard_copy();
}


float Agent::get_reward(float &reward, float (&reward_type)[5], vector<float> &new_state, vector<float> &old_state)
{
    //target is to reach the top cos(pi) = -1
    float new_distance = abs((-1) - new_state[1]);
    float old_distance = abs((-1) - old_state[1]);
    reward = 10*((0.99*old_distance)-new_distance);
    reward_type[0] += reward; // task reward

    reward -= abs(new_state[3]);
    reward_type[1] -= abs(new_state[3]);// negative reward for more torque

    reward -= 0.1*new_state[4];
    reward_type[2] -= 0.1*new_state[4];// negative reward for more time taken

    if(new_distance<0.01 && abs(new_state[2])<0.3)// if we are near the top and our speed is less
    {
        reward += 1000;
        reward_type[3] += 1000;// task successful
        return 1.0;
    }
    else if(abs(new_state[1])>10)//else if our speed is too much
    {
        reward -=1000;
        reward_type[4] -= 1000;// safety breached shut down
        return 1.0;
    }
    else
    {
        return 0.0;
    }

}

void Agent::step(Env &env, float &done, float &reward, float (&reward_type)[5], float &total_reward)
{
    a_local->eval();
    NoGradGuard gaurd;
    
    vector<float> sensor_values_vector;
    env.get_state(sensor_values_vector);
    Tensor sensor_values_tensor = torch::tensor(sensor_values_vector,device=Agent::device);
    
    auto action_tensor_cpu = a_local->forward(sensor_values_tensor , action_scale).to(torch::kCPU);//global action scale
    vector<double> action_vector(action_tensor_cpu.data<float>(), action_tensor_cpu.data<float>() + action_tensor_cpu.numel());
    action_vector = Agent::noise.sample(action_vector);   //adding noise to the action vector

    env.take_action(action_vector);
    auto action_tensor = torch::tensor(action_vector, device = Agent::device);    //getting action back on gpu for saving

    vector<float> sensor_values_vector_new;
    env.get_state(sensor_values_vector_new);
    Tensor sensor_values_tensor_new = torch::tensor(sensor_values_vector_new,device=Agent::device);

    done = Agent::get_reward(reward, reward_type, sensor_values_vector_new, sensor_values_vector);
    total_reward += reward;

    Tensor reward_tensor = torch::tensor(reward, device = Agent::device);
    reward_tensor = torch::unsqueeze(reward_tensor, 0);
    
    Tensor done_tensor = torch::tensor(done, device = Agent::device);
    done_tensor = torch::unsqueeze(done_tensor, 0);

    Agent::Buffer.addExperienceState(sensor_values_tensor, action_tensor, reward_tensor, sensor_values_tensor_new, done_tensor);
}


void Agent::learn(double &total_actor_loss, double &total_critic_loss)
{
    if(Agent::Buffer.getLength() > size_t(batch_size))//batchsixe global
    {
        a_local->train();
        auto& [state, action, reward_batch, next_state, donezo] = Buffer.sample(batch_size);//global batch_size
        auto actions_next = a_target->forward(next_state,action_scale);//global actions acale 
        auto Q_targets_next = c_target->forward(next_state, actions_next);
        auto Q_targets = reward_batch + (gamma * Q_targets_next * (1 - donezo));//global gamma
        auto Q_expected = c_local->forward(state, action); 
    
        torch::Tensor critic_loss = torch::mse_loss(Q_expected, Q_targets.detach());
        critic_optimizer.zero_grad();
        critic_loss.backward();
        critic_optimizer.step();

        auto actions_pred = a_local->forward(state,action_scale);////global actions acale
    
        auto actor_loss = -(c_local->forward(state, actions_pred).mean());
        actor_optimizer.zero_grad();
        actor_loss.backward();
        actor_optimizer.step();

        Agent::soft_copy();

        total_actor_loss += actor_loss.item<double>();
        total_critic_loss += critic_loss.item<double>();
    }
}

void Agent::hard_copy()
{
    torch::NoGradGuard no_grad;
    for(size_t i=0; i < a_target->parameters().size(); i++)
        a_target->parameters()[i].copy_(a_local->parameters()[i]);
    for(size_t i=0; i < c_target->parameters().size(); i++)
        c_target->parameters()[i].copy_(c_local->parameters()[i]);
}

void Agent::soft_copy()
{
    torch::NoGradGuard no_grad;//       disables calulation of gradients
    for(size_t i=0; i < a_target->parameters().size(); i++)
        a_target->parameters()[i].copy_(tau * a_local->parameters()[i] + (1.0 - tau) * a_target->parameters()[i]);//global tau
    for(size_t i=0; i < c_target->parameters().size(); i++)
        c_target->parameters()[i].copy_(tau * c_local->parameters()[i] + (1.0 - tau) * c_target->parameters()[i]);//global tau
}

