#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <torch/torch.h>
#include <boost/circular_buffer.hpp>
#include <vector>

using namespace torch;
using Experience = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;   //this is how we are going to save expirience in buffer

/*              Critic definition               */
struct CriticImpl: nn::Module
{
    CriticImpl(int state_size, int action_size, int layer1,int layer2, int layer3, int out):
    CLayer_state(nn::Linear(state_size,state_size)),
    CLayer_action(nn::Linear(action_size,action_size)),
    CLayer1(nn::Linear(state_size+action_size,layer1)),
    //Cbatch_norm1(layer1),
    CLayer2(nn::Linear(layer1,layer2)),
    //Cbatch_norm2(layer2),
    CLayer3(nn::Linear(layer2,layer3)),
    //Cbatch_norm3(layer3),
    CLayer4(nn::Linear(layer3,out))
    {
        register_module("CLayer_state",CLayer_state);
        register_module("CLayer_action",CLayer_action);
        register_module("CLayer1",CLayer1);
        register_module("CLayer2",CLayer2);
        register_module("CLayer3",CLayer3);
        register_module("CLayer4",CLayer4);
        //register_module("Cbatch_norm1",Cbatch_norm1);
        //register_module("Cbatch_norm2",Cbatch_norm2);
        //register_module("Cbatch_norm3",Cbatch_norm3);
    }

    torch::Tensor forward(torch::Tensor state , torch::Tensor action) 
    {
        if (state.dim() == 1)
            state = torch::unsqueeze(state, 0);//Returns a new tensor with a dimension of size one inserted at the specified position.

        if (action.dim() == 1)
            action = torch::unsqueeze(action,0);
        state = torch::tanh(CLayer_state(state));
        action = torch::tanh(CLayer_action(action));
        auto xa = torch::cat({state,action}, /*dim=*/1);
        xa=torch::tanh(CLayer1(xa));
        xa=torch::tanh(CLayer2(xa));
        xa=torch::tanh(CLayer3(xa));
        xa=CLayer4(xa);
        return xa;
    }
    void print_params()
    {
        for (const auto& p : this->parameters()) 
        {
            std::cout << p << std::endl;
        }
    }

    torch::nn::Linear CLayer_state,CLayer_action,CLayer1,CLayer2,CLayer3,CLayer4;
    //torch::nn::BatchNorm1d Cbatch_norm1, Cbatch_norm2, Cbatch_norm3;
};
TORCH_MODULE(Critic);       //refer to https://pytorch.org/tutorials/advanced/cpp_frontend.html

/*              Actor definition            */
struct ActorImpl: nn::Module
{
    ActorImpl(int state_size, int layer1,int layer2, int layer3, int layer4, int out):
    //ALayer1(nn::Linear(nn::LinearOptions(state_size, layer1).bias(false))),
    ALayer1(nn::Linear(state_size,layer1)),
    ALayer2(nn::Linear(layer1,layer2)),
    ALayer3(nn::Linear(layer2,layer3)),
    ALayer4(nn::Linear(layer3,layer4)),
    ALayer5(nn::Linear(layer4,out))
    {
        register_module("ALayer1",ALayer1);
        register_module("ALayer2",ALayer2);
        register_module("ALayer3",ALayer3);
        register_module("ALayer4",ALayer4);
        register_module("ALayer5",ALayer5);
    }

    torch::Tensor forward(torch::Tensor x, float action_scale) 
    {
        x=torch::tanh(ALayer1(x));
        x=torch::tanh(ALayer2(x));
        x=torch::tanh(ALayer3(x));
        x=torch::tanh(ALayer4(x));
        x=action_scale*torch::tanh(ALayer5(x));
        return x;
    }
    void print_params()
    {
        for (const auto& p : this->parameters()) 
        {
            std::cout << p << std::endl;
        }
    }
    torch::nn::Linear ALayer1,ALayer2,ALayer3,ALayer4,ALayer5;
};
TORCH_MODULE(Actor);

class ReplayBuffer 
{
        public:
            ReplayBuffer() {}

            void addExperienceState(torch::Tensor state, torch::Tensor action, torch::Tensor reward, torch::Tensor next_state, torch::Tensor done)// add the 5 info to buffer
            {
                addExperienceState(std::make_tuple(state, action, reward, next_state, done));//but first convert them to a tuple
            }

            void addExperienceState(Experience experience) {
             circular_buffer.push_back(experience);//finally add them
            }

         Experience sample(int num_agent)//make a batch of size = num_Agent(rows) x 5(colums)// we need to transpose for batches
         {
             Experience experiences;
             Experience exp = sample();
             experiences = exp;
                
                for (short i = 0; i < num_agent-1; i++) {
                    exp=sample();
                    std::get<0>(experiences) = torch::cat({std::get<0>(experiences),std::get<0>(exp)}, /*dim=*/0);
                    std::get<1>(experiences) = torch::cat({std::get<1>(experiences),std::get<1>(exp)}, /*dim=*/0);
                    std::get<2>(experiences) = torch::cat({std::get<2>(experiences),std::get<2>(exp)}, /*dim=*/0);
                    std::get<3>(experiences) = torch::cat({std::get<3>(experiences),std::get<3>(exp)}, /*dim=*/0);
                    std::get<4>(experiences) = torch::cat({std::get<4>(experiences),std::get<4>(exp)}, /*dim=*/0);

                 }
             return experiences;
          }

          Experience sample() {
                  Experience exp = circular_buffer.at(static_cast<size_t>(rand() % static_cast<int>(circular_buffer.size())));//sample a data point randomly
                  //for(size_t i=0; i<5; i++)
                        std::get<0>(exp) = torch::unsqueeze(std::get<0>(exp), 0);
                        std::get<1>(exp) = torch::unsqueeze(std::get<1>(exp), 0);
                        std::get<2>(exp) = torch::unsqueeze(std::get<2>(exp), 0);
                        std::get<3>(exp) = torch::unsqueeze(std::get<3>(exp), 0);
                        std::get<4>(exp) = torch::unsqueeze(std::get<4>(exp), 0);
                   return exp;
          }

          size_t getLength() {
              return circular_buffer.size();//size of the buffer
          }

          boost::circular_buffer<Experience> circular_buffer{1000000};//max size of buffer = 10000
};      //refer to https://github.com/EmmiOcean/DDPG_LibTorch/blob/master/replayBuffer.h

class OUNoise {
//"""Ornstein-Uhlenbeck process."""
    private:
        size_t size;
        std::vector<double> mu;
        std::vector<double> state;
        double theta=0.15;
        double sigma=0.1;

    public:
        OUNoise (size_t size_in) {
            size = size_in;
            mu = std::vector<double>(size, 0);
            reset();
        }

        void reset() {
            state = mu;
        }

        std::vector<double> sample(std::vector<double> action) {
        //"""Update internal state and return it as a noise sample."""
            for (size_t i = 0; i < state.size(); i++) {
                auto random = ((double) rand() / (RAND_MAX));
                double dx = theta * (mu[i] - state[i]) + sigma * random;
                state[i] = state[i] + dx;
                action[i] += state[i];
            }
            return action;
    }
};