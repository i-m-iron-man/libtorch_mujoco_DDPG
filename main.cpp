#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <torch/torch.h>
#include <boost/circular_buffer.hpp>
#include "mujoco.h"
#include "glfw3.h"
#include <vector>

#include "agent.h"
#include "env.h"

using namespace std;

void train(Env &env, Agent &agent, short total_epochs=1500, float max_time = 10.0/*sec*/)
{
    for(short epoch=0; epoch<total_epochs; epoch++)
    {
        agent.noise.reset();
        env.reset();
        float total_reward = 0.0;
        float reward = 0.0;
        float done = 0.0;
        float reward_type[5] = {0.0,0.0,0.0,0.0,0.0};//type of total rewards: [task torque time reached_top too_speedy]
        double total_actor_loss = 0.0;
        double total_critic_loss = 0.0;

        env.set_epoch_time();//env.epoch_start_time is set to zero
        env.set_render_time();//env.render_time is set to zero

        while(env.d->time - env.epoch_start_time < max_time)
        {
            agent.step(env, done, reward, reward_type, total_reward);
            agent.learn(total_actor_loss, total_critic_loss);
            
            if(env.d->time-env.render_time>0.01)
            {
                env.render();
                env.set_render_time();
            }
            if(done>0.5 || glfwWindowShouldClose(env.window))
                break;
        };
        cout<<"end of epoch: "<<epoch<<" total reward:"<<total_reward<<"time: "<<env.d->time - env.epoch_start_time<<endl;
        cout<<"reward_type: distance: "<<reward_type[0]<<" torque: "<<reward_type[1]<<" time: "<<reward_type[2]<<" done: "<<reward_type[3]<<" too_speedy: "<<reward_type[4]<<endl;
        cout<<"actor_loss(-critic) : "<<total_actor_loss<<endl;
        cout<<"critic_loss : "<<total_critic_loss<<endl<<endl;
        if(glfwWindowShouldClose(env.window))
            break;
    }
}

int main(int argc, const char** argv)
{
    Env env(argv);
    Agent agent;
    train(env,agent);
    

    return 0;
}