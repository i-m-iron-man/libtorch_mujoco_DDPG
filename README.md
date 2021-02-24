The task is to balance the pendulum upright using continuous control. In this implementation a DDPG algorithm is used to learn the task. Mujoco libarary(http://www.mujoco.org/) is used to model the environment and pytorch c++ frontend(https://pytorch.org/cppdocs/frontend.html) provides the support for reinforcement learning framework.

platform:
OS: windows10(64 bit)
CUDA: 10.1
Compiler: 
C++:C++17

Dependencies:
Mujoco: mujoco200 win64(https://www.roboti.us/download/mujoco200_win64.zip)
pytorch C++ frontend: 1.7.1(https://pytorch.org/get-started/locally/)
boost:1.75.0
