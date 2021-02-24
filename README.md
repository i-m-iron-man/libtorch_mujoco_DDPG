## Inverted Pendulum task
The task is to balance the pendulum upright using continuous control when the maximum torque available is not enough to overcome gravity. In this implementation a DDPG algorithm is used to learn the task. Mujoco libarary(http://www.mujoco.org/) is used to model the environment and pytorch c++ frontend(https://pytorch.org/cppdocs/frontend.html) provides the support for reinforcement learning framework.

## platform

  * OS: windows10(64 bit)
  * CUDA: 10.1
  * cuDNN: 7.6.3
  * Compiler: MSVC 19.28.29336.0
  * C++17
  * Cmake: 3.19.2

## Dependencies
  * Mujoco: [mujoco200 win64](https://www.roboti.us/download/mujoco200_win64.zip)
  * pytorch C++ frontend: [1.7.1 release](https://pytorch.org/get-started/locally/)
  * boost: [1.75.0](https://www.boost.org/users/history/version_1_75_0.html)

## Instructions
  * In Cmake.txt replace `C:/Users/kings/Desktop/source/mujoco200_win64/mujoco200_win64` at line 13 with the path to your mujoco installation
  * In Cmake.txt replace `C:/Users/kings/Desktop/source/libtorch-mujoco-git` at line 14 with the path to where you download these source files
  * open a command-prompt and `cd` to the location of these files
  * `mkdir build && cd build`
  * In order to build use command `cmake -DCMAKE_PREFIX_PATH=<absolute path to libtorch>\libtorch ..`
  * In order to make use command `cmake --build . --config Release`
  * In order to run first `cd Release` then `libtorch-mujoco-git.exe ..\..\model\simple_pendulum.xml`

## Inspiration
https://github.com/EmmiOcean/DDPG_LibTorch
