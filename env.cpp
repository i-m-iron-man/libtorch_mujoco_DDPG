#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include "mujoco.h"
#include "glfw3.h"
#include <vector>
#include "env.h"

using namespace std;
Env* Env_helper_pointer;

bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

//callback functions for events, working on making them member functions somehow
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(Env_helper_pointer->m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &(Env_helper_pointer->scn), &(Env_helper_pointer->cam));
}
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}

void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(Env_helper_pointer->m, action, dx/height, dy/height, &(Env_helper_pointer->scn), &(Env_helper_pointer->cam));
}

Env::Env(const char** argv)
{
    Env_helper_pointer = this;
    mj_activate("mjkey.txt");
    char error[1000] = "Could not load binary model";

    if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
        Env::m = mj_loadModel(argv[1], 0);
    else
        Env::m = mj_loadXML(argv[1], 0, error, 1000);
    if( !Env::m )
        mju_error_s("Load model error: %s", error);
    Env::d = mj_makeData(Env::m);

    if( !glfwInit() )
        mju_error("Could not initialize GLFW");
    // create window, make OpenGL context current, request v-sync
    Env::window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(Env::window);
    glfwSwapInterval(1);
    // initialize visualization data structures
    mjv_defaultCamera(&(Env::cam));
    mjv_defaultOption(&(Env::opt));
    mjv_defaultScene(&(Env::scn));
    mjr_defaultContext(&(Env::con));
    // create scene and context
    mjv_makeScene(Env::m, &(Env::scn), 2000);
    mjr_makeContext(Env::m, &(Env::con), mjFONTSCALE_150);
    glfwSetScrollCallback(Env::window, scroll);

}

void Env::reset()
{
    mj_resetData(Env::m,Env::d);
        //get random positions and velocities
    for(short i=0; i<Env::m->nq; i++)
        d->qpos[i] = 3.142*(((double) rand() / (RAND_MAX))-0.5);
    for(short i=0; i<Env::m->nv; i++)
        d->qvel[i] = 5*(((double) rand() / (RAND_MAX))-0.5);
    for(short i=0; i<Env::m->nu; i++)
        d->ctrl[i] = 0.0;
}

void Env::get_state(vector<float> &sensordata)
{
    //change this block according to what is your env state
    mj_step1(Env::m,Env::d);
    sensordata.push_back(sin(Env::d->sensordata[0]));
    sensordata.push_back(cos(Env::d->sensordata[0]));
    sensordata.push_back(Env::d->sensordata[1]);
    sensordata.push_back(Env::d->ctrl[0]);
    sensordata.push_back(Env::d->time-Env::epoch_start_time);
    //std::this_thread::sleep_for(std::chrono::microseconds(1));

}
void Env::take_action(vector<double> &action)
{
    d->ctrl = &action[0];
    mj_step2(Env::m,Env::d);
}

void Env::render(/*mjtNum &render_time*/)
{
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(Env::window, &viewport.width, &viewport.height);

    // update scene and render
    mjv_updateScene(Env::m, Env::d, &(Env::opt), NULL, &(Env::cam), mjCAT_ALL, &(Env::scn));
    mjr_render(viewport, &(Env::scn), &(Env::con));

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(Env::window);

                            // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
    //render_time = Env::d->time;

}

void Env::set_epoch_time()
{
    Env::epoch_start_time = Env::d->time;
}
void Env::set_render_time()
{
    Env::render_time = Env::d->time;
}


Env::~Env()
{
    mj_deleteData(Env::d);
    mj_deleteModel(Env::m);
    mj_deactivate();
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif
        cout<<"destructed"<<endl;
}