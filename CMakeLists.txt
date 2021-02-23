cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

get_filename_component(APP_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${APP_NAME})


find_package(Torch REQUIRED)
find_package(Boost 1.75.0 REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(MJC_path "C:/Users/kings/Desktop/source/mujoco200_win64/mujoco200_win64")
set(source_path "C:/Users/kings/Desktop/source/libtorch-mujoco-git")

include_directories(${MJC_path}/include ${source_path})
file(GLOB LIB_MUJOCO "${MJC_path}/bin/mujoco200.lib")
file(GLOB LIB_GL "${MJC_path}/bin/glfw3.lib")
file(GLOB LIB_GL_static "${MJC_path}/bin/glfw3static")




add_executable(${PROJECT_NAME} main.cpp)
target_sources(${PROJECT_NAME}
        PRIVATE
        agent.h
        agent.cpp
        model.hpp
        env.h
        env.cpp
        )

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" ${LIB_MUJOCO} ${LIB_GL} ${LIB_GL_static})
#target_link_libraries(${PROJECT_NAME} ${LIB_MUJOCO})
#target_link_libraries(${PROJECT_NAME} ${LIB_GL})
#target_link_libraries(${PROJECT_NAME} ${LIB_GL_static})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)



if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  file(GLOB MJC_DLLS "${MJC_path}/bin/*.dll")
  file(GLOB MJKEY "${MJC_path}/bin/mjkey.txt")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)

  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${MJC_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)

  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${MJKEY}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)