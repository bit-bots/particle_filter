cmake_minimum_required(VERSION 2.8.3)
project(particle_filter)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosconsole
  std_msgs
  geometry_msgs
  visualization_msgs
  eigen_conversions
)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenMP)
if(OPENMP_FOUND)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
if(MSVC90 OR MSVC10)
if(MSVC90)
set(OPENMP_DLL VCOMP90)
elseif(MSVC10)
set(OPENMP_DLL VCOMP100)
endif(MSVC90)
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DELAYLOAD:${OPENMP_DLL}D.dll")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DELAYLOAD:${OPENMP_DLL}.dll")
endif(MSVC90 OR MSVC10)

else(OPENMP_FOUND)
message (STATUS "OpenMP not found")
endif()

if (MSVC)
Set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /O2 ${SSE_FLAGS}")
else (MSVC)
set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS} ${SSE_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g ${SSE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE " ${CMAKE_CXX_FLAGS} -O3 ${SSE_FLAGS}")
endif (MSVC)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS
        roscpp
    DEPENDS EIGEN3
)

include_directories(
        include 
        ${EIGEN3_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)


# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


set(SOURCES
    src/CRandomNumberGenerator.cpp
    src/gaussian.cpp
    src/gaussian_mixture_model.cpp
    src/gmm_classifier.cpp
    src/gmm_regressor.cpp
    src/k_means.cpp
    src/matrix_io.cpp
    src/pca.cpp
)

set(CODE_LIBRARIES )

add_library(${PROJECT_NAME} ${SOURCES})

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
