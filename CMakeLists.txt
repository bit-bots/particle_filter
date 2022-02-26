cmake_minimum_required(VERSION 3.5)
project(particle_filter)

# Add support for C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(eigen_conversions REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(visualization_msgs REQUIRED)

set(INCLUDE_DIRS include ${std_msgs_INCLUDE_DIRS} ${geometry_msgs_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS} ${eigen_conversions_INCLUDE_DIRS}
  ${ament_cmake_INCLUDE_DIRS} ${visualization_msgs_INCLUDE_DIRS})
include_directories(${INCLUDE_DIRS})

set(LIBRARY_DIRS ${std_msgs_LIBRARY_DIRS} ${geometry_msgs_LIBRARY_DIRS}
  ${rclcpp_LIBRARY_DIRS} ${eigen_conversions_LIBRARY_DIRS}
  ${ament_cmake_LIBRARY_DIRS} ${visualization_msgs_LIBRARY_DIRS})

link_directories(${LIBRARY_DIRS})

set(LIBS ${std_msgs_LIBRARIES} ${geometry_msgs_LIBRARIES} ${rclcpp_LIBRARIES}
  ${eigen_conversions_LIBRARIES} ${ament_cmake_LIBRARIES}
  ${visualization_msgs_LIBRARIES})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
add_compile_options(-Wall -Werror -Wno-unused)

find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  if(MSVC90 OR MSVC10)
    if(MSVC90)
      set(OPENMP_DLL VCOMP90)
    elseif(MSVC10)
      set(OPENMP_DLL VCOMP100)
    endif(MSVC90)
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG
      "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DELAYLOAD:${OPENMP_DLL}D.dll")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE
      "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DELAYLOAD:${OPENMP_DLL}.dll")
  endif(MSVC90 OR MSVC10)

else(OPENMP_FOUND)
  message(STATUS "OpenMP not found")
  add_compile_options(-Wno-unknown-pragmas)
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /O2 ${SSE_FLAGS}")
else(MSVC)
  set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS} ${SSE_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g ${SSE_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE " ${CMAKE_CXX_FLAGS} -O3 ${SSE_FLAGS}")
endif(MSVC)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(SOURCES src/CRandomNumberGenerator.cpp src/gaussian.cpp
  src/gaussian_mixture_model.cpp src/gmm_classifier.cpp src/gmm_regressor.cpp
  src/k_means.cpp src/matrix_io.cpp src/pca.cpp)

set(CODE_LIBRARIES)

add_library(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${LIBS})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
  LIBRARY
  DESTINATION lib)

ament_export_dependencies(std_msgs)
ament_export_dependencies(geometry_msgs)
ament_export_dependencies(rclcpp)
ament_export_dependencies(eigen_conversions)
ament_export_dependencies(ament_cmake)
ament_export_dependencies(visualization_msgs)
ament_export_include_directories(${INCLUDE_DIRS})
ament_export_libraries(${PROJECT_NAME} ${LIBS})

install(DIRECTORY config  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})

ament_package()
