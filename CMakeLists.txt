cmake_minimum_required (VERSION 2.8)
project(lightweight_filtering)
add_definitions(-std=c++11)

##################### Set some variables #####################
SET(CMAKE_BUILD_TYPE "Debug")

##################### Find, include, and compile library #####################
if(DEFINED CATKIN_DEVEL_PREFIX)
	find_package(catkin REQUIRED)
	message(STATUS "Building lightweight_filtering as catkin package")
endif()
find_package(Eigen3 REQUIRED)
find_package(Kindr REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(./include)
include_directories(${Kindr_INCLUDE_DIRS})

if(DEFINED CATKIN_DEVEL_PREFIX)
	catkin_package(
	   INCLUDE_DIRS include)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gtest/" AND (NOT DEFINED CATKIN_DEVEL_PREFIX))
	message(STATUS "Building GTests!")
	option(BUILD_GTEST "build gtest" ON)
	add_subdirectory(gtest gtest)
	enable_testing()
	include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
	add_executable(testState src/testState.cpp)
	target_link_libraries(testState gtest_main gtest pthread )
	add_test(testState testState)
	add_executable(testSigmaPoints src/testSigmaPoints.cpp)
	target_link_libraries(testSigmaPoints gtest_main gtest pthread )
	add_test(testSigmaPoints testSigmaPoints)
	add_executable(testPrediction src/testPrediction.cpp)
	target_link_libraries(testPrediction gtest_main gtest pthread )
	add_test(testPrediction testPrediction)
	add_executable(testUpdate src/testUpdate.cpp)
	target_link_libraries(testUpdate gtest_main gtest pthread )
	add_test(testUpdate testUpdate)
	add_executable(testModelBase src/testModelBase.cpp)
	target_link_libraries(testModelBase gtest_main gtest pthread )
	add_test(testModelBase testModelBase)
	add_executable(testFilterBase src/testFilterBase.cpp)
	target_link_libraries(testFilterBase gtest_main gtest pthread )
	add_test(testFilterBase testFilterBase)
endif()

# Generate FindLWF.cmake file
file(WRITE cmake/FindLWF.cmake
"# This file was automatically generated during the installation of the lightweight_filtering library
# and can be used through cmake to find the corresponding header files. A copy of this
# file was created in ${CMAKE_ROOT}/Modules (depending on the CMAKE_ROOT variable).

set(LWF_INCLUDE_DIRS
${CMAKE_INSTALL_PREFIX}/include/LWF/include
)
set(LWF_FOUND TRUE)
message(\"-- LWF found (include: ${CMAKE_INSTALL_PREFIX}/include/LWF/include)\")
"
)

# Setting for make install
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/include/LWF)")
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX}/include/LWF)
install(FILES cmake/FindLWF.cmake DESTINATION ${CMAKE_ROOT}/Modules)