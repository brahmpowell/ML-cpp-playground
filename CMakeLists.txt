cmake_minimum_required(VERSION 3.5)
project(FirstProjWithEigen)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#target_link_libraries(fdf)
#link_directories("/usr/lib/x86_64-linux-gnu/")
#target_link_libraries(X11)

set(SOURCE_FILES main.cpp  #CImg/CImg.h
        MiscOp.cpp  MiscOp.h
        NN1.cpp     NN1.h
        NN2.cpp     NN2.h
        KMeans1.cpp KMeans1.h)
add_executable(FirstProjWithEigen ${SOURCE_FILES})

#
#cmake_minimum_required(VERSION 3.5)
#project(testAllegro)
#
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(SOURCE_FILES main.cpp)
#
#LINK_DIRECTORIES( d:\\Logiciels\\mingw\\lib\\allegro )
#add_executable(testAllegro ${SOURCE_FILES})
#INCLUDE_DIRECTORIES( d:\\Logiciels\\mingw\\include\\allegro5 )
#TARGET_LINK_LIBRARIES(testAllegro allegro_acodec allegro_audio ...)