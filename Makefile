CXX = g++
CXXFLAGS = -std=c++17 -Wall -g -O0

CUDA_PATH = /usr/local/cuda
TENSORRT_PATH = /usr

INCLUDE_DIRS = -I$(CUDA_PATH)/include \
               -I$(TENSORRT_PATH)/include \
               -I$(TENSORRT_PATH)/include/x86_64-linux-gnu \
               -I. -Iai -Igame -Imcts -Itesting

LIB_DIRS = -L$(CUDA_PATH)/lib64 \
           -L$(TENSORRT_PATH)/lib/x86_64-linux-gnu

LIBS = -lcudart -lnvinfer -lnvinfer_plugin

LDFLAGS = $(LIB_DIRS) $(LIBS)

SRCS = $(wildcard *.cpp) \
       $(wildcard ai/*.cpp) \
       $(wildcard game/*.cpp) \
       $(wildcard mcts/*.cpp)

OBJS = $(patsubst %.cpp,build/%.o,$(SRCS))

TARGET = myprogram

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

build/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf build $(TARGET)
