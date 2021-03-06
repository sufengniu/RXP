#
# Makefile in directory src
#
# resource usage:
#
# To compile a dynamic module
# (1) nvcc cannot accept -fPIC, so compile .cu to .cu.cpp first
#     nvcc -arch=sm_20 -cuda ../src/PFAC_kernel.cu
#
# (2) then use g++ to comple PFAC_notex_shared_reorder.cu.cpp
#     g++ -fPIC -c PFAC_kernel.cu.cpp
#
# (3) finally combine two object files to a .so library
#     g++ -shared -o libpfac.so  $(LIBS) PFAC_kernel.cu.o ...
#
#     $(LIBS) is necessary when compiling PFAC library on 32-bit machine
#

include ../common.mk

INC_DIR = ../include
LIB_DIR = ../lib
OBJ_DIR = ../obj

INCPATH += -I../include/

CU_SRC = PFAC_kernel.cu
CU_SRC += PFAC_reduce_kernel.cu
CU_SRC += PFAC_reduce_inplace_kernel.cu
CU_SRC += PFAC_kernel_spaceDriven.cu

CPP_SRC  = PFAC_reorder_Table.cpp
CPP_SRC += PFAC_CPU.cpp
CPP_SRC += PFAC_CPU_OMP.cpp
CPP_SRC += PFAC.cpp

inc_files = $(INC_DIR)/PFAC_P.h  $(INC_DIR)/PFAC.h

CU_OBJ = $(patsubst %.cu,%.o,$(CU_SRC))

CU_CPP = $(patsubst %.cu,%.cu.cpp,$(CU_SRC))

CPP_OBJ = $(patsubst %.cpp,%.o,$(CPP_SRC))

cppobj_loc = $(patsubst %.o,$(OBJ_DIR)/%.o,$(CPP_OBJ))

cppobj_fpic_loc = $(patsubst %.o,$(OBJ_DIR)/%_fpic.o,$(CPP_OBJ))


cu_cpp_sm20_loc = $(patsubst %.cpp,$(OBJ_DIR)/sm20_%.cpp,$(CU_CPP))
cu_cpp_sm13_loc = $(patsubst %.cpp,$(OBJ_DIR)/sm13_%.cpp,$(CU_CPP))
cu_cpp_sm12_loc = $(patsubst %.cpp,$(OBJ_DIR)/sm12_%.cpp,$(CU_CPP))
cu_cpp_sm11_loc = $(patsubst %.cpp,$(OBJ_DIR)/sm11_%.cpp,$(CU_CPP))

cu_cpp_obj_sm35_loc = $(patsubst %.cpp,$(OBJ_DIR)/sm35_%.cpp.o,$(CU_CPP))
cu_cpp_obj_sm30_loc = $(patsubst %.cpp,$(OBJ_DIR)/sm30_%.cpp.o,$(CU_CPP))
cu_cpp_obj_sm21_loc = $(patsubst %.cpp,$(OBJ_DIR)/sm21_%.cpp.o,$(CU_CPP))
cu_cpp_obj_sm20_loc = $(patsubst %.cpp,$(OBJ_DIR)/sm20_%.cpp.o,$(CU_CPP))
cu_cpp_obj_sm13_loc = $(patsubst %.cpp,$(OBJ_DIR)/sm13_%.cpp.o,$(CU_CPP))
cu_cpp_obj_sm12_loc = $(patsubst %.cpp,$(OBJ_DIR)/sm12_%.cpp.o,$(CU_CPP))
cu_cpp_obj_sm11_loc = $(patsubst %.cpp,$(OBJ_DIR)/sm11_%.cpp.o,$(CU_CPP))

all: mk_libso_no21 mk_libso_21 mk_libso_30 mk_libso_35 mk_lib_fpic 

sm35_command := $(if $(sm_35_support),$(CXX) -shared -o $(LIB_DIR)/libpfac_sm35.so $(LIBS) $(cu_cpp_obj_sm35_loc),)
sm30_command := $(if $(sm_30_support),$(CXX) -shared -o $(LIB_DIR)/libpfac_sm30.so $(LIBS) $(cu_cpp_obj_sm30_loc),)
sm21_command := $(if $(sm_21_support),$(CXX) -shared -o $(LIB_DIR)/libpfac_sm21.so $(LIBS) $(cu_cpp_obj_sm21_loc),)

cu_cpp_sm35_loc := $(if $(sm_35_support),$(patsubst %.cpp,$(OBJ_DIR)/sm35_%.cpp,$(CU_CPP)),)
cu_cpp_sm30_loc := $(if $(sm_30_support),$(patsubst %.cpp,$(OBJ_DIR)/sm30_%.cpp,$(CU_CPP)),)
cu_cpp_sm21_loc := $(if $(sm_21_support),$(patsubst %.cpp,$(OBJ_DIR)/sm21_%.cpp,$(CU_CPP)),)


mk_libso_no21: $(cu_cpp_sm20_loc) $(cu_cpp_sm13_loc) $(cu_cpp_sm12_loc) $(cu_cpp_sm11_loc)
	$(CXX) -shared -o $(LIB_DIR)/libpfac_sm20.so $(LIBS) $(cu_cpp_obj_sm20_loc)
	$(CXX) -shared -o $(LIB_DIR)/libpfac_sm13.so $(LIBS) $(cu_cpp_obj_sm13_loc)
	$(CXX) -shared -o $(LIB_DIR)/libpfac_sm12.so $(LIBS) $(cu_cpp_obj_sm12_loc)
	$(CXX) -shared -o $(LIB_DIR)/libpfac_sm11.so $(LIBS) $(cu_cpp_obj_sm11_loc)

mk_libso_35: $(cu_cpp_sm35_loc)
	$(sm35_command)

mk_libso_30: $(cu_cpp_sm30_loc)
	$(sm30_command)  

mk_libso_21: $(cu_cpp_sm21_loc)
	$(sm21_command)  

mk_liba: $(cppobj_loc)
	ar cru $(LIB_DIR)/libpfac.a  $(cppobj_loc)
	ranlib $(LIB_DIR)/libpfac.a

mk_lib_fpic: $(cppobj_fpic_loc)
	$(CXX) -shared -o $(LIB_DIR)/libpfac.so  $(cppobj_fpic_loc) $(LIBS)


$(OBJ_DIR)/%_fpic.o: %.cpp  $(inc_files)
	$(CXX) -fPIC -c $(CXXFLAGS) $(INCPATH) -o $@ $<

$(OBJ_DIR)/PFAC_CPU_OMP_reorder_fpic.o: PFAC_CPU_OMP_reorder.cpp  $(inc_files)
	$(CXX) -fPIC -c $(CXXFLAGS) $(INCPATH) -o $@ $<

$(OBJ_DIR)/PFAC_CPU_OMP_reorder.o: PFAC_CPU_OMP_reorder.cpp  $(inc_files)
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

$(OBJ_DIR)/%.o: %.cpp  $(inc_files)
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

$(OBJ_DIR)/sm35_%.cu.cpp: %.cu
	$(NVCC) -arch=sm_35 -cuda $(INCPATH) -o $@ $<
	$(CXX) -fPIC -O2 -c -o $@.o $@

$(OBJ_DIR)/sm30_%.cu.cpp: %.cu
	$(NVCC) -arch=sm_30 -cuda $(INCPATH) -o $@ $<
	$(CXX) -fPIC -O2 -c -o $@.o $@

$(OBJ_DIR)/sm21_%.cu.cpp: %.cu
	$(NVCC) -arch=sm_21 -cuda $(INCPATH) -o $@ $<
	$(CXX) -fPIC -O2 -c -o $@.o $@

$(OBJ_DIR)/sm20_%.cu.cpp: %.cu
	$(NVCC) -arch=sm_20 -cuda $(INCPATH) -o $@ $<
	$(CXX) -fPIC -O2 -c -o $@.o $@

$(OBJ_DIR)/sm13_%.cu.cpp: %.cu
	$(NVCC) -arch=sm_13 -cuda  $(INCPATH) -o $@ $<
	$(CXX) -fPIC -O2 -c -o $@.o $@

$(OBJ_DIR)/sm12_%.cu.cpp: %.cu
	$(NVCC) -arch=sm_12 -cuda  $(INCPATH) -o $@ $<
	$(CXX) -fPIC -O2 -c -o $@.o $@

$(OBJ_DIR)/sm11_%.cu.cpp: %.cu
	$(NVCC) -arch=sm_11 -cuda  $(INCPATH) -o $@ $<
	$(CXX) -fPIC -O2 -c -o $@.o $@

#clean :
#	rm -f *.linkinfo
#	rm -f $(OBJ_DIR)/*
#	rm -f $(EXE_DIR)/*


####### Implicit rules

.SUFFIXES: .o .c .cpp .cc .cxx .C

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.C.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

####### Build rules
