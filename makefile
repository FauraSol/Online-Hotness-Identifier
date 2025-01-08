LTROOT += ../libtorch

INCLUDE += $(LTROOT)/include

INCLUDEFLAG += -I$(INCLUDE) -I$(INCLUDE)/torch/csrc/api/include -Ilibtorch/include

CPLFLAG += $(INCLUDEFLAG) -std=c++17 -L$(LTROOT)/lib -lc10 -lfasttext -ltorch_cpu

test : 
	g++-9 libtorch/TestCompile.cc -o test $(CPLFLAG)

OnlineLSTM : 
	g++-9 libtorch/OnlineLSTM.cpp -o OnlineLSTM $(CPLFLAG)

BLSTM :
	g++-9 test_examples/BLSTM.cpp -o BLSTM $(CPLFLAG)