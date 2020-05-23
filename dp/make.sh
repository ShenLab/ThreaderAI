
g++ threading.cpp types.h ranker.h ranker.cpp \
	-o threading \
	-lgflags -lpthread -lglog -lcnpy \
	--std=c++11
