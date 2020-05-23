
g++ threading.cpp types.h ranker.h ranker.cpp \
	-o threading \
	-O3 \
	-lgflags -lpthread -lglog -lcnpy \
	--std=c++11
