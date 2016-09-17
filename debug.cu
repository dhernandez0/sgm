#include "debug.h"

void debug_log(const char *str) {
#if LOG
	std::cout << str << std::endl;
#endif
}
