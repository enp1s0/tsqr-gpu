#ifndef __RUNTIME_STATUS_GIT_HPP__
#define __RUNTIME_STATUS_GIT_HPP__
#include <iostream>
#include <cstdlib>

#ifndef RS_GIT_BRANCH
#define RS_GIT_BRANCH "NOT PROVIDED"
#endif
#ifndef RS_GIT_COMMIT
#define RS_GIT_COMMIT "NOT PROVIDED"
#endif

namespace mtk {
namespace runtime_status {
namespace git {
inline void print_info(const bool print_header = false) {
	if (print_header) {
		std::printf("# git information\n");
	}
	std::printf("%10s : %s\n", "commit", RS_GIT_COMMIT);
	std::printf("%10s : %s\n", "branch", RS_GIT_BRANCH);
}
} // namespace git
} // namespace runtime_status
} // namespace mtk

#endif /* end of include guard */
