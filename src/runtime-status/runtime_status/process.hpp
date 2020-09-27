#ifndef __REGISTER_PREFIX__UNTIME_STATUS_PROCESS_HPP__
#define __REGISTER_PREFIX__UNTIME_STATUS_PROCESS_HPP__

#include "../proc-status/proc_status.hpp"

namespace mtk {
namespace runtime_status {
namespace process {
void print_info(const bool print_header = false) {
	proc_status::proc_status proc_status;
	proc_status.load_self_info();

	if (print_header) {
		std::printf("# process information\n");
	}
	std::printf("%10s : %s\n", "Name", proc_status.get_Name().c_str());
	std::printf("%10s : %s\n", "State", proc_status.get_State().c_str());
}

void print_using_memory_size(const bool print_header = false) {
	proc_status::proc_status proc_status;
	proc_status.load_self_info();

	if (print_header) {
		std::printf("# process information\n");
	}
	std::printf("%10s : %u\n", "Vmem", proc_status.get_VmSize());
}
} // namespace process
} // namespace runtime_status
} // namespace mtk
#endif /* end of include guard */
