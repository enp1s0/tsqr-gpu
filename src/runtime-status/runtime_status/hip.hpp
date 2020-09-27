#ifndef __RUNTIME_STATUS_CUDA_HPP__
#define __RUNTIME_STATUS_CUDA_HPP__
#include <hip/hip_runtime.h>
#include <iostream>

namespace mtk {
namespace runtime_status {
namespace hip {
inline void print_info(const bool print_header = false) {
	if (print_header) {
		std::printf("# HIP information\n");
	}
	int num_device;
	hipGetDeviceCount(&num_device);
	for(int i = 0; i < num_device; i++){
		hipDeviceProp_t property;
		hipGetDeviceProperties(&property, i);
		std::printf("## -- Device %d\n", i);
		std::printf("%13s : %s\n", "Name", property.name);
		std::printf("%13s : %d [kHz]\n", "Clock", property.clockRate);
		std::printf("%13s : %e [GB]\n", "Memory", property.totalGlobalMem / static_cast<float>(1lu << 30));
	}
}

inline void print_current_device_id(const bool print_header = false) {
	if (print_header) {
		std::printf("# HIP information\n");
	}
	std::printf("## -- Using GPU ID\n");
	int device_id;
	hipGetDevice(&device_id);
	std::printf("%13s : %d\n", "ID", device_id);
}
} // namespace hip
} // namespace runtime_status
} // namespace mtk

#endif /* end of include guard */
