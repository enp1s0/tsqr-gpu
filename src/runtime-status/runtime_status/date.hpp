#ifndef __RUNTIME_STATUS_DATE_HPP__
#define __RUNTIME_STATUS_DATE_HPP__
#include <iostream>
#include <cstdlib>
#include <ctime>


namespace mtk {
namespace runtime_status {
namespace date {
namespace detail {
std::string unixtime2string(const std::time_t unix_time) {
	constexpr unsigned buffer_size = 32;
	char buffer[buffer_size];
	const auto lt = std::localtime(&unix_time);
	std::strftime(buffer, buffer_size, "%c", lt);
	return std::string{buffer};
}
} // namespace detail
inline void print_info(const bool print_header = false) {
	if (print_header) {
		std::printf("# date information\n");
	}
#ifndef RS_DATE_BUILD
	std::printf("%10s : %s\n", "build", "Not provided");
#else
	std::printf("%10s : %s\n", "build", detail::unixtime2string(std::stoul(RS_DATE_BUILD)).c_str());
#endif
	std::printf("%10s : %s\n", "run", detail::unixtime2string(std::time(nullptr)).c_str());
}
} // namespace date
} // namespace runtime_status
} // namespace mtk

#endif /* end of include guard */
