#include <iostream>
#include <runtime_staus/hip.hpp>

constexpr bool print_header = true;

int main() {
	mtk::runtime_status::hip::print_info(print_header);
	mtk::runtime_status::hip::print_current_device_id(print_header);
}
