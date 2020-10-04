#include <iostream>
#include <runtime_staus/git.hpp>

constexpr bool print_header = true;

int main() {
	mtk::runtime_status::git::print_info(print_header);
}
