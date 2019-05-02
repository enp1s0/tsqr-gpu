#include <iostream>
#include <string>
#include <cmath>

int main(int argc, char **argv){
	std::size_t m = 1999;
	if(argc >= 2){
		m = std::stoul(argv[1]);
	}
	std::cout<<"m  : "<<m<<std::endl;

	const auto batch_size = std::max(5u, static_cast<unsigned>( std::ceil( std::log2(static_cast<float>(m))))) - 5u;
	std::cout<<"bs : "<<batch_size<<std::endl;
	std::cout<<"sm : "<<(m / (1<<batch_size))<<std::endl;
}
