add_library(lapjv STATIC
	lapjv/lapjv.cpp
	lapjv/lapjv.h
)
target_include_directories(lapjv
	PRIVATE
		lapjv
)
set_property(TARGET lapjv PROPERTY POSITION_INDEPENDENT_CODE TRUE)
