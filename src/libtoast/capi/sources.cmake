
include(MacroDefineModule)

file(GLOB TOAST_HEADERS_H   ${CMAKE_CURRENT_LIST_DIR}/*.h)
file(GLOB TOAST_HEADERS_HPP ${CMAKE_CURRENT_LIST_DIR}/*.hpp)

DEFINE_MODULE(NAME libtoast.capi
    HEADERS     ${TOAST_HEADERS_H} ${TOAST_HEADERS_HPP}
    HEADER_EXT  ".hpp;.hh;.h"
    SOURCE_EXT  ".cpp;.cc;.c"
)

install(FILES ${TOAST_HEADERS_H} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
