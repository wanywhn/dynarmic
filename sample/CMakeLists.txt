set(TARGET add add2 bf bf_loongarch cpuinfo direct_write div dump label mul strcat strcmp strcpy strlen strstr sub)
foreach (src IN ITEMS ${TARGET})
    add_executable(${src} ${src}.cpp)
    find_package(GTest MODULE REQUIRED)
    target_link_libraries(${src} PRIVATE xbyak_loongarch64 GTest::GTest GTest::Main)
    set_target_properties(${src} PROPERTIES
            CXX_STANDARD 11
            CXX_STANDARD_REQUIRED YES
            CXX_EXTENSIONS NO)
    add_test(NAME ${src} COMMAND ${src})
endforeach ()