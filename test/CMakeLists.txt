set(TARGET make_nm_lasx make_nm_loongarch make_nm_branch make_nm_fp_loongarch mov_imm add_sub_imm meta_mnemonic)
foreach (src IN ITEMS ${TARGET})
    add_executable(${src} ${src}.cpp)
    target_link_libraries(${src} PRIVATE xbyak_loongarch64)
    set_target_properties(${src} PROPERTIES
            CXX_STANDARD 11
            CXX_STANDARD_REQUIRED YES
            CXX_EXTENSIONS NO)
endforeach ()