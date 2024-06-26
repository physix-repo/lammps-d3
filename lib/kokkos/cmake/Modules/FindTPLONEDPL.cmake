INCLUDE(CheckIncludeFileCXX)
CHECK_INCLUDE_FILE_CXX(oneapi/dpl/execution KOKKOS_COMPILER_HAS_ONEDPL_EXECUTION_HEADER)
CHECK_INCLUDE_FILE_CXX(oneapi/dpl/algorithm KOKKOS_COMPILER_HAS_ONEDPL_ALGORITHM_HEADER)

INCLUDE(CheckCXXSourceCompiles)
CHECK_CXX_SOURCE_COMPILES("
  #include <iostream>

  int main()
  {
    #if defined(_GLIBCXX_RELEASE) && (_GLIBCXX_RELEASE == 9 || _GLIBCXX_RELEASE == 10)
      static_assert(false);
    #endif
    return 0;
  }"
  KOKKOS_NO_TBB_CONFLICT)

IF (KOKKOS_COMPILER_HAS_ONEDPL_EXECUTION_HEADER AND KOKKOS_COMPILER_HAS_ONEDPL_ALGORITHM_HEADER)
  IF(KOKKOS_NO_TBB_CONFLICT)
    KOKKOS_CREATE_IMPORTED_TPL(
      ONEDPL INTERFACE
    )
  ELSE()
    KOKKOS_CREATE_IMPORTED_TPL(
      ONEDPL INTERFACE
      # https://stackoverflow.com/questions/67923287/how-to-resolve-no-member-named-task-in-namespace-tbb-error-when-using-oned/
      COMPILE_DEFINITIONS PSTL_USE_PARALLEL_POLICIES=0 _GLIBCXX_USE_TBB_PAR_BACKEND=0
    )
  ENDIF()
ELSE()
  FIND_PACKAGE(oneDPL REQUIRED)

  IF(KOKKOS_NO_TBB_CONFLICT)
    KOKKOS_CREATE_IMPORTED_TPL(
      ONEDPL INTERFACE
      LINK_LIBRARIES oneDPL
    )
  ELSE()
    KOKKOS_CREATE_IMPORTED_TPL(
      ONEDPL INTERFACE
      LINK_LIBRARIES oneDPL
      # https://stackoverflow.com/questions/67923287/how-to-resolve-no-member-named-task-in-namespace-tbb-error-when-using-oned/
      COMPILE_DEFINITIONS PSTL_USE_PARALLEL_POLICIES=0 _GLIBCXX_USE_TBB_PAR_BACKEND=0
    )
  ENDIF()

  # Export oneDPL as a Kokkos dependency
  KOKKOS_EXPORT_CMAKE_TPL(oneDPL)
ENDIF()
