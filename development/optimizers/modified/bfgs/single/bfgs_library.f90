!******************************************************************************
!******************************************************************************
MODULE bfgs_library

    USE bfgs_function, ONLY: dfpmin 
    USE shared_constants
    USE criterion_function, ONLY: criterion_func, criterion_dfunc
!******************************************************************************
!******************************************************************************
END MODULE