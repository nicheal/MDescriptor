module mdescriptor_scipy_blas
  use iso_c_binding, only: c_char, c_double, c_int
  implicit none
  interface
    subroutine scipy_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
        bind(C, name="scipy_dgemm_")
      import c_char, c_double, c_int
      character(kind=c_char) :: transa, transb
      integer(c_int) :: m, n, k, lda, ldb, ldc
      real(c_double) :: alpha, a(*), b(*), beta, c(*)
    end subroutine scipy_dgemm
    subroutine scipy_dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info) &
        bind(C, name="scipy_dgesvd_")
      import c_char, c_double, c_int
      character(kind=c_char) :: jobu, jobvt
      integer(c_int) :: m, n, lda, ldu, ldvt, lwork, info
      real(c_double) :: a(*), s(*), u(*), vt(*), work(*)
    end subroutine scipy_dgesvd
    subroutine scipy_dpotrf(uplo, n, a, lda, info) bind(C, name="scipy_dpotrf_")
      import c_char, c_double, c_int
      character(kind=c_char) :: uplo
      integer(c_int) :: n, lda, info
      real(c_double) :: a(*)
    end subroutine scipy_dpotrf
    subroutine scipy_dpotri(uplo, n, a, lda, info) bind(C, name="scipy_dpotri_")
      import c_char, c_double, c_int
      character(kind=c_char) :: uplo
      integer(c_int) :: n, lda, info
      real(c_double) :: a(*)
    end subroutine scipy_dpotri
  end interface
end module mdescriptor_scipy_blas

subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
  use mdescriptor_scipy_blas, only: scipy_dgemm
  character(len=1), intent(in) :: transa, transb
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  real(8), intent(in) :: alpha, a(*), b(*), beta
  real(8), intent(inout) :: c(*)
  call scipy_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
end subroutine dgemm

subroutine dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info)
  use mdescriptor_scipy_blas, only: scipy_dgesvd
  character(len=1), intent(in) :: jobu, jobvt
  integer, intent(in) :: m, n, lda, ldu, ldvt, lwork
  integer, intent(out) :: info
  real(8), intent(inout) :: a(*), s(*), u(*), vt(*), work(*)
  call scipy_dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info)
end subroutine dgesvd

subroutine dpotrf(uplo, n, a, lda, info)
  use mdescriptor_scipy_blas, only: scipy_dpotrf
  character(len=1), intent(in) :: uplo
  integer, intent(in) :: n, lda
  integer, intent(out) :: info
  real(8), intent(inout) :: a(*)
  call scipy_dpotrf(uplo, n, a, lda, info)
end subroutine dpotrf

subroutine dpotri(uplo, n, a, lda, info)
  use mdescriptor_scipy_blas, only: scipy_dpotri
  character(len=1), intent(in) :: uplo
  integer, intent(in) :: n, lda
  integer, intent(out) :: info
  real(8), intent(inout) :: a(*)
  call scipy_dpotri(uplo, n, a, lda, info)
end subroutine dpotri
