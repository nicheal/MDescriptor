module mdescriptor_scipy_lapack_extra
  use iso_c_binding, only: c_char, c_double, c_int
  implicit none
  interface
    subroutine scipy_dsytrf(uplo, n, a, lda, ipiv, work, lwork, info) &
        bind(C, name="scipy_dsytrf_")
      import c_char, c_double, c_int
      character(kind=c_char) :: uplo
      integer(c_int) :: n, lda, ipiv(*), lwork, info
      real(c_double) :: a(*), work(*)
    end subroutine scipy_dsytrf
    subroutine scipy_dsytri(uplo, n, a, lda, ipiv, work, info) &
        bind(C, name="scipy_dsytri_")
      import c_char, c_double, c_int
      character(kind=c_char) :: uplo
      integer(c_int) :: n, lda, ipiv(*), info
      real(c_double) :: a(*), work(*)
    end subroutine scipy_dsytri
  end interface
end module mdescriptor_scipy_lapack_extra

subroutine dsytrf(uplo, n, a, lda, ipiv, work, lwork, info)
  use mdescriptor_scipy_lapack_extra, only: scipy_dsytrf
  character(len=1), intent(in) :: uplo
  integer, intent(in) :: n, lda, lwork
  integer, intent(out) :: ipiv(*), info
  real(8), intent(inout) :: a(*), work(*)
  call scipy_dsytrf(uplo, n, a, lda, ipiv, work, lwork, info)
end subroutine dsytrf

subroutine dsytri(uplo, n, a, lda, ipiv, work, info)
  use mdescriptor_scipy_lapack_extra, only: scipy_dsytri
  character(len=1), intent(in) :: uplo
  integer, intent(in) :: n, lda
  integer, intent(in) :: ipiv(*)
  integer, intent(out) :: info
  real(8), intent(inout) :: a(*), work(*)
  call scipy_dsytri(uplo, n, a, lda, ipiv, work, info)
end subroutine dsytri
