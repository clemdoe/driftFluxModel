!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
MODULE PATHS_blas1
      !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      ! Module containing operations for vectors. Similar to BLAS level 1 however
      ! it includes more operations with generic interfaces for data types and 
      ! has actually been observed to outperform Intel's MKL.
      !
      ! Dependencies:
      !      intrinsic_kinds_M.F90 - NBI,NBS,NBD
      !
      ! Global variables: NONE
      !
      ! Contains subprograms: 
      !       -vec_init(r,a,ndat)      : r=a
      !       -vec_copy(r,a,ndat)      : r(i)=a(i)
      !       -vec_sum(r,a,ndat)       : r=r+a(i)
      !       -vec_add(r,a[,b],ndat)   : r(i)=r(i)+a(i)   | r(i)=r(i)+a*b(i)
      !                                | r(i)=r(i)+a(i)*b(i)
      !       -vec_sub(r,a[,b],ndat)   : r(i)=r(i)-a(i)   | r(i)=r(i)-a*b(i)
      !                                | r(i)=r(i)-a(i)*b(i)
      !       -vec_mult(r,a[,b],ndat)  : r(i)=a*r(i)      | r(i)=a(i)*r(i)
      !                                | r(i)=a*r(i)+b    | r(i)=a(i)*r(i)+b 
      !                                | r(i)=a*r(i)+b(i) | r(i)=a(i)*r(i)+b(i)
      !       -vec_div(r,a,ndat)       : r(i)=r(i)/a(i)   | r(i)=r/a(i)
      !       -vec_dotprod(r,a,b,ndat) : r=r+a(i)*b(i)
      !
      ! NOTE: Every PROCEDURE in each INTERFACE performs the same function
      !       (e.g. initialization, copying, etc.). The interfaces are used so that
      !       the programmer does not need a different call statement for each
      !       data-type and if some of the arguments are scalar or vector.
      !
      !       The naming convention for the procedures combines three fields the
      !       first field is "vec". The second field indicates the type of operation
      !       which is sometimes followed by an INTEGER to differentiate between 
      !       scalar/vector input arguments. The third field indicates the data type
      !       of the return value which are:
      !          *_i = operation uses INTEGER numbers
      !          *_s = operation uses SINGLE PRECISION floating point numbers
      !          *_d = operation uses DOUBLE PRECISION floating point numbers
      !          *_l = operation uses LOGICAL data types
      !
      ! NOTE: In general the arguments for vec_* are in the following form: 
      !            (r,a[,b],ndat)
      !       where "r" is the return value, "ndat" is the number of elements in
      !       the vector. "a" and "b" are scalars or vectors depending on the
      !       operation. Note that they must be all be the same data type as "r".
      !
      ! Author: Brendan Kochunas
      !   Date: 07/14/2008
      !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      !        USE intrinsic_kinds, ONLY: NBI,NBS,NBD
      USE IntrType, ONLY: sdk, sik
      IMPLICIT NONE
      !
      INTEGER(sik),PRIVATE :: i,j,k,l,c,ndat
      !
      INTERFACE vec_copy !copy a vector to another vector
         !MODULE PROCEDURE vec_copy_i
         !          MODULE PROCEDURE vec_copy_s
         MODULE PROCEDURE vec_copy_d
         !          MODULE PROCEDURE vec_copy_l
      ENDINTERFACE
      !
      INTERFACE vec_dotprod !get inner product of two vectors
         !MODULE PROCEDURE vec_dotprod_i
         !          MODULE PROCEDURE vec_dotprod_s
         MODULE PROCEDURE vec_dotprod_d
      ENDINTERFACE
      !
CONTAINS
      !
      !==============================================================================
      ! The following section contains all procedures for copying a vector
      ! (*Unused subroutines removed*)
      !----------------------------------------
      SUBROUTINE vec_copy_d(r,a,ndat)
            !
            INTEGER(sik),INTENT(in) :: ndat
            REAL(sdk),INTENT(in) :: a(ndat)
            REAL(sdk),INTENT(inout) :: r(ndat)
            !
            DO i=1,ndat
               r(i)=a(i)
            ENDDO
            !
      ENDSUBROUTINE vec_copy_d
      !
      !==============================================================================
      ! The following section contains all procedures for obtaining the inner product
      ! of two vectors. (*Unused subroutines removed*)
      !
      SUBROUTINE vec_dotprod_d(r,a,b,ndat)
            !
            INTEGER(sik),INTENT(in) :: ndat
            REAL(sdk),INTENT(in) :: a(ndat),b(ndat)
            REAL(sdk),INTENT(out) :: r
            !
            r=0.0_sdk
            DO i=1,ndat
               r=r+a(i)*b(i)
            ENDDO
            !
      ENDSUBROUTINE vec_dotprod_d
      !
END MODULE PATHS_blas1
