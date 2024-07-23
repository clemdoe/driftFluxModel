!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
MODULE PATHS_blas2
      !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      ! This module contains routines for some basic linear algebra operations 
      ! similar to Level-2 BLAS routines (e.g. matrix and matrix-vector operations).
      !
      ! Dependencies:
      !     intrinsic_kinds_M.F90 - sik,sdk
      !     blas1_M.F90 - vec_copy
      !
      ! Global variables:
      !
      ! Contains subprograms:
      !      ilufac - subroutine computes the incomplete LU factorization of a matrix
      !         axb - subroutine applies matrix A to a vector
      !        minv - subroutine applies inv(M) to a vector
      !        inva - subroutine inverts a square matrix
      !
      ! Author: Brendan Kochunas
      !   Date: 07/09/2009
      !
      !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      !      USE intrinsic_kinds, ONLY: NBI,NBF
      USE IntrType, ONLY: sdk, sik
      USE PATHS_blas1, ONLY: vec_copy
      IMPLICIT NONE

      REAL(sdk),PRIVATE,PARAMETER :: zero=0.0_sdk
      REAL(sdk),PRIVATE,PARAMETER :: one=1.0_sdk

      INTERFACE axb
         MODULE PROCEDURE axb_csr
         !MODULE PROCEDURE axb_full
      ENDINTERFACE

CONTAINS
      !
      !==============================================================================
      SUBROUTINE axb_csr(n,aa,ia,ja,x,b)
            !
            ! Subroutine applies the maxtrix A to a vector x and returns b.
            ! It assumes that A is stored in Compressed Sparse Row (CSR) format.
            !   n - dimension of matrix A
            !   aa - array of non-zero elements of A in CSR format
            !   ia - indexing array for aa
            !   ja - indexing array for aa CONTAINS column index
            !   x - input vector
            !   b - output vector b=Ax
            !
            REAL(sdk),POINTER :: aa(:),x(:),b(:)
            INTEGER(sik),POINTER :: ia(:),ja(:)
            INTEGER(sik),INTENT(in) :: n

            INTEGER(sik) :: i,j,k
            REAL(sdk) :: sum
            !
            ! Compute b for Ax=b        
            DO i=1,n
               sum=zero
               DO k=ia(i),ia(i+1)-1
                  j=ja(k)
                  sum=sum+aa(k)*x(j)
               ENDDO
               b(i)=sum
            ENDDO
            !
            RETURN
      ENDSUBROUTINE axb_csr
      !
      !==============================================================================
      SUBROUTINE minv(n,mm,im,jm,b,x)
            !
            ! Subroutines applies the inverse of a matrix M to a vector b and returns x.
            ! It assumes that M is stored in the Compressed Sparse Row (CSR) format, and
            ! that M is actually stored as LU.
            !
            ! inv(M) is applied to b by solving Mx=b
            !  n - dimension of matrix M
            !  mm - array of non-zero elements of L and U in CSR format for M=LU
            !       The diagonal and super-diagonal terms in mm are U, the subdiagonal
            !       terms in mm are L
            !  im - indexing array for mm
            !  jm - indexing array for mm CONTAINS column index
            !  b - input vector
            !  x - output vector x=inv(M)*b
            !       
            INTEGER(sik),INTENT(in) :: n
            INTEGER(sik),POINTER :: im(:),jm(:)
            REAL(sdk),POINTER :: x(:),mm(:),b(:)

            INTEGER(sik) :: i,j,k,d
            REAL(sdk),DIMENSION(n) :: y,uptr
            REAL(sdk) :: sum
            !
            ! Solve Ly=b for y
            DO i=1,n
               sum=zero
               DO k=im(i),im(i+1)-1
                  j=jm(k)
                  IF(i.EQ.j) THEN
                     uptr(i)=k
                     EXIT
                  ENDIF
                  sum=sum+mm(k)*y(j)
               ENDDO
               y(i)=b(i)-sum;
            ENDDO
            !
            ! Solve Ux=y for x
            DO i=n,1,-1
               sum=zero
               d=uptr(i)
               DO k=d+1,im(i+1)-1
                  j=jm(k)
                  sum=sum+mm(k)*x(j)
               ENDDO
               x(i)=(y(i)-sum)/mm(d);
            ENDDO
            !
            RETURN
      ENDSUBROUTINE minv
      !
      !==============================================================================
      SUBROUTINE ilufac(n,aa,ia,ja,bilu)
            !
            ! Performs incomplete LU factorization of a matrix
            ! Matrix is assumed to be stored in Compressed Sparse Row (CSR) format
            !   n - dimension of matrix A
            !   aa - array of non-zero elements of A in CSR format
            !   ia - indexing array for aa
            !   ja - indexing array for aa CONTAINS column index
            !   bilu - the ilu factorization of A stored in CSR format
            !
            ! Local variables
            !  i - row index of A
            !  j - column index of A
            !  k - index in A
            !  kk - index in aa of the kth diagonal A(k,k) == aa(kk)
            !  kj - index in aa of A(k,j)
            !  ij - index in aa of A(i,j)
            !  ik - index in aa of A(i,k)
            !
            INTEGER(sik),INTENT(in) :: n
            INTEGER(sik),POINTER :: ia(:),ja(:)
            REAL(sdk),POINTER :: aa(:),bilu(:)

            INTEGER(sik) :: i,j,k,kk,kj,ij,ik,j2,nn0
            REAL(sdk) :: m
            INTEGER(sik) :: uptr(n)

            nn0=ia(n+1)-1
            CALL vec_copy(bilu,aa,nn0)
            !
            ! Find the indeces in aa containing the diagonal terms
            DO i=1,n
               DO j=ia(i),ia(i+1)-1
                  IF(i.EQ.ja(j)) THEN
                     uptr(i)=j;
                     EXIT
                  ENDIF
               ENDDO
            ENDDO
            !
            ! Compute the ILU of aa        
            DO i=2,n
               DO ik=ia(i),uptr(i)-1
                  k=ja(ik)
                  kk=uptr(k)
                  m=bilu(ik)/bilu(kk)
                  bilu(ik)=m
                  DO ij=ik+1,ia(i+1)-1
                     j=ja(ij)
                     kj=0
                     DO j2=ia(k),ia(k+1)-1
                        IF(j.EQ.ja(j2)) THEN
                           kj=j2
                           EXIT
                        ENDIF
                     ENDDO
                     IF(kj.NE.0) THEN
                        bilu(ij)=bilu(ij)-m*bilu(kj)
                     ENDIF
                  ENDDO
               ENDDO
            ENDDO
            !
            RETURN
      ENDSUBROUTINE ilufac
      !
END MODULE PATHS_blas2

