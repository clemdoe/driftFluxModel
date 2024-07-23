!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
MODULE PATHS_linsolve
      !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      ! This module contains routines for solving linear systems of equations. It
      ! assumes the coefficient matrix is stored in the Compressed Sparse Row (CSR)
      ! format. All routines require the same input and give the same output.
      !
      ! Dependencies:
      !     intrinsic_kinds_M.F90 - NBI,NBF
      !     blas2_M.F90 - axb,ilufac
      !     bicgstab_M.F90 - initbic,bicgstb1
      !
      ! Global variables: none
      !
      ! Contains subprograms:
      !     solve_gs - solves the given linear system using Gauss-Seidel Iteration
      !     solve_bicg - solves the given linear system using the Bi-CGstab algorithm
      !                  with an ILU preconditioner
      !
      ! Author: Brendan Kochunas
      !   Date: 7/09/2009
      !
      !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      !      USE intrinsic_kinds, ONLY: NBI,NBF
      USE IntrType, ONLY: sdk, sik
      USE PATHS_blas2, ONLY: axb,ilufac
      USE PATHS_bicgstab, ONLY: initbicg,bicgstb1
      IMPLICIT NONE

      REAL(sdk),PARAMETER,PRIVATE :: zero=0.0_sdk

CONTAINS
      !
      !==============================================================================
      SUBROUTINE solve_gs(epscrt,itrmx,n,aa,ia,ja,b,x,erout,nout)
            !
            ! Subroutine solves a linear system of equations using a Gauss-Siedel
            ! iteration scheme. It assumes the coefficient matrix is stored in the 
            ! Compressed Sparse Row (CSR) format.
            !
            ! Finds x for Ax=b given A and b
            !
            !  ercrt - convergence criteria for solution (2-norm)
            !  itrmx - maximum number of G-S iterations to perform
            !  n - order of linear system (# of equations)
            !  aa - coefficient matrix A in CSR format
            !  ia - row indexing array for aa
            !  ja - column mapping index array for aa
            !  b - RHS of linear system
            !  x - unknown vector to solve for
            !  erout - relative error reduction in x upon return
            !  nout - # of G-S iterations performed upon return
            !
            !  i - row index
            !  j - column index
            !  k - CSR index for aa
            !  m - G-S iteration count
            !  d - index in aa for diagonal element of row i
            !
            REAL(sdk),INTENT(in) :: epscrt
            INTEGER(sik),INTENT(in) :: itrmx,n
            INTEGER(sik),POINTER :: ia(:),ja(:)
            REAL(sdk),POINTER :: aa(:),b(:),x(:)
            REAL(sdk),INTENT(out) :: erout
            INTEGER(sik),INTENT(out) :: nout

            INTEGER(sik) :: i,j,k,m,d
            REAL(sdk) :: sum,esum,xold,err0,err
            REAL(sdk),POINTER :: ax(:)

            esum=zero
            err0=zero
            ALLOCATE(ax(1:n))
            !
            ! Compute initial residual
            CALL axb(n,aa,ia,ja,x,ax) !Compute Ax
            DO i=1,n
               err=b(i)-ax(i)
               err0=err0+err*err
            ENDDO
            err0=SQRT(err0)
            DEALLOCATE(ax)
            !
            ! Gauss-Seidel Iteration
            DO m=1,itrmx
               esum=zero
               DO i=1,n
                  xold=x(i)
                  sum=zero
                  DO k=ia(i),ia(i+1)-1
                     j=ja(k)
                     IF(j.EQ.i) THEN
                        d=k
                     ELSE
                        sum=sum+aa(k)*x(j)
                     ENDIF
                  ENDDO
                  x(i)=(b(i)-sum)/aa(d)
                  xold=xold-x(i)
                  esum=esum+xold*xold !compute 2-norm of relative error
               ENDDO
               erout=SQRT(esum)
               IF(erout.LE.epscrt) THEN
                  EXIT
               ENDIF
            ENDDO
            nout=m-1
            erout=erout/err0
            !
            RETURN
      ENDSUBROUTINE solve_gs
      !
      !------------------------------------------------------------------------------
      SUBROUTINE solve_bicg(epscrt,itrmx,n,aa,ia,ja,b,x,erout,nout)
            !
            ! Solves a system of equations using the Bi-CGstab algorithm
            ! with an ILU preconditioner. It assumes the coefficient matrix is stored in 
            ! the Compressed Sparse Row (CSR) format.
            !
            ! Finds x for Ax=b given A and b
            !
            !  ercrt - convergence criteria for solution (L-2 norm)
            !  itrmx - maximum number of Bi-CGstab iterations to perform
            !  n - order of linear system (# of equations)
            !  aa - coefficient matrix A in CSR format
            !  ia - row indexing array for aa
            !  ja - column mapping index array for aa
            !  b - RHS of linear system
            !  x - unknown vector to solve for
            !  erout - relative error reduction in x upon return
            !  nout - # of Bi-CGstab iterations performed upon return
            !
            REAL(sdk),INTENT(in) :: epscrt
            INTEGER(sik),INTENT(in) :: itrmx,n
            INTEGER(sik),POINTER :: ia(:),ja(:)
            REAL(sdk),POINTER :: aa(:),b(:),x(:)
            REAL(sdk),INTENT(out) :: erout
            INTEGER(sik),INTENT(out) :: nout

            INTEGER(sik) :: iin,nn0
            REAL(sdk) :: r2i,r20
            REAL(sdk),POINTER :: ailu(:)
            !
            ! Compute Preconditioner for BiCGSTAB using incomplete LU factorization.
            nn0=ia(n+1)-1
            ALLOCATE(ailu(1:nn0))
            CALL ilufac(n,aa,ia,ja,ailu)
            !
            ! Initialize BiCGSTAB variables for 1-group fixed source problem
            CALL initbicg(n,aa,ia,ja,x,b,r20)
            !
            ! Iterate until converged
            DO iin=1,itrmx
               CALL bicgstb1(n,aa,ailu,ia,ja,x,r2i)
               IF(r2i.LT.epscrt*r20) THEN
                  EXIT
               ENDIF
            ENDDO
            nout=iin
            erout=r2i/r20
            DEALLOCATE(ailu)
            !
            RETURN
      ENDSUBROUTINE solve_bicg
      !
END MODULE PATHS_linsolve
