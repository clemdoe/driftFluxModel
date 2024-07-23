!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Module contains routines and variables related to the Bi-CGstab algorithm for
! solving linear systems of equations. It is assumed the coefficient matrix is
! stored in the Compressed Sparse Row (CSR) format.
!
! Dependencies:
!     intrinsic_kinds_M.F90 - NBI,NBF
!     blas1_M.F90 - vec_dotprod
!     blas2_M.F90 - axb,minv
!     
! Global variables: none
! 
! Contains subprograms:

!    initbicg - subroutine initializes the vectors needed for the Bi-CGstab
!               iteration
!    bicgstb1 - Performs 1 Bi-CGstab iteration
!
! Author: Brendan Kochunas
!   Date: 06/19/2009
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!      USE intrinsic_kinds, ONLY: NBI,NBF
MODULE PATHS_bicgstab

      USE IntrType, ONLY: sdk, sik
      USE PATHS_blas1, ONLY: vec_dotprod
      USE PATHS_blas2, ONLY: axb,minv
      IMPLICIT NONE

      REAL(sdk),PRIVATE,PARAMETER :: one=1.0_sdk,zero=0.0_sdk
      REAL(sdk),PRIVATE,SAVE :: calpha,crho,comega
      REAL(sdk),POINTER,DIMENSION(:),PRIVATE,SAVE :: vr,vr0,vs,vv,vp,vy,vz,vt

CONTAINS
      !
      !==============================================================================
      SUBROUTINE initbicg(n,aa,ia,ja,x,b,r20)
            !
            ! Initialize bi-cgstab vectors and residuals for a given linear system, Ax=b.
            !   n - DIMENSION of matrix A
            !   aa - array of non-zero elements of A in CSR format
            !   ia - indexing array for aa
            !   ja - indexing array for aa contains column index
            !   x - initial guess of solution vector
            !   b - RHS of of linear system
            !   r20 - output variable, norm of residual vector.
            !
            INTEGER(sik),INTENT(in) :: n
            REAL(sdk),POINTER :: x(:),aa(:),b(:)
            INTEGER(sik),POINTER :: ia(:),ja(:)
            REAL(sdk),INTENT(out) :: r20

            INTEGER(sik) :: i
            LOGICAL,SAVE :: needdeallocate=.FALSE.

            IF(needdeallocate) THEN
               DEALLOCATE(vr)
               DEALLOCATE(vr0)
               DEALLOCATE(vs)
               DEALLOCATE(vz)
               DEALLOCATE(vt)
               DEALLOCATE(vp)
               DEALLOCATE(vy)
               DEALLOCATE(vv)
            ENDIF
            ALLOCATE(vr(1:n))    ! residual vector for ith iteration r_j=b-A*x_j
            ALLOCATE(vr0(1:n))   ! initial residual vector r_0=b-A*x_0
            ALLOCATE(vs(1:n))    ! vector s in bicgstab algorithm
            ALLOCATE(vz(1:n))    ! z=inv(M)*s
            ALLOCATE(vt(1:n))    ! t=A*z=A*inv(M)*s
            ALLOCATE(vp(1:n))    ! vector p in bicgstab algorithm
            ALLOCATE(vy(1:n))    ! y=inv(M)*p
            ALLOCATE(vv(1:n))    ! v=A*y=A*inv(M)*p
            needdeallocate=.TRUE.

            calpha=one                ! alpha in bicgstab algorithm
            crho=one                  ! inner product (r0,r_i)
            comega=one                ! omega in bicgstab algorithm
            r20=zero                  ! norm of residual
            CALL axb(n,aa,ia,ja,x,vp) ! temporarily USE p to store A*x to compute
            ! r and r0

            DO i=1,n
               vr0(i)=b(i)-vp(i)       ! r0=b-A*x_0
               vr(i)=vr0(i)
               r20=r20+vr0(i)*vr0(i)
               vp(i)=zero
               vv(i)=zero
            ENDDO
            r20=SQRT(r20)

            RETURN
      ENDSUBROUTINE initbicg
      !
      !==============================================================================
      SUBROUTINE bicgstb1(n,aa,mm,ia,ja,x,r2)
            !
            ! Perform 1 Bi-CGstab iteration to solve linear system Ax=b with 
            ! right-preconditioner inv(M). So A*inv(M)*y=b and x=inv(M)*y
            !                                 A*inv(M)*M*x=b
            !
            !   n - DIMENSION of matrix A
            !   aa - array of non-zero elements of A in CSR format
            !   mm - preconditioner M stored as LU in CSR format
            !   ia - indexing array for aa and mm
            !   ja - indexing array for aa and mm contains column index
            !    x - solution vector
            !   r2 - norm of residual vector 
            !

            INTEGER(sik),INTENT(in) :: n
            INTEGER(sik),POINTER :: ia(:),ja(:)
            REAL(sdk),POINTER :: aa(:),mm(:),x(:)
            REAL(sdk),INTENT(out) :: r2

            INTEGER(sik) :: i
            REAL(sdk) :: crhod,cbeta,r0v,pts,ptt

            crhod=crho                               ! rho_j-1
            CALL vec_dotprod(crho,vr0,vr,n)          ! rho_j = (r_j,r_0)
            cbeta=crho*calpha/(crhod*comega)         ! beta_j = (r_j,r_0)/(r_j-1,r_0)
            !          *alpha_j-1/omega_j-1

            DO i=1,n
               vp(i)=vr(i)+cbeta*(vp(i)-comega*vv(i)) ! p_j = r_j+beta_j*(p_j-1 - omega_j-1*v_j-1)
            ENDDO                                    ! v_j-1 = A*y_j-1 = A*inv(M)*p_j-1

            CALL minv(n,mm,ia,ja,vp,vy)              ! y_j = inv(M)*p_j
            CALL axb(n,aa,ia,ja,vy,vv)               ! v_j = A*y_j = A*inv(M)*p_j

            CALL vec_dotprod(r0v,vv,vr0,n)
            calpha=crho/r0v                          ! alpha_j = (r_j,r_0)/(v_j,r_0)
            !         = (r_j,r_0)/(A*inv(M)*p,r_0)
            DO i=1,n
               vs(i)=vr(i)-calpha*vv(i)               ! s_j = r_j - alpha*v_j
            ENDDO                                    !     = r_j - alpha*A*inv(M)*p_j
            CALL minv(n,mm,ia,ja,vs,vz)              ! z_j = inv(M)*s_j
            CALL axb(n,aa,ia,ja,vz,vt)               ! t_j = A*inv(M)*s_j

            pts=zero
            ptt=zero
            DO i=1,n
               pts=pts+vs(i)*vt(i)
               ptt=ptt+vt(i)*vt(i)
            ENDDO
            comega=pts/ptt           ! omega_j = (t_j,s_j)/(t_j,t_j)
            !         = (A*inv(M)*s_j,s_j)/(A*inv(M)*s_j,(A*inv(M)*s_j)
            r2=zero
            DO i=1,n
               x(i)=x(i)+calpha*vy(i)+comega*vz(i)  ! x_j = x_j-1 + alpha_j*y_j+omega_j*z_j
               !     = x_j-1 + alpha_j*inv(M)*p_j+omega_j*inv(M)*s_j
               vr(i)=vs(i)-comega*vt(i)             ! r_j = s_j - omega_j*t_j
               !     = s_j - omega_j*A*inv(M)*s_j
               r2=r2+vr(i)*vr(i)
            ENDDO
            r2=SQRT(r2)
            !
            RETURN
      ENDSUBROUTINE bicgstb1
      !
END MODULE PATHS_bicgstab

