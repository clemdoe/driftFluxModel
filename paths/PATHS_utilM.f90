!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!     This Software was developed under NRC Contract no. NRC-04-07-116 (Task #1) !
!--------------------------------------------------------------------------------!
MODULE PATHS_util
      
      USE IntrType, ONLY: sdk, sik
      USE PATHS_varcpld, ONLY: mxncol, BLANK, zero, &
                               oneline, sc, AST, BANG,  &
                               fid_out

      IMPLICIT NONE
      
      CHARACTER(len=80) ::  mesg      
      
CONTAINS

      FUNCTION gdsum(f1,i1,f2)
!            USE IntrType, ONLY: sdk, sik
            IMPLICIT NONE
            REAL(sdk) :: f1(10), f2(10), gdsum
            INTEGER(sik) :: i1
            gdsum=f1(i1)
            RETURN
      END FUNCTION gdsum

      FUNCTION gdhigh(f1,i1,f2)
!            USE IntrType, ONLY: sdk, sik
            IMPLICIT NONE
            REAL(sdk) :: gdhigh, f1, f2(10)
            INTEGER(sik) :: i1
            gdhigh=f1
            RETURN
      END FUNCTION gdhigh

      FUNCTION dclock()
!            USE IntrType, ONLY: sdk, sik
!      USE ifport, ONLY: etime
!            USE paramM
            IMPLICIT NONE
            REAL(sdk) :: dclock, dum
            REAL(sdk) :: ta(2)
            dclock = 0.0
            call cpu_time(ta(1))   !removed #if defined LINUX || LINUX_NW
!      dum=etime(ta)
            dclock=ta(1)
            RETURN
      END FUNCTION dclock      
      
      FUNCTION form7(b,ncol)
!            USE IntrType, ONLY: sdk, sik, ONLY: sdk, sik
            !
            IMPLICIT NONE
            !
            !    determines the form
            !
            CHARACTER(len=1) :: a(1024)
            CHARACTER(len=1024) :: b
            CHARACTER(len=7) :: form7
            INTEGER(sik) :: i, ncol, inn
            LOGICAL :: fncol

            DO i = 1, 1024
               a(i) = b(i:i)
            END DO
            ncol = 0
            fncol = .FALSE.
            DO i=mxncol,1,-1
               IF(a(i).NE.BLANK) THEN
                  fncol = .TRUE.
                  EXIT
               END IF
            ENDDO
            IF(fncol) THEN
               ncol=i
            ELSE
               i=1
            END IF

            WRITE(form7,'("(a",i3,")")') ncol

            RETURN
      END FUNCTION form7      

      FUNCTION nfields(aline)
            !
!            USE IntrType, ONLY: sdk, sik, ONLY: sdk, sik

            IMPLICIT NONE

            CHARACTER(len=1) :: quotation
            CHARACTER(len=1024) ::  aline

            LOGICAL :: nonblankd,nonblank,multidata, nonpair
            INTEGER(sik) :: nfields, ncol, n, i, multidcol, nmult
            INTEGER(sik) :: fnd_bang
            DO i = 1, mxncol
               sc(i) = oneline(i:i)
            END DO
            !
            nonblankd=.FALSE.
            multidata=.FALSE.
            oneline=aline
            ncol=LEN_TRIM(aline)
            fnd_bang = 0
            DO i = 1, ncol
                IF (oneline(i:i) .EQ. BANG) THEN
                    fnd_bang = i
                    EXIT
                END IF    
            END DO
            IF (fnd_bang .GT. 0) ncol = fnd_bang - 1
            n=0
            i=ncol
            DO WHILE(i>0)
               IF(ICHAR(sc(i))==34 .OR. ICHAR(sc(i))==39)THEN  !ichar(")=34, ichar(')=39
                  quotation=sc(i)
                  i=i-1
                  nonpair=.TRUE.
                  DO WHILE(i>0)
                     IF(sc(i)==quotation)THEN
                        nonpair=.FALSE.
                        EXIT
                     ENDIF
                     i=i-1
                  ENDDO
                  IF(nonpair)THEN
                     STOP 'quotation marks must be pair wise!'
!                     CALL term_execution(mesg)
                  ENDIF
               ENDIF
               IF(sc(i).EQ.BLANK .OR.                                         &
                    ICHAR(sc(i)).EQ.9  .OR.    &    !ichar(LF)=10
                    ICHAR(sc(i)).EQ.10 .OR.    &    !ichar(CR)=13
                    ICHAR(sc(i)).EQ.13) THEN        !ichar(CR)=13
                  nonblank=.FALSE.
               ELSE
                  IF(sc(i).EQ.AST) THEN
                     multidata=.TRUE.
                     multidcol=i
                  ENDIF
                  nonblank=.TRUE.
                  IF(sc(i).EQ.BANG) THEN
                     n=-1
                     nonblankd=.TRUE.
                  ENDIF
               ENDIF
               IF((.NOT.nonblankd.AND.nonblank) .OR. &
                    (nonblankd.AND..NOT.nonblank)) THEN
                  n=n+1
                  IF(multidata.AND.(nonblankd.AND. .NOT.nonblank)) THEN
                     READ(oneline(i+1:multidcol-1),*) nmult
                     n=n+(nmult-1)*2
                     multidata=.FALSE.
                  ENDIF
               ENDIF
               nonblankd=nonblank
               i=i-1
            ENDDO
            IF(MOD(n,2).NE.0) THEN
               nfields=n/2+1
            ELSE
               nfields=n/2
            ENDIF
            RETURN
      END FUNCTION nfields      

      FUNCTION ftfavg(x,nr,dr2)
!
            IMPLICIT NONE

! fuel volume avg. temperature

            REAL(sdk) :: ftfavg, x(*), dr2
            REAL(sdk) ::  area , xavg, z, a  !dmm
            INTEGER(sik) :: i, nrm1, l, nr
            xavg=zero
            i=1
            a=(x(i+1)-x(i))/dr2
            xavg=(7*x(i)+x(i+1))*0.25
            DO l=2,nr
               area=2/3.*l*x(l+1)+44/3.*i*x(l)+2/3.*(i-1)*x(l-1)
               xavg=xavg+area
               i=l
            ENDDO
            nrm1=nr-1
            area=(16/3.*nrm1+4+5/24.)*x(nr+1)+(10/3.*nrm1+2.25)*x(nr) &
                 -(2/3.*nrm1+11/24.)*x(nr-1)
            ftfavg=(xavg+area)*dr2/(8.*(nr*nr*dr2))
            RETURN
      END FUNCTION ftfavg      

      SUBROUTINE toupper(aa)
!            USE IntrType, ONLY: sdk, sik, ONLY: sdk, sik

            IMPLICIT NONE
            !
            ! convert lowercase string to uppercase

            INTEGER(sik), PARAMETER :: INDXA=97, IDNXZ=122
            INTEGER(sik) :: LENAA, i, ia
            CHARACTER :: aa*(*)

            !
            lenaa=LEN_TRIM(aa)
            i=1
            DO WHILE (aa(i:i).NE.' ' .AND. i.LE.lenaa)
               ia=ICHAR(aa(i:i))
               IF(ia.GE.INDXA) aa(i:i)=CHAR(ia-32)
               i=i+1
               IF(i.GT.lenaa) RETURN
            ENDDO
      END SUBROUTINE toupper      

      SUBROUTINE read1more(indev,onelinet,ndataf,cname)
            !
!            USE IntrType, ONLY: sdk, sik, ONLY: sdk, sik

            IMPLICIT NONE

            CHARACTER(len=10), INTENT(IN) :: cname
            CHARACTER(len=1024),INTENT(INOUT) ::  onelinet
            INTEGER(sik), INTENT(INOUT) :: ndataf, indev
            !
            INTEGER(sik) :: ncol, ios            
            LOGICAL :: ifreadmore            
            !
            ifreadmore=.TRUE.
            ndataf=0
            !
            DO WHILE (ifreadmore)
               READ(indev,'(a1024)',IOSTAT=ios) oneline
               IF (ios .NE. 0) THEN
                   mesg = 'Error reading '//cname//'Please check input'
                   WRITE(fid_out,*) mesg
                   STOP
               END IF
               IF(oneline.NE.BLANK .AND. oneline(1:1).NE.BANG) THEN
                  ndataf=nfields(oneline)
                  ifreadmore=.FALSE.
               ENDIF
            ENDDO
            onelinet=oneline
            RETURN
            !
    END SUBROUTINE read1more   
    
    SUBROUTINE term_execution(msg)

        USE PATHS_varcpld, ONLY: fid_out

        IMPLICIT NONE

        CHARACTER(len=80),  INTENT(IN) :: msg

        WRITE(fid_out,*) 'Error: ',msg
        STOP

    END SUBROUTINE term_execution

END MODULE PATHS_util
