!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
MODULE PATHS_M

      USE PATHS_THCal
      USE PATHS_thvarM
      USE PATHS_blas1
      USE PATHS_blas2
      USE PATHS_bicgstab
      USE PATHS_linsolve
      USE PATHS_heattc
      USE PATHS_ftcalc
      USE PATHS_WaterProp
      USE PATHS_thmodels
      USE PATHS_util
      USE IntrType, ONLY: sdk, sik  !, ckelvin
!      USE PATHS_varcpld, ONLY: plevnew, THconv, tlap, tth !      USE PowerM
      USE PATHS_varcpld, ONLY: plevnew, tlap, tth !      USE PowerM

!      USE ThgeomM, ONLY: nzth ! nchan
      
      IMPLICIT NONE

CONTAINS

      SUBROUTINE PATHS

        IMPLICIT NONE

        INTEGER(sik) :: i, j, k, ii
        REAL(sdk)    :: nonlin
        REAL(sdk)    :: tstart, tend
        REAL(sdk)    :: x,pres,tsat,tbulk,tdoplold,fnchan,dum,tout,henth, hbulk
        REAL(sdk)    :: dcmin,dcmax, tcmin,tcmax, hcmin, hcmax

        tlap=dclock()

            flagpaths=.false.

            dcmin=0
            dcmax=0
            tcmin=0
            tcmax=0
            hcmin=0
            hcmax=0

            IF(.NOT.freezedc) THEN
                
               CALL PATHS_UpdatePower
               CALL PATHS_Update_hin
               CALL PATHS_ssthiter
               i=1
               nonlin=thnonlintol()

               DO WHILE ((i .LT. PATHS_noutmax).AND.(PATHS_err .GT. PATHS_epsout))
                  i=i+1
                  IF (i .LT. 5) CALL PATHS_Update_hin  !fix hin after 5 iterations to avoid slowing convergence
                  CALL PATHS_ssthiter

                  IF(MOD(i,10) .EQ.0 .AND. PATHS_debug .GT. 1) THEN
                     WRITE(fid_tmp1,*) 'Inners: ',i,PATHS_err,PATHS_epsout
                  END IF
               ENDDO
               
               IF(PATHS_debug .GT. 0) THEN
                  WRITE(fid_tmp1,*) 'Number of outer iters: ',i
                  WRITE(fid_tmp1,*)
               ENDIF
               
               IF(PATHS_err .LT. PATHS_epsout) THEN
                  flagpaths=.true.
               ELSE
                  flagpaths=.false.
               ENDIF

               CALL calc_tw    ! Wall temperature
               CALL calc_tavg  ! Average fuel temperature
               
               CALL calc_tcool
              
            ENDIF

            ! Only with verbosity high
            IF (PATHS_verbos .GE. 2) CALL PATHS_PRINT(0,0)                       
               
            tend=dclock()-tlap
            WRITE(fid_out,'(A27,1x,1pE10.4)')'     Total Time Elapsed(s):', tend

            first_solve = .FALSE.                        
            
            RETURN

      END SUBROUTINE PATHS

      SUBROUTINE PATHS_Update_hin
         IMPLICIT NONE 
         CHARACTER(len=80) ::  mesg
         
         ! Convert PATHS_Tin to enthalpy if given
         IF (PATHS_Tin > 0) THEN
            IF (stable) THEN
               PATHS_hin= t_to_enth(PATHS_P(1,1), PATHS_Tin)*0.001
            ELSE
               mesg = 'Error: PATHS_Tin can only be input if S_TABLE==T'
               CALL term_execution(mesg) 
            ENDIF
         ENDIF
      END SUBROUTINE PATHS_Update_hin      
      
      SUBROUTINE PATHS_PRINT(icyc,idep)
      
        IMPLICIT NONE  
        
        INTEGER(sik), INTENT(IN) :: icyc,idep
        
        INTEGER(sik) :: i, j
        REAL(sdk)    :: tflarea,tflrate,chflrate,chflarea,chflden
        REAL(sdk)    :: avevoid,aveden,avevel,avepre,aveent,avetco,avepow
        REAL(sdk)    :: zedge(PATHS_nax), norm_fact
        REAL(sdk)    :: t_rho, t_cool, t_tave, t_tcen, t_tsur        
        CHARACTER(len=79) :: form1,form2
        
        form1 = '(i6,3x,  1F9.3)'
        WRITE(form1(8:10),'(I3)') nzpl

        form2 = '(i6,3x,  1F9.4)'
        WRITE(form2(8:10),'(I3)') PATHS_nz

        zedge(1)=0.0_sdk
        DO j=1,PATHS_nz
           zedge(j+1)=zedge(j)+PATHS_hz(j)
        ENDDO

        WRITE(fid_pth,'(A,I3,A,I4)') ' Cycle: ',icyc,' Point: ',idep
        WRITE(fid_pth,'(A)') ''
        
        IF (PATHS_verbos .GT. 0) THEN
        
           WRITE(fid_xth,'(A,I3,A,I4)') ' Cycle: ',icyc,' Point: ',idep
           WRITE(fid_xth,'(A)') ''
        
           DO i=1,PATHS_nchan
               WRITE(fid_xth,*) ''
               WRITE(fid_xth,*) 'Channel ', i
               WRITE(fid_xth,"(A28)",advance="no")'    Den Cool      TempCool'
               WRITE(fid_xth,"(A42)")'      Ave Fuel        Center       Surface'
               DO j=PATHS_nax-1,1,-1               
                   t_rho  = 0.5 * (PATHS_dens(j,i) + PATHS_dens(j+1,i))*0.001                    
                   t_cool = 0.5 * (PATHS_tcool(j,i) + PATHS_tcool(j+1,i))+ckelvin
                   t_tave = PATHS_tavg(PATHS_ncm+1,j,i)+ckelvin
                   t_tcen = PATHS_tfuel(1,j,i)+ckelvin
                   t_tsur = PATHS_tfuel(PATHS_nfm+1,j,i)+ckelvin
                   WRITE(fid_xth,'(1p5E14.7)')  t_rho, t_cool, t_tave, t_tcen, t_tsur
               ENDDO
           ENDDO            
           IF (PATHS_mvd.NE.7) THEN
              DO i=1,PATHS_nchan
                 WRITE(fid_pth,*) ''
                 WRITE(fid_pth,*) 'Channel ', i
                 WRITE(fid_pth,"(A70)",advance="no")'     Height       RelPower      Density       Enthalpy    Cool. Temp  '
                 WRITE(fid_pth,"(A54)")'   Pressure     Void Fract.    Velocity      Flow Rate'  
                 WRITE(fid_pth,'(1p9E14.7)') zedge(PATHS_nax),0.0,PATHS_dens(PATHS_nax,i),PATHS_enth(PATHS_nax,i), &
                       PATHS_tcool(PATHS_nax,i)+ckelvin, PATHS_P(PATHS_nax,i), PATHS_vf(PATHS_nax,i),PATHS_u(PATHS_nax,i), &
                       PATHS_dens(PATHS_nax,i)*PATHS_u(PATHS_nax,i)*PATHS_achan(PATHS_nax-1,PATHS_chantype(i))*PATHS_nasmchan(i)                       
                 DO j=PATHS_nax-1,1,-1                 
                    WRITE(fid_pth,'(1p9E14.7)') zedge(j),PATHS_prel(j,i),PATHS_dens(j,i),PATHS_enth(j,i), &
                          PATHS_tcool(j,i)+ckelvin, PATHS_P(j,i), PATHS_vf(j,i),PATHS_u(j,i), &
                          PATHS_dens(j,i)*PATHS_u(j,i)*PATHS_achan(j,PATHS_chantype(i))*PATHS_nasmchan(i)                                              
                 ENDDO
              ENDDO
           ELSE
              DO i=1,PATHS_nchan
                 WRITE(fid_pth,*) ''
                 WRITE(fid_pth,*) 'Channel ', i
                 WRITE(fid_pth,"(A70)",advance="no")'     Height       RelPower      Density       Enthalpy    Cool. Temp  '
                 WRITE(fid_pth,"(A66)")'   Pressure     Void Fract.    Velocity      Flow Rate      FlwReg'  
                 WRITE(fid_pth,'(1p10E14.7)') zedge(PATHS_nax),0.0,PATHS_dens(PATHS_nax,i),PATHS_enth(PATHS_nax,i), &
                       PATHS_tcool(PATHS_nax,i)+ckelvin, PATHS_P(PATHS_nax,i), PATHS_vf(PATHS_nax,i),PATHS_u(PATHS_nax,i), &
                       PATHS_dens(PATHS_nax,i)*PATHS_u(PATHS_nax,i)*PATHS_achan(PATHS_nax-1,PATHS_chantype(i))*PATHS_nasmchan(i), &                       
                       PATHS_regnm(PATHS_nax,i)
                 DO j=PATHS_nax-1,1,-1                 
                    WRITE(fid_pth,'(1p10E14.7)') zedge(j),PATHS_prel(j,i),PATHS_dens(j,i),PATHS_enth(j,i), &
                          PATHS_tcool(j,i)+ckelvin, PATHS_P(j,i), PATHS_vf(j,i),PATHS_u(j,i), &
                          PATHS_dens(j,i)*PATHS_u(j,i)*PATHS_achan(j,PATHS_chantype(i))*PATHS_nasmchan(i), &
                          PATHS_regnm(j,i)
                 ENDDO
              ENDDO
           ENDIF
           WRITE(fid_pth,*) ''

           WRITE(fid_tmp3,*) 'Chan num   Nodewise centerline fuel temperature values'
!           WRITE(form,'("(i6,3x,",i2,"F9.3)")') nzpl
           DO i=1,PATHS_nchan
              WRITE(fid_tmp3,form1) i,(PATHS_tfuel(1,j,i),j=1,nzpl)
           ENDDO
           WRITE(fid_tmp3,*)

           WRITE(fid_tmp5,*) 'Chan num   Nodewise relative power values'
!           WRITE(form,'("(i6,3x,",i2,"F9.4)")') PATHS_nz
           norm_fact = 0.0
           DO i=1,PATHS_nchan
              DO j=nzlr+1,nzpl-nzur
                 norm_fact = norm_fact + PATHS_Prel(j,i)*PATHS_hz(j)
              ENDDO
           ENDDO
           norm_fact = PATHS_nchan*sum(PATHS_hz(nzlr+1:nzpl-nzur))/norm_fact
           DO i=1,PATHS_nchan
              WRITE(fid_tmp5,form2) i,(PATHS_Prel(j,i)*norm_fact,j=1,PATHS_nz)
           ENDDO
           WRITE(fid_tmp5,*)

           WRITE(fid_tmp6,*) 'Chan num   Flow Rate (kg/s)  Quality'
           DO i=1,PATHS_nchan
              WRITE(fid_tmp6,'(i6,1X,F11.4,1x,F9.3)') i,PATHS_dens(1,i)*PATHS_u(1,i)*PATHS_achan(1,PATHS_chantype(i))  &
                                                  *PATHS_nasmchan(i),PATHS_qual(PATHS_nax,i)
           ENDDO
           WRITE(fid_tmp6,*)
        ENDIF

        IF(PATHS_cpropt .GT. 0) THEN
           CALL calc_cpr

           WRITE(fid_tmp7,*) 'Chan num  x_exit   x_crit   exit CPR'
           DO i=1,PATHS_nchan
              WRITE(fid_tmp7,'(i6,1X,3F9.3)') i,PATHS_qual(PATHS_nax,i),PATHS_xcrit(PATHS_nax,i),PATHS_cpr(PATHS_nax,i)
           ENDDO

           WRITE(fid_tmp8,*) 'Chan num   Nodewise CPR values'
!           WRITE(form,'("(i6,3x,",i2,"F9.3)")') PATHS_nax
           DO i=1,PATHS_nchan
              WRITE(fid_tmp8,form1) i,(PATHS_cpr(j,i),j=1,PATHS_nax)
           ENDDO

           WRITE(fid_tmp8,*)

        ENDIF       
        
        IF(PATHS_nchan==1)RETURN
        WRITE(fid_pth,*) 'All Channels '

        WRITE(fid_pth,"(A70)",advance="no")'     Height       RelPower      Density       Enthalpy    Cool. Temp  '
        WRITE(fid_pth,"(A54)")'   Pressure     Void Fract.    Velocity      Flow Rate'         
        
        ! Aaron - modified for axially variable flow area
        DO j=PATHS_nax,1,-1
           tflarea=0
           DO i=1,PATHS_nchan
              IF(j.LT.PATHS_nax) THEN
                 tflarea=tflarea+PATHS_achan(j,PATHS_chantype(i))*PATHS_nasmchan(i)
              ELSE
                 tflarea=tflarea+PATHS_achan(j-1,PATHS_chantype(i))*PATHS_nasmchan(i)
              ENDIF
           ENDDO
           tflrate=0
           avevoid=0
           aveden=0
           avepre=0
           aveent=0
           avetco=0
           DO i=1,PATHS_nchan
              IF(j.LT.PATHS_nax) THEN
                 chflarea=PATHS_achan(j,PATHS_chantype(i))*PATHS_nasmchan(i)
              ELSE
                 chflarea=PATHS_achan(j-1,PATHS_chantype(i))*PATHS_nasmchan(i)
              ENDIF
              chflden=chflarea*PATHS_dens(j,i)
              chflrate=chflden*PATHS_u(j,i)
              tflrate=tflrate+chflrate
              avepre=avepre+chflarea*PATHS_P(j,i)
              avevoid=avevoid+chflarea*PATHS_vf(j,i)
              aveden=aveden+chflden
              aveent=aveent+chflrate*PATHS_enth(j,i)
              avetco=avetco+chflarea*PATHS_tcool(j,i)
           ENDDO
           avepow=0           
           IF (j .LT. PATHS_nax) THEN
              DO i=1,PATHS_nchan
                 avepow=avepow+PATHS_prel(j,i)
              ENDDO              
           END IF
           avevel=tflrate/aveden
           aveent=aveent/tflrate
           aveden=aveden/tflarea
           avevoid=avevoid/tflarea
           avepre=avepre/tflarea
           avetco=avetco/tflarea
           WRITE(fid_pth,'(1p9E14.7)') zedge(j), avepow, aveden, aveent, avetco, avepre, avevoid, avevel, tflrate
        ENDDO
        WRITE(fid_pth,*) ""
      END SUBROUTINE PATHS_PRINT
      
      SUBROUTINE calc_cpr
         INTEGER(sik) :: i, j
         REAL(sdk) :: A,B,PHPW,Pc,P,Dh,Dht,G,A0,Fp,F,xin
         REAL(sdk)  :: zedge(PATHS_nax), pi
         REAL(sdk) :: Fpmin
         
         pi=3.1415926
      
         zedge(1)=0.0_sdk
         DO j=1,PATHS_nz
            zedge(j+1)=zedge(j)+PATHS_hz(j)
         ENDDO

         ! Find equilibrium quality based on enthalpy
         DO i=1,PATHS_nchan
            DO j=1,PATHS_nax
               PATHS_xeq(j,i)=(hf(PATHS_P(j,i), j, i, .TRUE.)-PATHS_enth(j,i))/(hf(PATHS_P(j,i), j, i, .TRUE.)-hg(PATHS_P(j,i)))
            ENDDO
         ENDDO
                  
         ! Find where xeq first becomes positive, and interpolate two nearest nodes to find best estimate of boiling boundary
         DO i=1,PATHS_nchan
            DO j=1,PATHS_nax
               IF(PATHS_xeq(j,i).GT.0.0) EXIT
            ENDDO
            PATHS_boil_len(i) = ((0-PATHS_xeq(j-1,i))*zedge(j)+(PATHS_xeq(j,i)-0)*zedge(j-1))/(PATHS_xeq(j,i)-PATHS_xeq(j-1,i))
         ENDDO
         
         Pc=22.0878e6  ! Pascals
         FPmin = 0.0
         
         SELECT CASE (PATHS_cpropt)
         CASE(0) ! No CPR calculation
            
         CASE(1) ! Hitachi recommended CISE modified correlation
            DO i=1,PATHS_nchan
               G=PATHS_dens(1,i)*PATHS_u(1,i)
            
               !DO j=1,PATHS_nax
               !   IF(PATHS_qual(j,i).GT.0.0) EXIT
               !ENDDO
               !LB=zedge(j)
            
               xin=PATHS_xeq(1,i)
            
               DO j=1,PATHS_nax
                  P=PATHS_P(j,i)
                  IF(j<PATHS_nax) THEN
                     Dh=PATHS_dhz(j,PATHS_chantype(i))
                     PHPW=PATHS_nrod(j,PATHS_chantype(i))*2*pi*PATHS_rrod(PATHS_chantype(i))/ & 
                     (4*PATHS_achan(j,PATHS_chantype(i))/Dh)
                  ELSE
                     Dh=PATHS_dhz(j-1,PATHS_chantype(i))
                     PHPW=PATHS_nrod(j-1,PATHS_chantype(i))*2*pi*PATHS_rrod(PATHS_chantype(i))/ & 
                     (4*PATHS_achan(j-1,PATHS_chantype(i))/Dh)
                  ENDIF
               
      
                  Fp=max(1.5-P/1.379e7,Fpmin)
                  F=1.0+0.1*(PATHS_boil_len(i)-Fp)
               
                  IF(G<406.8) THEN
                     A0=(1-P/Pc)/(406.8/1200.0)**0.4*F
                     A=(1.0+G/406.8*A0)/(1.0+G/406.8)
                     !A=1-(A-A0)*G/406.8   ! Check for typo in A term!!
                  ELSEIF(G>=406.8 .AND. G<1085.0) THEN
                     A=(1-P/Pc)/(G/1200.0)**0.4*F
                  ELSEIF(G>=1085.0) THEN
                     A=(1-P/Pc)/(G/1176.0)**0.5*F
                  ENDIF
               
                  B=0.19875*(Pc/P-1.0)**0.4*Dh**1.4*G
                  PATHS_xcrit(j,i)=A*PATHS_boil_len(i)/(PATHS_boil_len(i)+B)*PHPW
                  PATHS_cpr(j,i)=(PATHS_xcrit(j,i)-xin)/(PATHS_qual(j,i)-xin)
               ENDDO
            ENDDO
         CASE(2) ! MIT recommended CISE modified correlation
            DO i=1,PATHS_nchan
               G=PATHS_dens(1,i)*PATHS_u(1,i)
            
               !DO j=1,PATHS_nax
               !   IF(PATHS_qual(j,i).GT.0.0) EXIT
               !ENDDO
               !LB=zedge(j)
            
               xin=PATHS_xeq(1,i)
            
               DO j=1,PATHS_nax
                  P=PATHS_P(j,i)
                  IF(j<PATHS_nax) THEN
                     Dh=PATHS_dhz(j,PATHS_chantype(i))
                     Dht=4*PATHS_achan(j,PATHS_chantype(i))/(PATHS_nrod(j,PATHS_chantype(i))*2*pi*PATHS_rrod(PATHS_chantype(i)))
                  ELSE
                     Dh=PATHS_dhz(j-1,PATHS_chantype(i))
                     Dht=4*PATHS_achan(j-1,PATHS_chantype(i))/(PATHS_nrod(j-1,PATHS_chantype(i))*2*pi*PATHS_rrod(PATHS_chantype(i)))
                  ENDIF
      
                  IF(G<(3375.0*(1-P/Pc)**3)) THEN
                     A=(1.0+1.481e-4/(P/Pc)**3*G*0.4)**(-1)
                  ELSE
                     A=(1-P/Pc)/((G*0.4/1000.0)**0.33)
                  ENDIF
               
                  B=0.199*(Pc/P-1.0)**0.4*Dh**1.1*G
                  PATHS_xcrit(j,i)=Dh/Dht*(A*PATHS_boil_len(i)/(PATHS_boil_len(i)+B))
                  PATHS_cpr(j,i)=(PATHS_xcrit(j,i)-xin)/(PATHS_qual(j,i)-xin)
               ENDDO
            ENDDO
         END SELECT
      END SUBROUTINE calc_cpr      

      SUBROUTINE calc_tcool

        IMPLICIT NONE  
        INTEGER(sik) :: i, j
       
        REAL(sdk)    :: t_rho, t_qual, t_pres, t_enth
        REAL(sdk)    :: hbulk, tbulk, tcool, tsat

        ! Calculate the coolant temperature, just to have it            
        DO i=1,PATHS_nchan
            DO j=1,PATHS_nax
                
                t_rho  = PATHS_dens(j,i)
                t_qual = PATHS_qual(j,i)
                t_pres = PATHS_P(j,i)
                t_enth = PATHS_enth(j,i)
                
                tsat=ts(t_pres)                      ! saturated coolant temp based upon pressure
                hbulk=(t_enth-t_qual*hg(t_pres))/(1-t_qual)     ! bulk enthalpy
                
                ! Bulk coolant temperature
                IF (t_qual.LT.0.05)THEN
                    tbulk=tfh(t_pres,hbulk, j, i, .FALSE.)
                ELSE
                    tbulk=tsat
                ENDIF
                ! Coolant temperature
                IF (t_qual.EQ.0.0 .OR. tbulk .LT. tsat) THEN
                    PATHS_tcool(j,i) = tbulk
                ELSE
                    PATHS_tcool(j,i) = tsat
                ENDIF                  
            ENDDO
        ENDDO             
            
      END SUBROUTINE calc_tcool
      
END MODULE PATHS_M
