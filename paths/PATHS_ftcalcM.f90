!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
MODULE PATHS_ftcalc
      ! Wall temperature and fuel temperature calculation module
      ! Saturated boiling region: q"=(hc+hnb)*(Tw-Tsat)
      ! Subcooled region        : q"=hnb*(Tw-Tsat)+hc(Tw-Tbulk) 
      ! Single-phase region     : q"=hdb*(Tw-Tbulk)
      ! 
      ! Fuel w/o heat source region: T(r)=qv/4ku*(Rout^2-r^2)+Tout
      ! Fuel w/  heat source region: T(r)=Ru^2*qv/2ku*log(Rout/r)+Tout
      !
      ! Linsen LI 05/19/2010

      USE IntrType, ONLY: sdk, sik
      USE PATHS_thvarM    
      USE PATHS_thmodels
      USE PATHS_WaterProp
      USE PATHS_heattc
      USE PATHS_FuelProbM, ONLY:fkf,fkc
      USE PATHS_varcpld, ONLY: plevnew !      USE PowerM

      IMPLICIT NONE

CONTAINS

      ! Wall temperature calculation
      ! In the single-phase and subcooled region, Tbulk as a function of enthalpy in the pressure of 7.5 Mpa.
      SUBROUTINE calc_tw

!            USE CntlM, ONLY: popt 
            IMPLICIT NONE
            INTEGER(sik) ::i,j,iter
            REAL(sdk) :: pres,tsat,errh,epsh,twtemp,qdp,h1,h2,x,xeq,tbulk,henth,hbulk,tw,maxtwc

            maxtwc=0
            DO i=1,PATHS_nchan
               DO j=1,PATHS_nz
             
                  IF (j.NE.PATHS_nz) THEN
                     pres=0.5*(PATHS_P(j,i)+PATHS_P(j+1,i)-PATHS_dens(j+1,i)*PATHS_u(j+1,i)**2/2*                  &
                                          ((PATHS_achan(j+1,PATHS_chantype(i))/PATHS_achan(j,PATHS_chantype(i)))**2-1))
                  ELSE
                     pres=0.5*(PATHS_P(j,i)+PATHS_P(j+1,i))
                  ENDIF
                  tsat=ts(pres)

                  qdp=PATHS_Prel(j,i)*PATHS_hz(j)*plevnew*PATHS_RTP*PATHS_symmmult*(1.0-PATHS_fracdc-PATHS_fracdwr-PATHS_fracdvb)/ &
                      (REAL(PATHS_nasm)*PATHS_nrod(j,PATHS_chantype(i))*2*3.1415926*PATHS_rrod(PATHS_chantype(i))*PATHS_hz(j)) 
                       ! Aaron - need to fix for possibility of 0 rods (div. by 0)

                  xeq=(hf(pres, j, i, .FALSE.)-0.5*(PATHS_enth(j,i)+PATHS_enth(j+1,i)))/(hf(pres, j, i, .FALSE.)-hg(pres))
!                            
                  x=0.5*(PATHS_qual(j,i)+PATHS_qual(j+1,i))
!
                  henth=0.5*(PATHS_enth(j,i)+PATHS_enth(j+1,i))
                  IF(x > 0.99) THEN
                     hbulk = hg(pres)
                  ELSE
                     hbulk=(henth-x*hg(pres))/(1-x)
                  ENDIF
                  
                  IF (x.LT.0.05)THEN   
                     tbulk=tfh(pres,hbulk, j, i, .FALSE.)
                  ELSE
                     tbulk=tsat
                  ENDIF

                  SELECT CASE (PATHS_mhtc)
                  CASE(1)
                     IF (x.EQ.0.0) THEN
                        tw=qdp/hdb(j,i)+tbulk
                     ELSEIF (tbulk.LT.tsat) THEN
                        IF (qdp.NE.0.0)THEN                       
                           epsh=1.0e-6
                           errh=1.0
                           h1=hc(j,i)
                           iter=0        
                           tw=PATHS_tw(j,i)
                           DO WHILE(errh.GT.epsh)
                              iter=iter+1
                              twtemp=tw
                              h2=hnb(j,i,twtemp)
                              tw=(qdp+h1*tbulk+h2*tsat)/(h1+h2)
                              errh=ABS((tw-twtemp)/twtemp)      
                           ENDDO
                        ELSE
                           tw=tbulk
                        ENDIF
                     ELSE
                        IF (qdp.NE.0.0)THEN 
                           epsh=1.0e-6
                           errh=1.0
                           h1=hc(j,i)
                           iter=0        
                           tw=PATHS_tw(j,i)
                           DO WHILE(errh.GT.epsh)
                              iter=iter+1
                              twtemp=tw
                              
                              h2=hnb(j,i,twtemp)
                              tw=qdp/(h1+h2)+tsat
                              
                              ! Aaron - limit tw to 1 degree below max. value allowed in steam tables.  
                              ! Will avoid crashing and allow it to iterate to convergence
                              ! However, should consider Newton iteration since fixed point
                              ! iteration may be very inefficient for this problem!
                              IF(tw>372.946) THEN
                                 tw=372.946
                              ENDIF
                              errh=ABS((tw-twtemp)/twtemp)      
                             
                           ENDDO
                        ELSE
                           tw=tsat
                        ENDIF
                     ENDIF
                  CASE(2)
                     IF (x.NE.0) THEN
                        tw=SQRT(qdp/(EXP(pres/4.35e6)/(22.7)**2*1000.0))+tsat
                     ELSE
                        tw=qdp/hdb(j,i)+tbulk         
                     ENDIF
                  END SELECT
                  twtemp=ABS(tw-PATHS_tw(j,i))
                  IF(maxtwc<twtemp)maxtwc=twtemp
                  PATHS_tw(j,i)=tw
                 
               ENDDO
            ENDDO

            WRITE(fid_out,'(A34,1x,1pE9.3)')'     Maximum Wall Temp. Change(K):', maxtwc

      END SUBROUTINE calc_tw

      ! Fuel temperature distribution calculation
      SUBROUTINE calc_tavg

            IMPLICIT NONE        
            INTEGER(sik):: i,j,k,ite
            INTEGER(sik):: ct
            REAL(sdk):: Touter, errft,epsft,qdot,PI,CKELVIN

            PI=3.1415926
            CKELVIN=273.15
            epsft=1.0e-6

            DO i=1,PATHS_nchan
               ct=PATHS_chantype(i)
               DO j=1,PATHS_nz
                  !Initiate conductivity and fuel temperature
                  !Assign the radius of each fuel node 
                  DO k=1,PATHS_nfm
                     keff(k)=fkf(PATHS_tavg(k,j,i)+CKELVIN)
                  ENDDO
                  DO k=PATHS_nfm+1,PATHS_ngm
                     keff(k)=PATHS_kgap(ct)
                  ENDDO
                  DO k=PATHS_ngm+1,PATHS_ncm
                     keff(k)=fkc(PATHS_tavg(k,j,i)+CKELVIN)
                  ENDDO

                  ! Heat flux assignment - Aaron: need to account for 0 rods (div. by 0)
                  qdot=1000*PATHS_Prel(j,i)*PATHS_hz(j)*plevnew*PATHS_RTP*PATHS_symmmult* &
                       (1.0-PATHS_fracdc-PATHS_fracdwr-PATHS_fracdvb) &
                       /(REAL(PATHS_nasm)*PATHS_nrod(j,ct)*PI*rfuel(PATHS_nfm)**2*PATHS_hz(j))

                  errft=1.0

                  ite=0

!                  IF (j.NE.1) THEN   ! implies one lower reflector??
                  IF (qdot .GT. 0.0) THEN                
                     DO WHILE (errft.GT.epsft)
                        ite=ite+1                    
                        Touter=PATHS_tw(j,i)
                        PATHS_tfuel(PATHS_ncm+1,j,i)=Touter

                        DO k=PATHS_ncm,PATHS_nfm+1,-1
                           tavgo(k)=PATHS_tavg(k,j,i)
                           PATHS_tavg(k,j,i)=Touter &
                                +qdot*PATHS_rf(ct)**2/(2*keff(k))*LOG(rfuel(k))&
                                +qdot*PATHS_rf(ct)**2/(4*keff(k))&
                                -qdot*PATHS_rf(ct)**2/(2*keff(k))*(rfuel(k)**2*LOG(rfuel(k))  &
                                -rfuel(k-1)**2*LOG(rfuel(k-1)))/(rfuel(k)**2-rfuel(k-1)**2)
                           Touter=Touter-qdot*PATHS_rf(ct)**2/(2*keff(k))*LOG(rfuel(k-1)/rfuel(k))
                           PATHS_tfuel(k,j,i)=Touter 
                        ENDDO

                        DO k=PATHS_nfm,2,-1
                           tavgo(k)=PATHS_tavg(k,j,i)
                           PATHS_tavg(k,j,i)=Touter+qdot*rfuel(k)**2/(4*keff(k))-qdot*(rfuel(k)**2+rfuel(k-1)**2)/(8*keff(k))
                           Touter=Touter+qdot*(rfuel(k)**2-rfuel(k-1)**2)/(4*keff(k))
                           PATHS_tfuel(k,j,i)=Touter 
                        ENDDO
                        tavgo(1)=PATHS_tavg(1,j,i)
                        PATHS_tavg(1,j,i)=Touter+qdot/(8*keff(1))*rfuel(1)**2
                        PATHS_tfuel(1,j,i)=Touter+qdot*rfuel(1)**2/(4*keff(1))


                        DO k=1,PATHS_nfm
                           keff(k)=fkf(PATHS_tavg(k,j,i)+CKELVIN)
                        ENDDO
                        DO k=PATHS_nfm+1,PATHS_ngm
                           keff(k)=PATHS_kgap(ct)
                        ENDDO
                        DO k=PATHS_ngm+1,PATHS_ncm
                           keff(k)=fkc(PATHS_tavg(k,j,i)+CKELVIN)
                        ENDDO

                        errft=0.0
                        DO k=1,PATHS_ncm
                           errft=MAX(errft,ABS((PATHS_tavg(k,j,i)-tavgo(k))/tavgo(k)))
                        ENDDO
                        
                        IF (PATHS_debug .GT. 0) THEN
                            !Andrew - Added to find where code is hanging (RBWR)
                            IF (ite .GT. 200) THEN
                                write(fid_tmp1,*)'Maximum number of iters in calc_tavg reached in channel: ',i
                                write(fid_tmp1,*)'Axial node: ',j
                                write(fid_tmp1,*)'Error reached: ', errft
                                write(fid_tmp1,*)'Error needed: ',epsft
                                !read(*,*)
                                EXIT
                            END IF                        
                        END IF
                        
                     ENDDO
                     
                     ! Calculate the volume average and store
                     PATHS_tavg(PATHS_ncm+1,j,i)=PATHS_tavg(1,j,i)*PI*rfuel(1)*rfuel(1)
                     DO k=2,PATHS_nfm
                        PATHS_tavg(PATHS_ncm+1,j,i)=PATHS_tavg(PATHS_ncm+1,j,i) + &
                                   PATHS_tavg(k,j,i)*PI*(rfuel(k)**2-rfuel(k-1)**2)
                     ENDDO
                     PATHS_tavg(PATHS_ncm+1,j,i)=PATHS_tavg(PATHS_ncm+1,j,i)  &
                               / (PI*rfuel(PATHS_nfm)**2)
                  ELSE
                     PATHS_tavg(:,j,i) = PATHS_tw(j,i)
                     PATHS_tfuel(:,j,i) = PATHS_tw(j,i)
                  ENDIF                     

               ENDDO
            ENDDO


      END SUBROUTINE calc_tavg

      ! initialize fuel temperature
      SUBROUTINE init_tf

            IMPLICIT NONE        
            INTEGER(sik):: i,j,k,ct
            REAL(sdk) :: pres,tsat

            DO i=1,PATHS_nchan
               ct=PATHS_chantype(i)
               DO j=1,PATHS_nz
                  IF (j.NE.PATHS_nz) THEN
                     pres=0.5*(PATHS_P(j,i)+PATHS_P(j+1,i)-PATHS_dens(j+1,i)*PATHS_u(j+1,i)**2/2*                  &
                                          ((PATHS_achan(j+1,PATHS_chantype(i))/PATHS_achan(j,PATHS_chantype(i)))**2-1))
                  ELSE
                     pres=0.5*(PATHS_P(j,i)+PATHS_P(j+1,i))
                  ENDIF
                  tsat=ts(pres)
                  ! initiate the wall temperature
                  PATHS_tw(j,i)=1.2*tsat
                  !Assign the radius of each fuel node 
                  DO k=1,PATHS_nfm
                     PATHS_tavg(k,j,i)=PATHS_tw(j,i)
                     PATHS_tfuel(k,j,i)=PATHS_tw(j,i)
                     rfuel(k)=REAL(k)*PATHS_rf(ct)/REAL(PATHS_nfm)
                  ENDDO
                  DO k=PATHS_nfm+1,PATHS_ngm
                     PATHS_tavg(k,j,i)=PATHS_tw(j,i)
                     PATHS_tfuel(k,j,i)=PATHS_tw(j,i)
                     rfuel(k)=PATHS_rf(ct)+REAL(k-PATHS_nfm)*(PATHS_rrod(ct)-PATHS_rf(ct)-PATHS_tc(ct))/REAL(PATHS_ngm-PATHS_nfm)
                  ENDDO
                  DO k=PATHS_ngm+1,PATHS_ncm
                     PATHS_tavg(k,j,i)=PATHS_tw(j,i)
                     PATHS_tfuel(k,j,i)=PATHS_tw(j,i)
                     rfuel(k)=PATHS_rrod(ct)-PATHS_tc(ct)+REAL(k-PATHS_ngm)*PATHS_tc(ct)/REAL(PATHS_ncm-PATHS_ngm)
                  ENDDO
                  PATHS_tfuel(PATHS_ncm+1,j,i)=PATHS_tw(j,i)
               ENDDO
            ENDDO
      END SUBROUTINE init_tf


END MODULE PATHS_ftcalc

