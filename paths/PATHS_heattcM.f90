!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
MODULE PATHS_heattc
      ! Heat transfer coefficient calculation module
      ! Saturated boiling region: q"=(hc+hnb)*(Tw-Tsat)
      ! Subcooled region        : q"=hnb*(Tw-Tsat)+hc(Tw-Tbulk) 
      ! Single-phase region     : q"=hdb*(Tw-Tbulk)
      !
      ! Linsen LI 05/14/2010

      USE IntrType, ONLY: sdk, sik
      USE PATHS_thvarM    
      USE PATHS_thmodels
      USE PATHS_WaterProp

      IMPLICIT NONE

CONTAINS

      ! Convection heat transfer coefficient, a modified Dittus-Boelter correlation in kw/m2k
      FUNCTION hc(z,chan)

            IMPLICIT NONE
            INTEGER(sik) ::z, chan
            REAL(sdk) :: hd, hc
            REAL(sdk) :: F

            hd=hdb(z,chan)        
            F=Fft(z,chan)                
            hc=hd*F

            RETURN  

      END FUNCTION hc

      ! Nucleation boiling heat transfer coefficient in kw/m2k, based on the Forster-Zuber equation.  
      FUNCTION hnb(z,chan,tw)

            IMPLICIT NONE
            INTEGER(sik) ::z, chan
            REAL(sdk) :: hnb,pres,k,cp,mu,tsat,sigma,hfg,pw,delta_t,delta_p,deno,nume,tw
            REAL(sdk) :: S
            
            IF (z.NE.PATHS_nz) THEN
               pres=0.5*(PATHS_P(z,chan)+PATHS_P(z+1,chan)-PATHS_dens(z+1,chan)*PATHS_u(z+1,chan)**2/2*                  &
                                    ((PATHS_achan(z+1,PATHS_chantype(chan))/PATHS_achan(z,PATHS_chantype(chan)))**2-1))
            ELSE
               pres=0.5*(PATHS_P(z,chan)+PATHS_P(z+1,chan))
            ENDIF
            k=kf(pres, z, chan, .FALSE.)
            cp=cpf(pres, z, chan, .FALSE.)
            mu=muf(pres, z, chan, .FALSE.)
            tsat=ts(pres)                          
            sigma=surften(pres)
            hfg=hg(pres)-hf(pres, z, chan, .FALSE.)
            pw=ps(tw)
!
            delta_p=MAX(0.0_sdk,(pw-pres)) 
            delta_t=MAX(0.0_sdk,(tw-tsat)) 

            nume=0.00122*(k**0.79*cp**0.45*rhof(pres, z, chan, .FALSE.)**0.49)*delta_t**0.24*delta_p**0.75        
            deno=sigma**0.5*mu**0.29*hfg**0.24*rhog(pres)**0.24

            S=Sft(z,chan)

            hnb=S*nume/deno

            RETURN

      END FUNCTION hnb

      !Dittus-Boelter correlation for heat transfer coefficient in the single-phase flow region        
      FUNCTION hdb(z,chan)

            IMPLICIT NONE
            INTEGER(sik) ::z,chan
            INTEGER(sik) ::ct
            REAL(sdk) :: PI
            REAL(sdk) :: Pit,Dia
            REAL(sdk) :: hdb,pres,k, mu, cp, pr, re, x, G, Cdb 
            PI=3.141592
            ct=PATHS_chantype(chan)
            Pit=PATHS_pit(ct)
            Dia=2*PATHS_rf(ct)
            x=0.5*(quality(z,chan)+quality(z+1,chan))
            !G=0.5*(PATHS_dens(z,chan)+PATHS_dens(z+1,chan))*0.5*(PATHS_u(z,chan)+PATHS_u(z+1,chan))
            G=PATHS_dens(1,chan)*PATHS_u(1,chan) !More accurate, based on continuity
            
            IF (z.NE.PATHS_nz) THEN
               pres=0.5*(PATHS_P(z,chan)+PATHS_P(z+1,chan)-PATHS_dens(z+1,chan)*PATHS_u(z+1,chan)**2/2*                  &
                                    ((PATHS_achan(z+1,PATHS_chantype(chan))/PATHS_achan(z,PATHS_chantype(chan)))**2-1))
            ELSE
               pres=0.5*(PATHS_P(z,chan)+PATHS_P(z+1,chan))
            ENDIF
            k=kf(pres, z, chan, .FALSE.)
            mu=muf(pres, z, chan, .FALSE.)
            cp=cpf(pres, z, chan, .FALSE.)
            pr=cp*mu/k
            re=G*(1-x)*PATHS_dhz(z,PATHS_chantype(chan))/mu

            SELECT CASE(PATHS_mdb)
            CASE(1) ! Ben's
            ! Aaron - NEED TO INCLUDE PART-LENGTH RODS FOR AREA CALCS
               Cdb=0.033*PATHS_achan(z,PATHS_chantype(chan))/(PATHS_achan(z,PATHS_chantype(chan))         &
                    +PATHS_nrod(z,PATHS_chantype(chan))*PI*PATHS_rrod(PATHS_chantype(chan))**2           &
                    +PATHS_nwrod(PATHS_chantype(chan))*PI*PATHS_rwrod(PATHS_chantype(chan))**2)+0.013
               hdb=Cdb*k*(pr**0.4)*(re**0.8)/PATHS_dhz(z,PATHS_chantype(chan))
            CASE(2) ! Yunlin's
               hdb=Pit/Dia*0.023*k*(pr**0.4)*(re**0.8)/PATHS_dhz(z,PATHS_chantype(chan))
            END SELECT

            RETURN

      END FUNCTION hdb


      FUNCTION Fft(z,chan)

            IMPLICIT NONE
            INTEGER(sik) ::z,chan
            REAL(sdk) :: Fft, G, pres, x, henth, hbulk, tsat, tb, td, Xtt_1

            !G=0.5*(PATHS_dens(z,chan)+PATHS_dens(z+1,chan))*0.5*(PATHS_u(z,chan)+PATHS_u(z+1,chan))
            G=PATHS_dens(1,chan)*PATHS_u(1,chan) !More accurate, based on continuity
            IF (z.NE.PATHS_nz) THEN
               pres=0.5*(PATHS_P(z,chan)+PATHS_P(z+1,chan)-PATHS_dens(z+1,chan)*PATHS_u(z+1,chan)**2/2*                  &
                                    ((PATHS_achan(z+1,PATHS_chantype(chan))/PATHS_achan(z,PATHS_chantype(chan)))**2-1))
            ELSE
               pres=0.5*(PATHS_P(z,chan)+PATHS_P(z+1,chan))
            ENDIF
            x=0.5*(quality(z,chan)+quality(z+1,chan))

            Xtt_1=(x/(1-x))**0.9*(rhof(pres, z, chan, .FALSE.)/rhog(pres))**0.5*(mug(pres)/muf(pres, z, chan, .FALSE.))**0.1

            SELECT CASE(PATHS_mfs)
            CASE(1)
               IF (Xtt_1 .GE. 0.100207) THEN
                  Fft=2.35*(0.213+Xtt_1)**0.736
               ELSE
                  Fft=1.0
               ENDIF
            CASE(2)
               IF (Xtt_1 .GE. 0.100207) THEN
                  Fft=2.35*(0.213+Xtt_1)**0.736
               ELSE
                  Fft=1.0
               ENDIF
               henth=0.5*(PATHS_enth(z,chan)+PATHS_enth(z+1,chan))
               hbulk=(henth-x*hg(pres))/(1-x)
               tsat=ts(pres)
               tb=tfh(pres,hbulk, z, chan, .FALSE.)
               td=tsat-tb
               IF(td.GT.0.0)THEN
                  Fft=1.0
               ENDIF
            CASE(3)
               IF (Xtt_1 .GE. 0.100207) THEN
                  Fft=2.35*(0.213+Xtt_1)**0.736
                  ! Added by Yunlin XU 11/04/2010
                  henth=0.5*(PATHS_enth(z,chan)+PATHS_enth(z+1,chan))
                  hbulk=(henth-x*hg(pres))/(1-x)
                  tsat=ts(pres)
                  tb=tfh(pres,hbulk, z, chan, .FALSE.)
                  td=tsat-tb
                  IF(td.GT.0.0)THEN
                     IF(td.GT.5.0)THEN   
                        Fft=1.0
                     ELSE
                        Fft=Fft-0.2*td*(Fft-1)
                     ENDIF
                  ENDIF
                  !
               ELSE
                  Fft=1.0
               ENDIF
            END SELECT

            RETURN  

      END FUNCTION Fft


      FUNCTION Sft(z,chan)

            IMPLICIT NONE
            INTEGER(sik) ::z, chan
            REAL(sdk) :: Sft, G, pres, x, ref, re, henth, hbulk, td, tsat, tb, F, retp

            F=Fft(z,chan)

            !G=0.5*(PATHS_dens(z,chan)+PATHS_dens(z+1,chan))*0.5*(PATHS_u(z,chan)+PATHS_u(z+1,chan))
            G=PATHS_dens(1,chan)*PATHS_u(1,chan) !More accurate, based on continuity
            IF (z.NE.PATHS_nz) THEN
               pres=0.5*(PATHS_P(z,chan)+PATHS_P(z+1,chan)-PATHS_dens(z+1,chan)*PATHS_u(z+1,chan)**2/2*                  &
                                    ((PATHS_achan(z+1,PATHS_chantype(chan))/PATHS_achan(z,PATHS_chantype(chan)))**2-1))
            ELSE
               pres=0.5*(PATHS_P(z,chan)+PATHS_P(z+1,chan))
            ENDIF
            x=0.5*(quality(z,chan)+quality(z+1,chan))
            ref=G*(1-x)*PATHS_dhz(z,PATHS_chantype(chan))/muf(pres, z, chan, .FALSE.)
            re=ref*F**1.25

            SELECT CASE(PATHS_mfs)
            CASE(1)
               Sft=1.0/(1+2.53e-6*re**1.17)
            CASE(2)
               henth=0.5*(PATHS_enth(z,chan)+PATHS_enth(z+1,chan))
               hbulk=(henth-x*hg(pres))/(1-x)
               tsat=ts(pres)
               tb=tfh(pres,hbulk, z, chan, .FALSE.)
               td=tsat-tb
               IF(td.GT.0.0)THEN
                  ref=G*PATHS_dhz(z,PATHS_chantype(chan))/muf(pres, z, chan, .FALSE.)
                  re=ref*F**1.25
               ENDIF
               Sft=1.0/(1+2.53e-6*re**1.17)
            CASE(3)
               retp=MIN(70.0_sdk, 1.0e-4*ref*F**1.25)
               IF(retp .GE. 70.0) THEN
                  Sft=0.0797
               ELSEIF (retp .GE. 32.5) THEN
                  Sft=(1.0+0.42*retp**0.78)**(-1.0)
               ELSE
                  Sft=(1.0+0.12*retp)**(-1.14)
               END IF
            END SELECT

            RETURN  

      END FUNCTION Sft


END MODULE PATHS_heattc
