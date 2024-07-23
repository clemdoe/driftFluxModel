!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
MODULE PATHS_thmodels

      USE IntrType, ONLY: sdk, sik
      USE PATHS_thvarM
      USE PATHS_varcpld  !      USE PowerM

      IMPLICIT NONE

CONTAINS

      FUNCTION frict(ax,chan,edgeflag)

            USE PATHS_WaterProp, ONLY: rhof, muf

            IMPLICIT NONE
            ! Friction Models
            INTEGER(sik) :: chan, ax, ct, edgeflag
            REAL(sdk) :: frict, Re, A, B, C, pres, vel, D, eps, rgh
            REAL(sdk) :: r,r9,r9e,f0,f1,fL,t,tL,tL2,t2,Rel,Ret
            ct=PATHS_chantype(chan)
            rgh=PATHS_rgh(ct)
            ! Aaron - edgeflag added to choose velocity and pressure on plus or minus side
            IF (edgeflag.EQ.1) THEN
               pres=PATHS_P(ax,chan)
               vel=ABS(PATHS_u(ax,chan))
               IF (ax.LT.PATHS_nz+1) THEN
                  D=PATHS_dhz(ax,ct)
               ELSE
                  D=PATHS_dhz(ax-1,ct)
               ENDIF
            ELSEIF (edgeflag.EQ.-1) THEN
               ! subtract dP due to area change
               IF (ax.LT.PATHS_nz+1) THEN
                  pres=PATHS_P(ax,chan)-PATHS_dens(ax,chan)*PATHS_u(ax,chan)**2/2*((PATHS_achan(ax,ct)/PATHS_achan(ax-1,ct))**2-1)
                  vel=ABS(PATHS_u(ax,chan)*PATHS_achan(ax,ct)/PATHS_achan(ax-1,ct))
               ELSE
                  pres=PATHS_P(ax,chan)
                  vel=ABS(PATHS_u(ax,chan))
               ENDIF
               IF (ax.GT.1) THEN
                  D=PATHS_dhz(ax-1,ct)
               ELSE
                  D=PATHS_dhz(ax,ct)
               ENDIF
            ENDIF
            
            Re=PATHS_dens(ax,chan)*vel*D/muf(pres, ax, chan, .TRUE.)   ! liquid only


            SELECT CASE (PATHS_mfr)
            CASE (1)
               A=0.186
               B=-0.2
               C=0.0
               frict=A*Re**B+C
            CASE (2) ! Churchill 
               eps=rgh/D
               !eps=0
               frict=8*((8.0/Re)**12+((2.475*LOG((7.0/Re)**0.9+0.27*eps))**16+(37530.0/Re)**16)**(-1.5))**(1.0/12.0)   
            CASE (3) ! Smooth laminar-turbulent by Yunlin Xu 
               Rel=2000
               IF(Re<=Rel)THEN
                  frict=64/Re
               ELSE
                  eps=rgh/D
                  Ret=3100+20000*eps
                  IF(Re>=Ret)THEN
                     frict=8/(2.475*LOG((7.0/Re)**0.9+0.27*eps))**2
                  ELSE
                     r=7.0/Ret
                     r9=r**0.9
                     r9e=r9+0.27*eps
                     f0=1/(2.475*LOG(r9e))**2
                     f1=SQRT(f0**3)*1.8*2.475*r9*r/(7*r9e)
                     fl=8/Rel
                     t=Ret-Re
                     t2=t*t
                     tL=Ret-Rel
                     tL2=tL*tL
                     frict=8*(f0-t*f1+t2*(2*f1/tL+(fL*(4-Ret/Rel)-3*f0)/tL2)    &
                          -t2*t*(f1/tL2+(fL*(3-Ret/Rel)-2*f0)/(tL2*tL)))
                  ENDIF
               ENDIF
            END SELECT

            RETURN
      END FUNCTION frict

      FUNCTION tpfmult(ax,chan,edgeflag)

            USE PATHS_WaterProp, ONLY: rhof, rhog, mug, muf

            IMPLICIT NONE
            ! Two Phase Friction Multiplier Models
            INTEGER(sik) :: chan, ax, ct, edgeflag
            REAL(sdk) :: tpfmult, m, P, G, PAI, Psia, vel

            ct=PATHS_chantype(chan)
            ! Aaron - edgeflag added to choose velocity and pressure on plus or minus side
            IF (edgeflag.EQ.1) THEN
               P=PATHS_P(ax,chan)
               vel=ABS(PATHS_u(ax,chan))
            ELSEIF (edgeflag.EQ.-1) THEN
               ! subtract dP due to area change
               IF (ax.LT.PATHS_nz+1) THEN
                  P=PATHS_P(ax,chan)-PATHS_dens(ax,chan)*PATHS_u(ax,chan)**2/2*((PATHS_achan(ax,ct)/PATHS_achan(ax-1,ct))**2-1)
                  vel=ABS(PATHS_u(ax,chan)*PATHS_achan(ax,ct)/PATHS_achan(ax-1,ct))
               ELSE
                  P=PATHS_P(ax,chan)
                  vel=ABS(PATHS_u(ax,chan))
               ENDIF
            ENDIF


            SELECT CASE (PATHS_mtpf)
            CASE (0) !1.5* HEM 1, only for reproduce solution of old code
               tpfmult=(1.0+PATHS_qual(ax,chan)*(rhof(P, ax, chan, .TRUE.)/rhog(P)-1.0))   &                                     
                    * PATHS_dens(ax,chan)/rhof(P, ax, chan, .TRUE.)*1.5
            CASE (1) !HEM 1
               tpfmult=(1.0+PATHS_qual(ax,chan)*(rhof(P, ax, chan, .TRUE.)/rhog(P)-1.0))   &                                     
                    * PATHS_dens(ax,chan)/rhof(P, ax, chan, .TRUE.) 
            CASE (2) !HEM 2
               m=mug(P)/muf(P, ax, chan, .TRUE.)
               tpfmult=(1+PATHS_qual(ax,chan)*(m-1))**(0.25)*(1.0+PATHS_qual(ax,chan)*(rhof(P, ax, chan, .TRUE.)/rhog(P)-1.0))  &
                    * PATHS_dens(ax,chan)/rhof(P, ax, chan, .TRUE.)
            CASE (3) !Martinelli-Nelson  
               tpfmult=(1.2*(rhof(P, ax, chan, .TRUE.)/rhog(P)-1)*PATHS_qual(ax,chan)**0.824+1) &
                    * PATHS_dens(ax,chan)/rhof(P, ax, chan, .TRUE.) 
            CASE (4) !Martinelli-Nelson-Jones  
               G=PATHS_dens(ax,chan)*ABS(vel)*737.338117E-6    !10-6 lb/h/ft2
               Psia=P*0.000145
               IF(G<0.7)THEN
                  PAI=1.36+0.0004998*Psia+(0.1 -0.000714*Psia)*G
               ELSE
                  PAI=1.26-0.0004*Psia+(0.119+0.00028*Psia)/G
               ENDIF
               tpfmult=(PAI*1.2*(rhof(P, ax, chan, .TRUE.)/rhog(P)-1)*PATHS_qual(ax,chan)**0.824+1) &
                    * PATHS_dens(ax,chan)/rhof(P, ax, chan, .TRUE.)
            CASE (5)
               tpfmult=PATHS_dens(ax,chan)/rhof(P, ax, chan, .TRUE.) 
            CASE (6)
               tpfmult=1.0
            END SELECT

            RETURN
      END FUNCTION tpfmult

      FUNCTION quality(ax,chan)
            USE PATHS_WaterProp

            IMPLICIT NONE
            ! Subcooled Boiling Models
            INTEGER(sik) :: chan, ax
            REAL(sdk) :: PI
            REAL(sdk) :: quality
            REAL(sdk) :: xd, xeq, hb, hhn, hdb, Re, Pr, Chn, Cdb, qdp, tmp1, tmp2, tmp3,Xs,Xh,dhz,nrod,achan

            PI=3.141592
            Xs=0.05
            Xh=Xs/2

            xeq=(hf(PATHS_P(ax,chan), ax, chan, .TRUE.)-PATHS_enth(ax,chan)) &
                /(hf(PATHS_P(ax,chan), ax, chan, .TRUE.)-hg(PATHS_P(ax,chan)))
            IF (ax.GT.1) THEN
               IF (PATHS_nrod(ax-1,PATHS_chantype(chan)).EQ.0) THEN
                  quality=MAX(0.0_sdk,xeq)
                  IF(quality>1) quality=1
                  RETURN
               ENDIF
            ENDIF
            
            SELECT CASE (PATHS_mscb)
            CASE (1)
               IF (xeq .GE. Xs) THEN
                  quality=xeq
               ELSE
                  IF (ax.LT.PATHS_nz+1) THEN
                     dhz = PATHS_dhz(ax,PATHS_chantype(chan))
                     nrod = PATHS_nrod(ax,PATHS_chantype(chan))
                     achan = PATHS_achan(ax,PATHS_chantype(chan))
                  ELSE
                     dhz = PATHS_dhz(ax-1,PATHS_chantype(chan))
                     nrod = PATHS_nrod(ax-1,PATHS_chantype(chan))
                     achan = PATHS_achan(ax-1,PATHS_chantype(chan))
                  ENDIF
                  Re=rhof(PATHS_P(ax,chan), ax, chan, .TRUE.)* &
                     PATHS_u(ax,chan)*dhz/muf(PATHS_P(ax,chan), ax, chan, .TRUE.)
                  Pr=(Cpf(PATHS_P(ax,chan), ax, chan, .TRUE.)* &
                      muf(PATHS_P(ax,chan), ax, chan, .TRUE.))/kf(PATHS_P(ax,chan), ax, chan, .TRUE.)
                  IF (ax .EQ. 1) THEN
                     qdp=0.0
                  ELSE
                     !IF (PATHS_nrod(ax-1,PATHS_chantype(chan)).EQ.0) THEN   ! Assume qdp=0 for nodes with no fuel rods
                        !qdp=0.0  
                        !qdp=PATHS_Prel(ax-1,chan)*plevel*PATHS_RTP*PATHS_symmmult* &
                        ! (1.0-PATHS_fracdc-PATHS_fracdwr-PATHS_fracdvb)/(PI*PATHS_dhz(PATHS_chantype(chan))*PATHS_hz(ax-1))
                     !ELSE
                           !Aaron - check indexing (ax-1 or ax?)
                        qdp=PATHS_Prel(ax-1,chan)*PATHS_hz(ax-1)*plevnew*PATHS_RTP*PATHS_symmmult* &
                            (1.0-PATHS_fracdc-PATHS_fracdwr-PATHS_fracdvb) &
                            /(REAL(PATHS_nasm)*PATHS_nrod(ax-1,PATHS_chantype(chan)) &
                            *2*PI*PATHS_rrod(PATHS_chantype(chan))*PATHS_hz(ax-1))
                     !ENDIF
                  ENDIF
                  hb=EXP(PATHS_P(ax,chan)/4.35e6)/(22.7)**2*1000.0
                  Chn=0.2/4.0*dhz/PATHS_rrod(PATHS_chantype(chan))
                  hhn=Chn*Re**0.662*Pr*kf(PATHS_P(ax,chan), ax, chan, .TRUE.)/dhz
                  ! Aaron - NEED TO INCLUDE PART-LENGTH RODS FOR AREA CALCS; also, check achan indexing
                  Cdb=0.033*achan/(achan        &
                     +nrod*PI*PATHS_rrod(PATHS_chantype(chan))**2            &
                     +PATHS_nwrod(PATHS_chantype(chan))*PI*PATHS_rwrod(PATHS_chantype(chan))**2)+0.013
                  hdb=Cdb*Re**0.8*Pr**0.4*kf(PATHS_P(ax,chan), ax, chan, .TRUE.)/dhz
                  tmp1=4.0*hb*(hdb+hhn)**2
                  tmp2=2.0*hdb**2*(hhn+hdb/2.0)+8.0*qdp*hb*(hdb+hhn)
                  tmp3=qdp*(4.0*hb*qdp+hdb**2)
                  xd=-Cpf(PATHS_P(ax,chan), ax, chan, .TRUE.)/(hg(PATHS_P(ax,chan))                                       &
                       -hf(PATHS_P(ax,chan), ax, chan, .TRUE.))*((-tmp2+SQRT(tmp2**2-4.0*tmp1*tmp3))/(2.0*tmp1))
                  !|xd| is stored in xd
                  IF (xeq .LE. 0.0) THEN
                     IF (xeq .LE. (-xd)) THEN
                        quality=0.0
                     ELSE
                        tmp1=1+xeq/xd
                        tmp2=tmp1*tmp1
                        quality= xd*tmp2*(0.1+0.087*tmp1+0.05*tmp2)
                     ENDIF
                  ELSEIF (Xh .GT. xd ) THEN
                     IF(xeq .GE. 2*xd) THEN
                        quality=xeq
                     ELSE
                        tmp1=xeq/xd
                        quality= xd*(0.237 + tmp1*(0.661+ tmp1*(0.153+ tmp1*(-0.01725 - tmp1*0.0020625) ) ) )
                     ENDIF
                  ELSE
                     tmp1=xeq/Xh
                     tmp2=xd/Xh
                     tmp3=0.237*tmp2  
                     quality= Xh*( tmp3 + tmp1*(0.661 +tmp1*(0.5085 - 0.3555*tmp2              &
                          + tmp1*(tmp3-0.25425 + tmp1 *(0.042375 - 0.0444375*tmp2) ) ) ) )
                  ENDIF
               ENDIF
            CASE (2) 
               quality=MAX(0.0_sdk,xeq)
            END SELECT
            !IF(quality>1)quality=1
            IF(quality>=1.0)quality=0.99
            RETURN
      END FUNCTION quality

      SUBROUTINE voidfract(ax,chan)
            ! Aaron - need to fix this with edgeflag

            USE PATHS_WaterProp, ONLY: rhof, rhog, muf, surften

            IMPLICIT NONE
            ! Void Fraction Models
            INTEGER(sik) :: chan, ax
            REAL(sdk) :: C0, Dh_nd, Nul, Dhlimit, rho
            REAL(sdk) :: Pcrit, pf, pg, mf, sten, Re, k1, k0, r, C1, vftol
            REAL(sdk) :: dvf, vfnew, alphavf, vgj0, jg, jf, jgst, vf_slug, mflux, qual_dum, dhz,iter,beta

            ! PATHS version 2.0 revision
            Pcrit=22.0640*10**6
            vftol=1E-5
            alphavf=0.75
            pf=rhof(PATHS_P(ax,chan), ax, chan, .TRUE.)
            pg=rhog(PATHS_P(ax,chan))
            mf=muf(PATHS_P(ax,chan), ax, chan, .TRUE.)
            sten=surften(PATHS_P(ax,chan))
            
!            mf=muf(PATHS_P(ax,chan), ax, chan, .TRUE.)
!            sten=17.189e-3

            IF (ax < PATHS_nz+1) THEN
               dhz = PATHS_dhz(ax,PATHS_chantype(chan))
            ELSE
               dhz = PATHS_dhz(ax-1,PATHS_chantype(chan))
            ENDIF

            !PATHS_qual(ax,chan) = 0.5
            !PATHS_u(ax,chan) = 1.0
            !PATHS_P(ax,chan) = 7.2e6
            !PATHS_dens(ax,chan) = pf
            !PATHS_dhz(PATHS_chantype(chan)) = 0.013
            !WRITE(34,*) 'Test'
            PATHS_vgj(ax,chan)=0
            PATHS_vf(ax,chan)=0
            IF (PATHS_qual(ax,chan) .EQ. 0.0) THEN
               PATHS_vf(ax,chan) = 0.0_sdk
            ELSEIF(PATHS_qual(ax,chan) .EQ. 1.0) THEN
               PATHS_vf(ax,chan) = 1.0_sdk
            ELSE
               SELECT CASE (PATHS_mvd)
               CASE (1) ! EPRI Void
                  PATHS_vgj(ax,chan)=0
                  C0=1.0
                  PATHS_vf(ax,chan)=PATHS_qual(ax,chan)/(C0*(PATHS_qual(ax,chan)                           &
                       +rhog(PATHS_P(ax,chan))/rhof(PATHS_P(ax,chan), ax, chan, .TRUE.)*(1.-PATHS_qual(ax,chan)))             &
                       +rhog(PATHS_P(ax,chan))*PATHS_vgj(ax,chan)/(PATHS_dens(ax,chan)*PATHS_u(ax,chan)))
                  dvf=2.0*vftol
                  DO WHILE (dvf .GT. vftol)
! v32m10                      
!                     rho=PATHS_vf(ax,chan)*pg+(1.0-PATHS_vf(ax,chan))*pf
!                     Re=rho*ABS(PATHS_u(ax,chan))*PATHS_dhz(PATHS_chantype(chan))/mf                      
! v32m10 updated
!                     rho=PATHS_vf(ax,chan)*pg+(1.0-PATHS_vf(ax,chan))*pf
!                     IF(ax.LE.PATHS_nz) THEN
!                        Re=rho*ABS(PATHS_u(ax,chan))*PATHS_dhz(ax,PATHS_chantype(chan))/mf   
!                     ELSE
!                        Re=rho*ABS(PATHS_u(ax,chan))*PATHS_dhz(ax-1,PATHS_chantype(chan))/mf   
!                     ENDIF                     
! p2.0
                     IF(ax.LE.PATHS_nz) THEN
                        Re=PATHS_dens(ax,chan)*ABS(PATHS_u(ax,chan))*PATHS_dhz(ax,PATHS_chantype(chan))/mf   
                     ELSE
                        Re=PATHS_dens(ax,chan)*ABS(PATHS_u(ax,chan))*PATHS_dhz(ax-1,PATHS_chantype(chan))/mf   
                     ENDIF
                     k1=MIN(0.8_sdk,1.0/(1.0+EXP(-Re/1E5)))
                     k0=k1+(1.0-k1)*(pg/pf)**0.2
                     r=(1.0+1.57*pg/pf)/(1.0-k1)
                     C1=4.0*Pcrit**2/(PATHS_P(ax,chan)*(Pcrit-PATHS_P(ax,chan)))
                     C0=1.0/(k0+(1.0-k0)*PATHS_vf(ax,chan)**r*(1.0-EXP(-C1))/(1.0-EXP(-C1*PATHS_vf(ax,chan))))
                     PATHS_vgj(ax,chan)=SQRT(2.0)*(sten*grav*(pf-pg)/pf**2)**0.25*(1.0-PATHS_vf(ax,chan))**1.5
                     vfnew=PATHS_qual(ax,chan)/(C0*(PATHS_qual(ax,chan)+pg/pf*(1.-PATHS_qual(ax,chan)))         &
                          +pg*PATHS_vgj(ax,chan)/(PATHS_dens(ax,chan)*PATHS_u(ax,chan)))   
                     IF (vfnew .GT. 1.0) THEN
                        vfnew=1.0
                     ELSEIF (vfnew .LT. 0.0) THEN
                        vfnew=0.0
                     ENDIF
                     dvf=ABS(vfnew-PATHS_vf(ax,chan))
                     PATHS_vf(ax,chan)=alphavf*vfnew+(1.0-alphavf)*PATHS_vf(ax,chan)
                  ENDDO
               CASE (2)  ! GE Ramp
                  PATHS_vgj(ax,chan)=0
                  C0=1.0
                  PATHS_vf(ax,chan)=PATHS_qual(ax,chan)/(C0*(PATHS_qual(ax,chan)+pg   &
                       /pf*(1.-PATHS_qual(ax,chan)))                               &
                       +pg*PATHS_vgj(ax,chan)                                      &
                       /(PATHS_dens(ax,chan)*PATHS_u(ax,chan)))
                  vgj0=2.9*(grav*sten*(pf-pg)/pf**2)**0.25
                  dvf=2.0*vftol
                  DO WHILE (dvf .GT. vftol)
                     !rho=PATHS_vf(ax,chan)*pg+(1.0-PATHS_vf(ax,chan))*pf  ! Aaron - shouldn't use this
                     IF (PATHS_vf(ax,chan) .LE. 0.65) THEN
                        C0=1.1
                        PATHS_vgj(ax,chan)=vgj0
                     ELSE
                        C0=1+0.1*(1-PATHS_vf(ax,chan))/0.35
                        PATHS_vgj(ax,chan)=vgj0*(1-PATHS_vf(ax,chan))/0.35
                     ENDIF
                     vfnew=PATHS_qual(ax,chan)/(C0*(PATHS_qual(ax,chan)+pg/pf*(1.-PATHS_qual(ax,chan)))  &
                          +pg*PATHS_vgj(ax,chan)/(PATHS_dens(ax,chan)*PATHS_u(ax,chan)))   
                          ! Aaron - changed rho*PATHS_u(ax,chan) to PATHS_dens(ax,chan)*PATHS_u(ax,chan)
                     IF (vfnew .GT. 1.0) THEN
                        vfnew=1.0
                     ELSEIF (vfnew .LT. 0.0) THEN
                        vfnew=0.0
                     ENDIF
                     dvf=ABS(vfnew-PATHS_vf(ax,chan))
                     PATHS_vf(ax,chan)=alphavf*vfnew+(1.0-alphavf)*PATHS_vf(ax,chan)
                  ENDDO
               CASE (3)  ! Modified Bestion
                  PATHS_vgj(ax,chan)=0.188*SQRT(grav*(pf-pg)*dhz/pg)
                  C0=1.2-0.2*SQRT(pg/pf)
                  PATHS_vf(ax,chan)=PATHS_qual(ax,chan)/(C0*(PATHS_qual(ax,chan)+pg   &
                       /pf*(1.-PATHS_qual(ax,chan)))                                  &
                       +pg*PATHS_vgj(ax,chan)/(PATHS_dens(ax,chan)*PATHS_u(ax,chan)))
               CASE (4)  ! HEM
                  PATHS_vgj(ax,chan)=0
                  C0=1.0
                  PATHS_vf(ax,chan)=PATHS_qual(ax,chan)/(C0*(PATHS_qual(ax,chan)+pg  &
                       /pf*(1.-PATHS_qual(ax,chan)))                                    &
                       +pg*PATHS_vgj(ax,chan)/(PATHS_dens(ax,chan)*PATHS_u(ax,chan)))
               CASE (5)  ! Bestion matching TRACE
                  PATHS_vgj(ax,chan)=0.188*SQRT(grav*(pf-pg)*PATHS_dhz(ax,PATHS_chantype(chan))/pg)
                  C0=1.0 
                  PATHS_vf(ax,chan)=PATHS_qual(ax,chan)/(C0*(PATHS_qual(ax,chan)+rhog(PATHS_P(ax,chan))   &
                       /rhof(PATHS_P(ax,chan), ax, chan, .TRUE.)*(1.-PATHS_qual(ax,chan)))                &
                       +rhog(PATHS_P(ax,chan))*PATHS_vgj(ax,chan)/(PATHS_dens(ax,chan)*PATHS_u(ax,chan)))
               CASE (6) ! Kataoka-Ishii Model
                   Dhlimit=30
                   Dh_nd=PATHS_dhz(ax,PATHS_chantype(chan))/SQRT(sten/(grav*(pf-pg)))
                   Nul=mf/SQRT(pf*sten*SQRT(sten/(grav*(pf-pg))))
                   C0=1.2-0.2*SQRT(pg/pf)
                   PATHS_vgj(ax,chan)=0.0019*((min(Dhlimit,Dh_nd))**0.809)*((pg/pf)**(-0.157))* &
                         (Nul**(-0.562))*((sten*grav*(pf-pg)/(pf**2))**0.25)
                   PATHS_vf(ax,chan)=PATHS_qual(ax,chan)/(C0*(PATHS_qual(ax,chan)+pg   &
                       /pf*(1.-PATHS_qual(ax,chan)))                                  &
                       +pg*PATHS_vgj(ax,chan)/(PATHS_dens(ax,chan)*PATHS_u(ax,chan)))
               CASE (7)  ! Liao, Parlos, and Griffith
                  mflux = PATHS_dens(ax,chan)*PATHS_u(ax,chan)
                  jg=mflux*PATHS_qual(ax,chan)/pg
                  jf=mflux*(1.-PATHS_qual(ax,chan))/pf
                  jgst=jg*SQRT(pg/(grav*dhz*(pf-pg)))
                  qual_dum = pg/mflux*SQRT(grav*dhz*(pf-pg)/pg)
                  vf_slug=0
                  C0=1.0
                  PATHS_vgj(ax,chan)=0.188*SQRT(grav*dhz*(pf-pg)/pg)
                  PATHS_vf(ax,chan)=PATHS_qual(ax,chan)/(C0*(PATHS_qual(ax,chan)+pg   &
                       /pf*(1.-PATHS_qual(ax,chan)))                               &
                       +pg*PATHS_vgj(ax,chan)                                      &
                       /(mflux))
                  
                  IF (jgst.LE.1.0) THEN
                     dvf=2.0*vftol
                     iter=0
                     DO WHILE(.TRUE.) 
                        iter=iter+1
                        IF (jf.GT.(2.34-1.07*(grav*sten*(pf-pg)/pf**2)**0.25)) THEN  !Bubbly flow
                           C0=1.0
                           PATHS_vgj(ax,chan)=1.53*(1-PATHS_vf(ax,chan))**2*(grav*sten*(pf-pg)/pf**2)**0.25
                           PATHS_regnm(ax,chan)=1.0
                        ELSE  ! Churn/slug
                           C0=1.2-0.2*SQRT(pg/pf)*(1-exp(-18*PATHS_vf(ax,chan)))
                           PATHS_vgj(ax,chan)=0.33*(grav*sten*(pf-pg)/pf**2)**0.25
                           PATHS_regnm(ax,chan)=2.0
                        ENDIF
                        vfnew=PATHS_qual(ax,chan)/(C0*(PATHS_qual(ax,chan)+pg/pf*(1.-PATHS_qual(ax,chan)))  &
                             +pg*PATHS_vgj(ax,chan)/(mflux))
                        IF (vfnew .GT. 1.0) THEN
                           vfnew=1.0
                        ELSEIF (vfnew .LT. 0.0) THEN
                           vfnew=0.0
                        ENDIF
                        dvf=ABS(vfnew-PATHS_vf(ax,chan))
                        IF (dvf < vftol) EXIT
                        IF (iter > 100) EXIT
                        PATHS_vf(ax,chan)=alphavf*vfnew+(1.0-alphavf)*PATHS_vf(ax,chan)
                     ENDDO
                  ELSE
                     dvf=2.0*vftol
                     iter=0
                     DO WHILE(.TRUE.)
                        iter=iter+1
                        IF (jf.GT.(2.34-1.07*(grav*sten*(pf-pg)/pf**2)**0.25)) THEN  !Bubbly flow
                           C0=1.0
                           PATHS_vgj(ax,chan)=1.53*(1-vf_slug)**2*(grav*sten*(pf-pg)/pf**2)**0.25
                        ELSE  ! Churn/slug
                           C0=1.2-0.2*SQRT(pg/pf)*(1-exp(-18*vf_slug))
                           PATHS_vgj(ax,chan)=0.33*(grav*sten*(pf-pg)/pf**2)**0.25
                        ENDIF
                        vfnew=qual_dum/(C0*(qual_dum+pg/pf*(1.-qual_dum))  &
                             +pg*PATHS_vgj(ax,chan)/(mflux))
                        IF (vfnew .GT. 1.0) THEN
                           vfnew=1.0
                        ELSEIF (vfnew .LT. 0.0) THEN
                           vfnew=0.0
                        ENDIF
                        dvf=ABS(vfnew-vf_slug)
                        IF (dvf < vftol) EXIT
                        IF (iter > 100) EXIT
                        vf_slug=alphavf*vfnew+(1.0-alphavf)*vf_slug
                     ENDDO
                  
                     iter=0
                     DO WHILE(.TRUE.)
                        iter=iter+1
                        C0=1.0+(1-PATHS_vf(ax,chan))/(PATHS_vf(ax,chan)+4.0*SQRT(pg/pf))
                        PATHS_vgj(ax,chan)=(C0-1)*SQRT(grav*dhz*(pf-pg)*(1-PATHS_vf(ax,chan))/(0.015*pf))
                        PATHS_regnm(ax,chan)=3.0
                        vfnew=PATHS_qual(ax,chan)/(C0*(PATHS_qual(ax,chan)+pg/pf*(1.-PATHS_qual(ax,chan)))  &
                             +pg*PATHS_vgj(ax,chan)/(mflux))
                        IF (vfnew .GT. 1.0) THEN
                           vfnew=1.0
                        ELSEIF (vfnew .LT. 0.0) THEN
                           vfnew=0.0
                        ENDIF
                        dvf=ABS(vfnew-PATHS_vf(ax,chan))
                        IF (dvf < vftol) EXIT
                        IF (iter > 100) EXIT
                        PATHS_vf(ax,chan)=alphavf*vfnew+(1.0-alphavf)*PATHS_vf(ax,chan)
                     ENDDO
                     IF (PATHS_vf(ax,chan) < vf_slug) THEN
                        PATHS_vf(ax,chan)=vf_slug
                        PATHS_regnm(ax,chan)=2.5
                     ENDIF
                  ENDIF
               CASE (8)  ! Zuber-Findlay
                  PATHS_vgj(ax,chan)=2.9*((pf-pg)*grav*sten/(pf**2))**0.25
                  beta=PATHS_qual(ax,chan)/(PATHS_qual(ax,chan)+pg/pf*(1-PATHS_qual(ax,chan)))
                  C0=beta*(1+(1/beta-1)**((pg/pf)**0.1))
                  
                  PATHS_vf(ax,chan)=PATHS_qual(ax,chan)/(C0*(PATHS_qual(ax,chan)+pg  &
                       /pf*(1.-PATHS_qual(ax,chan)))                                    &
                       +pg*PATHS_vgj(ax,chan)/(PATHS_dens(ax,chan)*PATHS_u(ax,chan)))
               END SELECT
            ENDIF

      END SUBROUTINE voidfract

END MODULE PATHS_thmodels
