!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
MODULE PATHS_WaterProp

      USE IntrType, ONLY: sdk, sik
      USE EosIAPWSCrunch, ONLY: getsatatp, clearprops, gettsat, getsnglatpu, getsnglatpt
      USE EosIAPWSData
      USE PATHS_thvarM, ONLY: stable, PATHS_enth, PATHS_qual, PATHS_dens, PATHS_vf, stm_opt

      IMPLICIT NONE
      
      LOGICAL :: err, iterate
      INTEGER(sik) :: iLastLiqVisc = 1_sik  ! Last index used in the liquid viscosity interpolation.
      INTEGER(sik) :: iLastVapVisc = 1_sik
      INTEGER(sik) :: iLastLiqCond = 1_sik  ! Last index used in the liquid conductivity interpolation.
      INTEGER(sik) :: iLastVapCond = 1_sik

      ! If pressure above 12 MPa, the properties ~linear with pressure
      ! Using linear interpolation between 12 and 16 Mpa, which should
      ! cover VVER and PWR, thus avoding steam tables
      
CONTAINS

      FUNCTION t_to_enth(pres, t)
         REAL(sdk) :: pres, t,t_to_enth, u
         
         CALL clearprops(getprops)
         getprops(ubar)= .TRUE.
         getprops(hbar)= .TRUE.
         CALL getsnglatpt(tables, liquid, getprops, t, pres, s, err, iterate)
         u = s(ubar)
         t_to_enth = s(hbar)         
      END FUNCTION t_to_enth

      FUNCTION rhof(pres, aface,channel, edge)
            USE Eos, ONLY: RhoLiIAPWS
            
            IMPLICIT NONE
            ! cubic poly for fluid density as a func of pressure
            REAL(sdk) :: rhof, pres, pr, rhol, h, rho, u, x, t
            REAL(sdk) :: presB, presT, rhoB, rhoT, slp, brhs, hsat
            INTEGER(sik) :: channel, aface
            LOGICAL :: edge, hsub
            
            IF (stable) THEN 
                IF (edge) THEN
                    h=PATHS_enth(aface, channel)*1000
                    rho=PATHS_dens(aface, channel)
                    x=PATHS_qual(aface, channel)
                ELSE IF (.NOT. edge) THEN
                    h=0.5*(PATHS_enth(aface, channel)+PATHS_enth(aface+1, channel))*1000
                    rho=0.5*(PATHS_dens(aface, channel)+PATHS_dens(aface+1, channel))
                    x=0.5*(PATHS_qual(aface, channel)+PATHS_qual(aface+1, channel))
                ENDIF
                rho=(rho-x*rhog(pres))/(1-x)
                h=(h-x*hg(pres)*1000)/(1-x)
                !IF (x<=0.05) THEN
                    u=h-pres/rho
                    CALL clearprops(getprops)
                    getprops(tempbar)= .TRUE.
                    CALL getsnglatpu(tables, liquid, getprops, u, pres, s, err, iterate)
                    t=s(tempbar)
                    CALL RhoLiIAPWS(pres, t, rhol, iterate)
                    rhof=rhol
                !ELSE
                !    !Saturated
                !    CALL getsatatp(tables, pres, s, err)
                !    rhof=1/s(vsubf)
                !END IF
            ELSE
                IF (pres .LT. 12e6) THEN
                    hsub = .FALSE.
                    IF (stm_opt .LT. 0) THEN
                       pr=pres/1.0e6
                       h=PATHS_enth(aface, channel)
                       
                       hsat = -4.9662E-02 * pr**4 + 1.6049E+00 * pr**3 &
                             - 2.1104E+01 * pr**2 + 1.7916E+02 * pr + 6.1637E+02
                       IF (h .LT. hsat) hsub = .TRUE.
                    
                       presB = 6.0
                       presT = 8.0
                       IF (pr .LT. presB) THEN
                           rhof = -2.6539E-10 * h**4 + 1.0611E-06 * h**3 &
                                 - 1.7175E-03 * h**2 + 1.0181E+00 * h + 7.3112E+02
                       ELSE IF (pr .GE. presB .AND. pr .LT. presT) THEN
                           rhoB = -2.6539E-10 * h**4 + 1.0611E-06 * h**3 &
                                 - 1.7175E-03 * h**2 + 1.0181E+00 * h + 7.3112E+02
                           rhoT = -7.6425E-11 * h**4 + 2.5099E-07 * h**3 &
                                 - 4.1333E-04 * h**2 + 8.6953E-02 * h + 9.8129E+02
                           slp = (rhoT - rhoB) / (presT - presB)
                           brhs = rhoB - slp * presB
                           rhof = pr * slp + brhs
                       ELSE IF (pr .GT. presT) THEN
                           rhof = -7.6425E-11 * h**4 + 2.5099E-07 * h**3 &
                                 - 4.1333E-04 * h**2 + 8.6953E-02 * h + 9.8129E+02
                       END IF
                    END IF
                    IF (.NOT. hsub) THEN
                        pr=pres/10**6
                        rhof=-3.8586E-2*pr**3+1.1801*pr**2-2.8714E1*pr+8.9613E2
                    END IF
                ELSE
                    ! Liquid water
                    pr=pres/1.0e6
                    h=PATHS_enth(aface, channel)
                    
                    presB = 12.0
                    presT = 16.0
                    
                    rhoB = -2.1573E-10*h**4 + 9.5374E-07*h**3 &
                          - 1.7230E-03*h**2 + 1.1659E+00*h + 6.5217E+02
                    rhoT = -1.4090E-10*h**4 + 6.0798E-07*h**3 &
                          - 1.1132E-03*h**2 + 6.8714E-01*h + 7.9607E+02

                    slp = (rhoT - rhoB) / (presT - presB)
                    brhs = rhoB - slp * presB
                    rhof = pr * slp + brhs                    
                END IF    
            ENDIF
            RETURN
      END FUNCTION rhof

      FUNCTION rhog(pres)
            IMPLICIT NONE
            ! cubic poly for vapor density as a func of pressure
            REAL(sdk) :: rhog, pres, pr
            REAL(sdk) :: presB, presT, slp, brhs
            REAL(sdk) :: rhoB, rhoT, h

            IF (stable)THEN
                CALL getsatatp(tables, pres, s, err)
                rhog=1/s(vsubg)             
            ELSE
                IF (pres .LT. 12e6) THEN
                    pr=pres/10**6
                    rhog=7.2862E-3*pr**3-1.6397E-2*pr**2+4.9955*pr-1.3859E-1
                ELSE
                    ! Assuming vapor fraction should be zero, but data is correct
                    pr=pres/1.0e6
                    
                    presB = 12.0
                    presT = 16.0
                    
                    rhoB = 70.106  ! 597.83 Kelvin trips to steam
                    rhoT = 107.42  ! 620.05 Kelvin trips to steam

                    slp = (rhoT - rhoB) / (presT - presB)
                    brhs = rhoB - slp * presB
                    rhog = pr * slp + brhs                 
                END IF
            ENDIF

            RETURN
      END FUNCTION rhog

      FUNCTION hf(pres, aface, channel, edge)
            IMPLICIT NONE
            ! cubic poly for fluid enthalpy as a func of pressure
            REAL(sdk) :: hf, pres, pr, h, rho, u, x
            REAL(sdk) :: r, enthB, enthT
            REAL(sdk) :: presB, presT, slp, brhs
            INTEGER(sik) :: channel, aface
            LOGICAL :: edge
            
            IF (stable) THEN
                IF (edge) THEN
                    h=PATHS_enth(aface, channel)*1000
                    rho=PATHS_dens(aface, channel)
                    x=PATHS_qual(aface, channel)
                ELSE
                    h=0.5*(PATHS_enth(aface, channel)+PATHS_enth(aface+1, channel))
                    rho=0.5*(PATHS_dens(aface, channel)+PATHS_dens(aface+1, channel))
                    x=0.5*(PATHS_qual(aface, channel)+PATHS_qual(aface+1, channel))
                ENDIF
                rho=(rho-x*rhog(pres))/(1-x)
                h=(h-x*hg(pres)*1000)/(1-x)
                !IF (x<=0.05) THEN
                    u=h-pres/rho
                    CALL clearprops(getprops)
                    getprops(hsubf)= .TRUE. ! changed to hsubf instead of hbar
                    CALL getsnglatpu(tables, liquid, getprops, u, pres, s, err, iterate)
                    hf=s(hsubf)/1E3
                !ELSE
                !    !Saturated
                !    CALL getsatatp(tables, pres, s, err)
                !    hf=s(hsubf)/1E3
                !END IF
            ELSE
                IF (pres .LT. 12E6) THEN
                    pr=pres/10**6
                    hf=1.5219E-1*pr**3-5.2291*pr**2+1.0238E2*pr+7.5500E2
                ELSE
                    ! Liquid water
                    pr=pres/1.0e6

                    hf = 6.7032E-02*pr**3 - 2.9117E+00*pr**2 &
                       + 8.1389E+01*pr + 8.1825E+02

                END IF    
            ENDIF          

            RETURN
      END FUNCTION hf

      FUNCTION hg(pres)
            IMPLICIT NONE
            ! cubic poly for vapor enthalpy as a func of pressure
            REAL(sdk) :: hg, pres, pr, enthB, enthT, r
            REAL(sdk) :: presB, presT, slp, brhs
            
            IF (stable)THEN
                CALL getsatatp(tables, pres, s, err)
                hg=s(hsubg)/1E3
            ELSE
                IF (pres .LT. 12E6) THEN
                    pr=pres/10**6
                    hg=2.8956E-2*pr**3-1.6014*pr**2+5.1714*pr+2.8050E3
                ELSE
                    ! Assuming vapor fraction should be zero, but data is correct
                    pr=pres/1.0e6

                    hg = -5.4825E-02*pr**3 + 1.1137E+00*pr**2 &
                        - 2.4883E+01*pr + 2.9184E+03
                END IF
            ENDIF

            RETURN
      END FUNCTION hg

      FUNCTION muf(pres, aface, channel, edge)
            USE EosIAPWSViscosity, ONLY: ViscosityLiq
             
            IMPLICIT NONE
            ! cubic poly for fluid viscosity as a func of pressure
            REAL(sdk) :: muf, pres, pr, h, rho, u, x, t, dRhodP
            REAL(sdk) :: presB, presT, slp, brhs
            REAL(sdk) :: muB, muT
            INTEGER(sik) :: channel, aface  
            LOGICAL :: edge
            
            IF (stable) THEN
                IF (edge) THEN
                    h=PATHS_enth(aface, channel)*1000
                    rho=PATHS_dens(aface, channel)
                    x=PATHS_qual(aface, channel)
                ELSE
                    h=0.5*(PATHS_enth(aface, channel)+PATHS_enth(aface+1, channel))
                    rho=0.5*(PATHS_dens(aface, channel)+PATHS_dens(aface+1, channel))
                    x=0.5*(PATHS_qual(aface, channel)+PATHS_qual(aface+1, channel))
                ENDIF
                rho=(rho-x*rhog(pres))/(1-x)
                h=(h-x*hg(pres)*1000)/(1-x)
                dRhodP=0 ! Need a real value??
                !IF (x<=0.05) THEN
                    u=h-pres/rho
                    CALL clearprops(getprops)
                    getprops(tempbar)= .TRUE.
                    CALL getsnglatpu(tables, liquid, getprops, u, pres, s, err, iterate)
                    t=s(tempbar)
                    muf=ViscosityLiq(pres, t, rho, dRhodP, iLastLiqVisc) 
                !ELSE
                !    CALL getsatatp(tables, pres, s, err)
                !    t=s(tsatbar)
                !    rho=1/s(vsubf)
                !    muf=ViscosityLiq(pres, t, rho, dRhodP, iLastLiqVisc) 
                !ENDIF
            ELSE
                IF (pres .LT. 12E6) THEN
                    pr=pres/10**6
                    muf=-2.1044E-8*pr**3+6.9411E-7*pr**2-1.0344E-5*pr+1.3687E-4
                ELSE
                    ! Liquid water
                    pr=pres/1.0e6
                    h=PATHS_enth(aface, channel)
                    
                    presB = 12.0
                    presT = 16.0
                    
                    muB = -6.9005E-14*h**3 + 2.9949E-10*h**2 &
                          - 5.0339E-07*h + 3.9012E-04
                    muT = -6.5269E-14*h**3 + 2.8803E-10*h**2 &
                          - 4.9167E-07*h + 3.8720E-04

                    slp = (muT - muB) / (presT - presB)
                    brhs = muB - slp * presB
                    muf = pr * slp + brhs                     
                END IF
            ENDIF

            RETURN
      END FUNCTION muf

      FUNCTION mug(pres)
            USE EosIAPWSViscosity, ONLY: ViscosityVap
      
            IMPLICIT NONE
            ! cubic poly for vapor viscosity as a func of pressure
            REAL(sdk) :: mug, pres, pr
            REAL(sdk) :: muB, muT
            REAL(sdk) :: presB, presT, slp, brhs
            
            IF (stable)THEN
                mug= ViscosityVap(pres, iLastVapVisc) 
            ELSE
                IF (pres .LT. 12E6) THEN
                    pr=pres/10**6
                    mug=1.4276E-9*pr**3-3.6341E-8*pr**2+7.4087E-7*pr+1.5065E-5
                ELSE
                    ! Assuming vapor fraction should be zero, but data is correct
                    pr=pres/1.0e6
                    
                    presB = 12.0
                    presT = 16.0

                    muB = 7.7943e-05  ! 597.83 Kelvin trips to steam, liquid val
                    muT = 7.9968e-05  ! 620.05 Kelvin trips to steam, liquid val                    
!                    muB = 2.1183e-05  ! 597.83 Kelvin trips to steam, vapor val
!                    muT = 2.3442e-05  ! 620.05 Kelvin trips to steam, vapor val

                    slp = (muT - muB) / (presT - presB)
                    brhs = muB - slp * presB
                    mug = pr * slp + brhs                    
                END IF            
            ENDIF

            RETURN
      END FUNCTION mug

      FUNCTION cpf(pres, aface, channel, edge)
            USE Eos, ONLY: CpllIAPWS
            
            IMPLICIT NONE
            ! cubic poly for fluid specific heat as a func of pressure
            REAL(sdk) :: cpf, pres, pr, h, rho, u, x, t
            REAL(sdk) :: presB, presT, slp, brhs
            REAL(sdk) :: cpfB, cpfT
            INTEGER(sik) :: channel, aface
            LOGICAL :: edge
            
            IF (stable) THEN
                IF (edge) THEN
                    h=PATHS_enth(aface, channel)*1000
                    rho=PATHS_dens(aface, channel)
                    x=PATHS_qual(aface, channel)
                ELSE
                    h=0.5*(PATHS_enth(aface, channel)+PATHS_enth(aface+1, channel))
                    rho=0.5*(PATHS_dens(aface, channel)+PATHS_dens(aface+1, channel))
                    x=0.5*(PATHS_qual(aface, channel)+PATHS_qual(aface+1, channel))
                ENDIF
                rho=(rho-x*rhog(pres))/(1-x)
                h=(h-x*hg(pres)*1000)/(1-x)
                !IF (x<=0.05) THEN
                    u=h-pres/rho
                    CALL clearprops(getprops)
                    getprops(tempbar)= .TRUE.
                    CALL getsnglatpu(tables, liquid, getprops, u, pres, s, err, iterate)
                    t=s(tempbar)        
                    cpf=CpllIAPWS(t, pres, iterate)/1E3
                !ELSE
                !    CALL gettsat(tables, pres, s, err)
                !    t=s(tsatbar)
                !    cpf=CPllIAPWS(t, pres, iterate)/1E3
                !ENDIF
            ELSE
                IF (pres .LT. 12E6) THEN
                    pr=pres/10**6
                    cpf=1.0761E-3*pr**3-1.2432E-2*pr**2+2.1603E-1*pr+4.1305
                ELSE
                    ! Liquid water
                    pr=pres/1.0e6
                    h=PATHS_enth(aface, channel)
                    
                    presB = 12.0
                    presT = 16.0
                    
                    cpfB = 5.2873E-11*h**4 - 2.5066E-07*h**3 &
                         + 4.5071E-04*h**2 - 3.6139E-01*h + 1.1319E+02
                    cpfT = 2.9538E-11*h**4 - 1.3750E-07*h**3 &
                         + 2.4411E-04*h**2 - 1.9342E-01*h + 6.1905E+01

                    slp = (cpfT - cpfB) / (presT - presB)
                    brhs = cpfB - slp * presB
                    cpf = pr * slp + brhs                     
                END IF
            ENDIF                
            
            RETURN
      END FUNCTION cpf

      FUNCTION cpg(pres)
            IMPLICIT NONE
            ! cubic poly for vapor specific heat as a func of pressure
            REAL(sdk) :: cpg, pres, pr, t
            REAL(sdk) :: presB, presT, slp, brhs            
            REAL(sdk) :: cpB, cpT
            
            IF (stable) THEN
                CALL gettsat(tables, pres, s, err)
                t=s(tsatbar)
                CALL clearprops(getprops)
                getprops(cpbar) = .TRUE.
                CALL getsnglatpt(tables, vapor, getprops, t, pres, s, err, iterate)
                cpg=s(cpbar)/1E3
            ELSE
                IF (pres .LT. 12E6) THEN
                    pr=pres/10**6
                    cpg=2.7138E-3*pr**3-3.3158E-2*pr**2+5.6397E-1*pr+2.1028
                ELSE
                    ! Assuming vapor fraction should be zero, but data is correct
                    pr=pres/1.0e6
                    
                    presB = 12.0
                    presT = 16.0
                    
                    cpB = 8.8270  ! 597.83 Kelvin trips to steam
                    cpT = 15.198  ! 620.05 Kelvin trips to steam

                    slp = (cpT - cpB) / (presT - presB)
                    brhs = cpB - slp * presB
                    cpg = pr * slp + brhs                    
                END IF            
            ENDIF
            
            RETURN
      END FUNCTION cpg

      FUNCTION kf(pres, aface, channel, edge)
            USE EosIAPWSConductivity, ONLY: ConductivityLiq
      
            IMPLICIT NONE
            ! cubic poly for fluid thermal conductivity as a func of pressure
            REAL(sdk) :: kf, pres, pr, h, rho, u, x, t, rhol
            REAL(sdk) :: presB, presT, slp, brhs
            REAL(sdk) :: kfB, kfT    
            
            INTEGER(sik) :: channel, aface
            LOGICAL :: edge
            
            IF (stable) THEN
                IF (edge) THEN
                    h=PATHS_enth(aface, channel)*1000
                    rho=PATHS_dens(aface, channel)
                    x=PATHS_qual(aface, channel)
                ELSE
                    h=0.5*(PATHS_enth(aface, channel)+PATHS_enth(aface+1, channel))
                    rho=0.5*(PATHS_dens(aface, channel)+PATHS_dens(aface+1, channel))
                    x=0.5*(PATHS_qual(aface, channel)+PATHS_qual(aface+1, channel))
                ENDIF
                rho=(rho-x*rhog(pres))/(1-x)
                h=(h-x*hg(pres)*1000)/(1-x)
                !IF (x<=0.05) THEN
                    u=h-pres/rho
                    CALL clearprops(getprops)
                    getprops(tempbar)= .TRUE.
                    getprops(vsubf)=.TRUE.
                    CALL getsnglatpu(tables, liquid, getprops, u, pres, s, err, iterate)
                    rhol=1/s(vsubf) ! NOT GETTING RIGHT density
                    t=s(tempbar)        
                    kf=ConductivityLiq(pres, t, rhol, iLastLiqCond)/1E3
                !ELSE
                !    CALL getsatatp(tables, pres, s, err)
                !    rhol=1/s(vsubf)
                !    t=s(tsatbar)
                !    kf=ConductivityLiq(pres, t, rhol, iLastLiqCond)/1E3
                !ENDIF
            ELSE
                IF (pres .LT. 12E6) THEN
                    pr=pres/10**6
                    kf=1.4209E-8*pr**3-9.1659E-8*pr**2-1.6571E-5*pr+6.8750E-4
                ELSE
                    ! Liquid water
                    pr=pres/1.0e6
                    h=PATHS_enth(aface, channel)
                    
                    presB = 12.0
                    presT = 16.0
                    
                    kfB = 5.2330E-13*h**4 - 2.4311E-09*h**3 &
                        + 3.9765E-06*h**2 - 2.9513E-03*h + 1.5323E+00
                    kfT = 5.0626E-13*h**4 - 2.3610E-09*h**3 &
                        + 3.8783E-06*h**2 - 2.8920E-03*h + 1.5224E+00

                    slp = (kfT - kfB) / (presT - presB)
                    brhs = kfB - slp * presB
                    kf = pr * slp + brhs
                END IF                
            ENDIF
   
            RETURN
      END FUNCTION kf

      FUNCTION kg(pres)
            USE EosIAPWSConductivity, ONLY: ConductivityVap
            
            IMPLICIT NONE
            ! cubic poly for vapor thermal conductivity as a func of pressure
            REAL(sdk) :: kg, pres, pr
            REAL(sdk) :: presB, presT, slp, brhs
            REAL(sdk) :: kgB, kgT               
            
            IF (stable) THEN
                kg=ConductivityVap(pres, iLastVapCond)/1E3
            ELSE
                IF (pres .LT. 12E6) THEN
                    pr=pres/10**6
                    kg=2.1428E-8*pr**3-3.3468E-7*pr**2+5.5393E-6*pr+3.3190E-5
                ELSE
                    ! Assuming vapor fraction should be zero, but data is correct
                    pr=pres/1.0e6
                    
                    presB = 12.0
                    presT = 16.0
                    
                    kgB = 0.088605  ! 597.83 Kelvin trips to steam
                    kgT = 0.12804  ! 620.05 Kelvin trips to steam

                    slp = (kgT - kgB) / (presT - presB)
                    brhs = kgB - slp * presB
                    kg = pr * slp + brhs                    
                END IF                 
            ENDIF

            RETURN
      END FUNCTION kg

!      FUNCTION temp(pres)
!            IMPLICIT NONE
!            ! cubic poly for saturated temperature as a func of pressure
!            REAL(sdk) :: temp, pres, pr
!            REAL(sdk) :: presB, presT, slp, brhs
!            REAL(sdk) :: tB, tT               
!            
!            IF (stable) THEN
!                CALL gettsat(tables, pres, s, err)
!                temp=s(tsatbar)
!            ELSE
!                IF (pres .LT. 1.2e8) THEN
!                    pr=pres/10**6
!                    temp=3.3333E-2*pr**3-1.2300*pr**2+2.2001E1*pr+1.8066E2
!                ELSE
!                    ! Liquid water
!                    pr=pres/1.0e6
!                    h=PATHS_enth(aface, channel)
!                    
!                    presB = 12.0
!                    presT = 16.0
!
!                    tB = -7.1517E-11*h**4 + 3.0202E-07*h**3 - 5.2648E-04*h**2 &
!                          + 6.5141E-01*h + 1.4931E+02                    
!                    tT = -4.6250E-11*h**4 + 1.8189E-07*h**3 - 3.0746E-04*h**2 &
!                          + 4.7276E-01*h + 2.0368E+02
!
!                    slp = (tT - tB) / (presT - presB)
!                    brhs = tB - slp * presB
!                    temp = pr * slp + brhs                  
!                END IF    
!            ENDIF
!
!            RETURN
!      END FUNCTION temp

      FUNCTION surften(pres)
            USE Eos, ONLY: Sigma
            
            IMPLICIT NONE
            ! cubic poly for saturated surface tension as a func of pressure
            REAL(sdk) :: surften, pres, pr, t
            
            IF (stable) THEN
                CALL gettsat(tables, pres, s, err)
                t=s(tsatbar)
                surften=Sigma(t)
            ELSE
                IF (pres .LT. 12e6) THEN
                    pr=pres/10**6
                    surften=-7.2391E-6*pr**3+2.8345E-4*pr**2-5.1566E-3*pr+4.2324E-2
                ELSE    
                    pr=pres/10**6
                    surften = -1.2409E-06*pr**3 + 1.0224E-04*pr**2 - 3.2966E-03*pr + 3.5827E-02
                END IF
            ENDIF
            
            RETURN
      END FUNCTION surften

      FUNCTION tfh(P, h, aface, channel, edge)

            ! Calculates subcool liquid temperature [C] from pressure [Pa] and specific enthalpy [kJ/kg].
            ! Method from IAPWS, Revised Release on 1997 Industrial Formulation for the Thermodynamic
            !     Properties of Water and Steam. pg. 9-11
            !
            ! Range of validity:
            !           273.15 K .le. T .le. 623.15 K
            !           P_sat(T) .le. P .le. 100 MPa

            IMPLICIT NONE

            REAL(sdk) :: P,h,tfh, rho, u, x
            INTEGER(sik) :: I(20), J(20), m
            REAL(sdk) ::  n(20), eta, pi
            INTEGER(sik) :: channel, aface
            LOGICAL :: edge
            DATA I/6*0,7*1,2,2,3,3,4,5,6/
            DATA J/0,1,2,6,22,32,0,1,2,3,4,10,32,10,32,10,32,32,32,32/
            DATA n/-0.23872489924521D+03,  0.40421188637945D+03,  0.11349746881718D+03     &
                 , -0.58457616048039D+01, -0.15285482413140D-03, -0.10866707695377D-05    &
                 , -0.13391744872602D+02,  0.43211039183559D+02, -0.54010067170506D+02    &
                 ,  0.30535892203916D+02, -0.65964749423638D+01,  0.93965400878363D-02    &
                 ,  0.11573647505340D-06, -0.25858641282073D-04, -0.40644363084799D-08    &
                 ,  0.66456186191635D-07,  0.80670734103027D-10, -0.93477771213947D-12    &
                 ,  0.58265442020601D-14, -0.15020185953503D-16/
    
            !            I( 1: 6)=0
            !            I( 7:13)=1
            !            I(14:15)=2
            !            I(16:20)=[   3,   3,   4,   5,   6]
            !            J( 1: 5)=[   0,   1,   2,   6,  22]
            !            J( 6:10)=[  32,   0,   1,   2,   3]
            !            J(11:15)=[   4,  10,  32,  10,  32]
            !            J(16:20)=[  10,  32,  32,  32,  32]
            !            n( 1: 3)=[ -0.23872489924521D+03,  0.40421188637945D+03,  0.11349746881718D+03]
            !            n( 4: 6)=[ -0.58457616048039D+01, -0.15285482413140D-03, -0.10866707695377D-05]
            !            n( 7: 9)=[ -0.13391744872602D+02,  0.43211039183559D+02, -0.54010067170506D+02]
            !            n(10:12)=[  0.30535892203916D+02, -0.65964749423638D+01,  0.93965400878363D-02]
            !            n(13:15)=[  0.11573647505340D-06, -0.25858641282073D-04, -0.40644363084799D-08]
            !            n(16:18)=[  0.66456186191635D-07,  0.80670734103027D-10, -0.93477771213947D-12]
            !            n(19:20)=[  0.58265442020601D-14, -0.15020185953503D-16]

            IF (stable .AND. channel>0) THEN 
                IF (edge) THEN
                    rho=PATHS_dens(aface, channel)
                    x=PATHS_qual(aface, channel)
                ELSE
                    rho=0.5*(PATHS_dens(aface, channel)+PATHS_dens(aface+1, channel))
                    x=0.5*(PATHS_qual(aface, channel)+PATHS_qual(aface+1, channel))
                ENDIF
                rho=(rho-x*rhog(P))/(1-x)
                h=h*1000
                !IF (x<=0.05) THEN
                    u=h-P/rho
                    CALL clearprops(getprops)
                    getprops(tempbar)= .TRUE.
                    CALL getsnglatpu(tables, liquid, getprops, u, P, s, err, iterate)
                    tfh=s(tempbar)-273.15
                !ELSE
                !    CALL gettsat(tables, P, s, err)
                !    tfh=s(tsatbar)-273.15
                !ENDIF
                
            ELSE
                            
                pi=P/1.0E6
                eta=h/2500.0

                tfh=0

                DO m=1,20
                   tfh=tfh+n(m)*(pi**I(m))*((eta+1)**J(m))
                END DO

                tfh=tfh-273.15
            ENDIF

            RETURN

      END FUNCTION tfh

      FUNCTION ps(temp)

            ! Calculates saturation pressure [MPa] from temperature [K].
            ! Method from IAPWS, Revised Release on 1997 Industrial Formulation for the Thermodynamic
            !     Properties of Water and Steam. pg. 32-35
            !
            ! Range of validity:
            !           273.15 .le. T .le. 647.096 K
            USE EosIAPWSData
            USE EosIAPWSCrunch

            IMPLICIT NONE

            LOGICAL::err
            REAL(sdk) :: ps, T, temp, n(10), thet, A, B, C
            DATA n/  0.11670521452767D+04, -0.72421316703206D+06, -0.17073846940092D+02 &
                 ,  0.12020824702470D+05, -0.32325550322333D+07,  0.14915108613530D+02 &
                 , -0.48232657361591D+04,  0.40511340542057D+06, -0.23855557567849D+00 &
                 ,  0.65017534844798D+03/

            T=temp+273.15
            err=.FALSE.
            

            !            n( 1: 3)=[  0.11670521452767D+04, -0.72421316703206D+06, -0.17073846940092D+02]
            !            n( 4: 6)=[  0.12020824702470D+05, -0.32325550322333D+07,  0.14915108613530D+02]
            !            n( 7: 9)=[ -0.48232657361591D+04,  0.40511340542057D+06, -0.23855557567849D+00]
            !            n(10)= 0.65017534844798D+03

            IF(T<273.15) THEN
!               PRINT*,'Temperature out of bounds.  Changing T to lower bound of 274.15. T=',T
               T=274.15
            ENDIF
            IF(T>647.096) THEN
!               PRINT*,'Temperature out of bounds.  Changing T to upper bound of 646.096. T=',T
               T=646.096
            ENDIF
            
            IF(stable) THEN
                CALL getpsat(tables, T, s, err)
                ps=s(psatbar)
            ELSE
                
            thet=T+n(9)/(T-n(10))

            A=thet**2+n(1)*thet+n(2)
            B=n(3)*thet**2+n(4)*thet+n(5)
            C=n(6)*thet**2+n(7)*thet+n(8)

            ps=(2*C/(-B+((B**2-4*A*C)**(0.5))))**4

            ps=ps*1.0E6
            ENDIF

            RETURN

      END FUNCTION ps

      FUNCTION ts(pres)

            ! Calculates saturation temperature [K] from pressure [MPa].
            ! Method from IAPWS, Revised Release on 1997 Industrial Formulation for the Thermodynamic
            !     Properties of Water and Steam. pg. 32-35
            !
            ! Range of validity:
            !           611.213 Pa .le. P .le. 22.064 MPa
            USE EosIAPWSData, ONLY: tables, s, tsatbar
            USE EosIAPWSCrunch, ONLY: gettsat
            
            IMPLICIT NONE

            LOGICAL::err
            REAL(sdk) :: ts, P, pres, n(10), beta
            REAL(sdk) ::  D, E, F, G
            DATA n/  0.11670521452767D+04, -0.72421316703206D+06, -0.17073846940092D+02 &
                 ,  0.12020824702470D+05, -0.32325550322333D+07,  0.14915108613530D+02 &
                 , -0.48232657361591D+04,  0.40511340542057D+06, -0.23855557567849D+00 &
                 ,  0.65017534844798D+03/
            
            IF (stable) THEN
                CALL gettsat(tables, pres, s, err)
                ts=s(tsatbar)-273
            ELSE
                P=pres/1.0E6
                err=.FALSE.

!                n( 1: 3)=[  0.11670521452767D+04, -0.72421316703206D+06, -0.17073846940092D+02]
!                n( 4: 6)=[  0.12020824702470D+05, -0.32325550322333D+07,  0.14915108613530D+02]
!                n( 7: 9)=[ -0.48232657361591D+04,  0.40511340542057D+06, -0.23855557567849D+00]
!                n(10)= 0.65017534844798D+03
        
                beta=P**(0.25)

                E=beta**2+n(3)*beta+n(6)
                F=n(1)*beta**2+n(4)*beta+n(7)
                G=n(2)*beta**2+n(5)*beta+n(8)
                D=2*G/(-F-(F**2-4*E*G)**(0.5))

                ts=(n(10)+D-((n(10)+D)**2-4*(n(9)+n(10)*D))**(0.5))/2
                
                ts=ts-273.15
            ENDIF

            RETURN

      END FUNCTION ts

END MODULE PATHS_WaterProp
