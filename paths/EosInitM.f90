MODULE EosInit
!
!     (c) 2002 The Pennsylvania University Applied Research Laboratory
!     This Software was developed for Purdue Subcontract Agreement 640-01812-2
!     under NRC Prime Contract no. NRC-04-97-046 (Task #2)
!
!
!     BEGIN MODULE USE
      USE IntrType, ONLY: sdk, sik
!
      IMPLICIT NONE
!
!   Initialization for Equation of State (EOS) Module.
!
!      PRIVATE :: SetEoH, SetEoD, SetEoHe, SetEoNa
!      PRIVATE :: SetEoPbBi, SetEoIAPWS, SetEoN2, SetEoAir, SetEoD2oST
CONTAINS
!
      SUBROUTINE SetEos
!
         USE EosData
         USE EosIAPWSData, ONLY: SetCritPointIAPWS, tcrit, pcrit, vcrit
         !USE EosNCGData, ONLY: InitNCGases, ncGasSpecies
         !USE TraceSpeciesData, ONLY: InitTraceNCGases
!
         IMPLICIT NONE
!
!   Driver to initialize the ceoslp array for the available EOSs.
!
         INTEGER(sik) :: i

!        Initialize the non-condensible gas model.
         !CALL InitNCGases(iGas)
         !CALL InitTraceNCGases(ncGasSpecies)
!
!   Some H2O properties constants are shared by other fluids
!   so SetEoH is always called.
!
         CALL SetEoH
         CALL SetCritPointIAPWS(tcrit, pcrit, vcrit)
!
!        Loop over the fluids in this model to initialize EOS for those fluids.
         DO i = 1, nActEos
            SELECT CASE (actFluids(i))
               CASE (eosH2o)
!
!                 TRAC-P fits will use IAPWS fits for conductivity and viscosity,
!                 therefore ceoslp for steam tables must be set.
                  IF (use_IAPWS_st) THEN
                     CALL SetEoIAPWS
                  END IF
!               CASE (eosD2o)
!!
!!                 Heavy water will use IAPWS fits for conductivity, therefore ceoslp
!!                 for steam tables must be set.
!                  CALL SetEoIAPWS
!                  IF (use_D2O_st) THEN
!                     CALL SetEoD2oST()
!                  ELSE
!                     CALL SetEoD()
!                  END IF
!               CASE (eosHe)
!                  CALL SetEoHe()
!               CASE (eosNa)
!                  CALL SetEoNa()
!               CASE (eosPbBi)
!                  CALL SetEoPbBi()
!               CASE (eosN2)
!                  CALL SetEoN2()
!               CASE (eosAir)
!                  CALL SetEoAir()
               CASE DEFAULT
                  !CALL error(1, '*seteos* Bad eos flag.')
                  WRITE(*,*) '*seteos* Bad eos flag.'
            END SELECT
         END DO
!
      END SUBROUTINE SetEos
!
      SUBROUTINE MinMaxPT
!
!     This routine determines the minimum and maximum pressure,
!     liquid temperature, and vapor temperature for all fluids
!     in this model.
!
         USE EosData
!
         IMPLICIT NONE
!
         INTEGER(sik) :: i
!
         DO i = 1, nActEos
            minP = MIN(minP, ceoslp(30, actFluids(i)))
            maxP = MAX(maxP, ceoslp(31, actFluids(i)))
            minTL = MIN(minTL, ceoslp(32, actFluids(i)))
            maxTL = MAX(maxTL, ceoslp(33, actFluids(i)))
            minTV = MIN(minTV, ceoslp(34, actFluids(i)))
            maxTV = MAX(maxTV, ceoslp(35, actFluids(i)))
         END DO
!
      END SUBROUTINE MinMaxPT
!
     SUBROUTINE SetEoH
!
         !USE Eos, ONLY: HeV
         USE EosData
         !USE EosNCGData, ONLY: SetCeoslpNCG
         !USE GlobalDat, ONLY: gasCon
!
         IMPLICIT NONE
!
         REAL(sdk) vapmol
!
!
!     subroutine SetEoH initializes the h2o equation-of-state constants
!
!     aeos14     = reciprocal reference pressure = 1.0/ceoslp(11)
!     ceos1      = 1st coeff of saturation curve
!     ceos2      = 2nd coeff of saturation curve
!     ceos3      = 3rd coeff of saturation curve
!     vapmol     = molecular weight of vapor
!
         vapmol = h2oMWt
         aeos14(eosH2o) = 1.0d-05
         ceos1(eosH2o) = 117.8d0
         ceos2(eosH2o) = 0.223d0
         ceos3(eosH2o) = 255.2d0
         ceoslp(1, eosH2o) = - 2263.0d0
         ceoslp(2, eosH2o) = 0.434d0
         ceoslp(3, eosH2o) = - 6.064d0
!
         !CALL SetCeoslpNCG(iGas, eosH2o)
!c    ceoslp(28) = (ceoslp(12)-ceoslp(25))/ceoslp(12)
!
         ceoslp(5, eosH2o) = 273.15d0
         ceoslp(29, eosH2o) = 273.16d0
         ceoslp(30, eosH2o) = 1.0d0
         ceoslp(31, eosH2o) = 450.0d5
         ceoslp(32, eosH2o) = 273.15d0
         ceoslp(33, eosH2o) = 713.94025779311d0
         ceoslp(34, eosH2o) = 273.15d0
         ceoslp(35, eosH2o) = 3000.0d0
         ceoslp(36, eosH2o) = 610.8d0
         ceoslp(37, eosH2o) = 221.2d5
         ceoslp(38, eosH2o) = 647.3d0
         ceoslp(39, eosH2o) = 139.69971285053d5
         ceoslp(40, eosH2o) = 609.62462615967d0
         ceoslp(12, eosH2o) = gasCon / vapmol
         ceoslp(16, eosH2o) = 1.3d0
         ceoslp(4, eosH2o) = ceoslp(12, eosH2o) / (ceoslp(16, eosH2o)-1.0d0)
         ceoslp(23, eosH2o) = ceoslp(16, eosH2o) * ceoslp(4, eosH2o)
!c    ceoslp(28) = (ceoslp(12)-ceoslp(25))/ceoslp(12)
         ceoslp(24, eosH2o) = 4186.800d0
         ceoslp(14, eosH2o) = 0.65141d0
         ceoslp(7, eosH2o) = ceoslp(24, eosH2o)
         ceoslp(9, eosH2o) = 990.0d0
         ceoslp(15, eosH2o) = 0.0d0
         ceoslp(20, eosH2o) = 9.056466d4
         ceoslp(21, eosH2o) = 370.4251d0
         ceoslp(11, eosH2o) = 100000.0d0
!         ceoslp(10, eosH2o) = HeV(ceoslp(5, eosH2o), eosH2o)
         ceoslp(8, eosH2o) = - 611.2d0 * 0.0010002d0 + ceoslp(7, eosH2o) * (ceoslp(5, eosH2o)-ceoslp(29, &
        & eosH2o))
         ceoslp(26, eosH2o) = ceoslp(24, eosH2o) * (ceoslp(5, eosH2o)-ceoslp(29, eosH2o))
!         ceoslp(27, eosH2o) = ceoslp(26, eosH2o) + ceoslp(10, eosH2o)
         ceoslp(6, eosH2o) = ceoslp(27, eosH2o) - ceoslp(12, eosH2o) * ceoslp(5, eosH2o)
!
     END SUBROUTINE SetEoH
!     
      SUBROUTINE SetEoIAPWS
!
         USE EosData
         !USE Eos, ONLY: HeV
         !USE EosNCGData, ONLY: SetCeoslpNCG
         USE EosIAPWSData, ONLY: ttrip, ptrip, tcrit, pcrit, pmax, tmax, s, getprops, tables
         USE EosIAPWSData, ONLY: usubf, hsubf, tmin, pmin
         USE EosIAPWSCrunch, ONLY: getsatatp
!
         IMPLICIT NONE
!
!     subroutine SetEoIAPWS initializes the h2o equation-of-state constants
!     SetEoH must be called before SetEoIAPWS is called.
!
         LOGICAL :: err
!
         CHARACTER(LEN=120) :: ErrString
!
!         CALL SetCeoslpNCG(iGas, eosH2o)
!
!  reset the limits for the new tables
!
         ceoslp(5, eosH2o) = ttrip
         ceoslp(29, eosH2o) = ttrip
         ceoslp(30, eosH2o) = pmin
         ceoslp(31, eosH2o) = pmax
         ceoslp(32, eosH2o) = tmin
         ceoslp(33, eosH2o) = tmax
         ceoslp(34, eosH2o) = tmin
         ceoslp(35, eosH2o) = tmax
         ceoslp(36, eosH2o) = ptrip
         ceoslp(37, eosH2o) = pcrit
         ceoslp(38, eosH2o) = tcrit
!
!         ceoslp(10, eosH2o) = HeV(ceoslp(5, eosH2o), eosH2o)
         ceoslp(8, eosH2o) = - 611.2_sdk * 0.0010002_sdk + ceoslp(7, eosH2o) * (ceoslp(5, &
        & eosH2o)-ceoslp(29, eosH2o))
         ceoslp(26, eosH2o) = ceoslp(24, eosH2o) * (ceoslp(5, eosH2o)-ceoslp(29, eosH2o))
!         ceoslp(27, eosH2o) = ceoslp(26, eosH2o) + ceoslp(10, eosH2o)
         ceoslp(6, eosH2o) = ceoslp(27, eosH2o) - ceoslp(12, eosH2o) * ceoslp(5, eosH2o)
!
!  get two constants that are fixed
!
         CALL getsatatp(tables, ptrip, s, err)
!
         IF (err) THEN
            WRITE (ErrString, 814) ptrip
814         FORMAT ('*SetEoIAPWS* Error from getsatatp for liquid: ptrip = ', 1 p, e13.4)
            !CALL error(4, ErrString)
         END IF
!
         ceoslp(8, eosH2o) = s(usubf)
         ceoslp(26, eosH2o) = s(hsubf)
!
      END SUBROUTINE SetEoIAPWS

END MODULE EosInit