MODULE EosIAPWSConductivity
!
!     (c) 2002 The Pennsylvania University Applied Research Laboratory
!     This Software was developed for Purdue Subcontract Agreement 640-01812-2
!     under NCR Prime Contract no. NRC-04-97-046 (Task #2)
!
!  Purpose: This module contains functions to calculate the IAPWS thermal conductivity of water and steam.
!
!  Programmed by Jay Spore, ISL, Date (11/07)
!
   USE IntrType, ONLY: sdk, sik
!
   IMPLICIT NONE
!
!  Reference: The International Association for the Properties of Water and Steam, London, England, September 1998.
!             Revised Release on the IAPS Formulation 1985 for the Thermal Conductivity of Ordinary Water Substance.
!
!  Number of points in the saturated thermal conductivity table.
   INTEGER(sik), PARAMETER, PRIVATE :: nElements = 41
   REAL(sdk),PRIVATE,PARAMETER :: one=1.0_sdk,zero=0.0_sdk
!
! pSatCond (Pa) sat pressure for saturation thermal conductivity liquid and vapor phase.
   REAL(sdk), PARAMETER, DIMENSION(nElements), PRIVATE :: pSatCond = &
  &(/0.0006117e+06_sdk, 0.001228e+06_sdk, 0.002339e+06_sdk, 0.004247e+06_sdk, 0.007385e+06_sdk, &
  &  0.01235e+06_sdk,   0.01995e+06_sdk,  0.0312e+06_sdk,   0.04741e+06_sdk,  0.07018e+06_sdk, &
  &  0.1014e+06_sdk,    0.1434e+06_sdk,   0.1987e+06_sdk,   0.2703e+06_sdk,   0.3615e+06_sdk, &
  &  0.4762e+06_sdk,    0.6182e+06_sdk ,  0.7922e+06_sdk,   1.003e+06_sdk,    1.255e+06_sdk, &
  &  1.555e+06_sdk,     1.908e+06_sdk,    2.32e+06_sdk,     2.797e+06_sdk,    3.347e+06_sdk, &
  &  3.976e+06_sdk,     4.692e+06_sdk,    5.503e+06_sdk,    6.417e+06_sdk,    7.442e+06_sdk, &
  &  8.588e+06_sdk ,    9.865e+06_sdk ,  11.284e+06_sdk,   12.858e+06_sdk,   14.601e+06_sdk, &
  & 16.529e+06_sdk,    18.666e+06_sdk,   21.044e+06_sdk,   21.297e+06_sdk,   21.554e+06_sdk, &
  & 21.814e+06_sdk/)
!
   PRIVATE :: CondLiqSat, CondVapSat, TSatOfPCond, ConductivityIAPWS, FindIntervalPSat
!
CONTAINS
!
!
   REAL(sdk) FUNCTION CondLiqSat(p, iLast)
!
      IMPLICIT NONE
!
!  Purpose: This function returns thermal conductivity of saturated liquid water at pressure p.
!
!  Programmed by Jay Spore, ISL, Date (11/07)
!
!  Subroutine Argument Descriptions:
!
!  Input:
!      Variable Name   Variable Desription
!                  p   Pressure.
!              iLast   Last index used in the saturation table.
!
!  Output:
!      Variable Name   Variable Desription
!         CondLiqSat   Thermal conductivity of the saturated liquid phase.
!              iLast   Current index used in the saturation table.
!
!  Variable Declarations:
!
      INTEGER(sik), INTENT(INOUT) :: iLast
      INTEGER(sik) :: i
!
      REAL(sdk), INTENT(IN) :: p
      REAL(sdk) :: pLocal
!
      LOGICAL :: found
!
! condfSat (W/m/K) saturation liquid phase thermal conductivity.
      REAL(sdk), PARAMETER, DIMENSION(nElements) :: condfSat = &
     &        (/ 565.0e-03_sdk ,  584.0e-03_sdk ,  602.0e-03_sdk , 617.0e-03_sdk , 631.0e-03_sdk, &
     &           642.0e-03_sdk ,  653.0e-03_sdk ,  660.0e-03_sdk , 669.0e-03_sdk , 675.0e-03_sdk, &
     &           679.0e-03_sdk ,  681.0e-03_sdk ,  685.0e-03_sdk , 686.0e-03_sdk , 686.0e-03_sdk, &
     &           686.0e-03_sdk ,  682.0e-03_sdk ,  678.0e-03_sdk , 674.0e-03_sdk , 670.0e-03_sdk, &
     &           664.0e-03_sdk ,  654.0e-03_sdk ,  643.0e-03_sdk , 632.0e-03_sdk , 626.0e-03_sdk, &
     &           615.0e-03_sdk ,  602.0e-03_sdk ,  590.0e-03_sdk , 577.0e-03_sdk,  564.0e-03_sdk, &
     &           547.0e-03_sdk ,  532.0e-03_sdk,   512.0e-03_sdk,  485.0e-03_sdk,  455.0e-03_sdk, &
     &           447.0e-03_sdk,   425.0e-03_sdk,   418.0e-03_sdk,  429.0e-03_sdk,  450.0e-03_sdk, &
     &           520.0e-03_sdk/)
!
!     Limit the pressure to range in the saturation table.
      pLocal = MAX(MIN(p, pSatCond(nElements)), pSatCond(1))
!
!     Find the two points that bracket pLocal
      i = FindIntervalPSat(pLocal, iLast, found)
      IF (found) THEN
!
!        Interpolation in table
         CondLiqSat = condfSat(i) + (condfSat(i+1) - condfSat(i)) * (pLocal - pSatCond(i)) / &
        & (pSatCond(i+1) - pSatCond(i))
      ELSE
         !CALL error(1, ' *CondLiqSat* FindIntervalPSat failed.')
      END IF
!
   END FUNCTION CondLiqSat
!
!
   REAL(sdk) FUNCTION CondVapSat(p, iLast)
!
      IMPLICIT NONE
!
!  Purpose: This function returns thermal conductivity of saturated steam at pressure p.
!
!  Programmed by Jay Spore, ISL, Date (11/07)
!
!  Subroutine Argument Descriptions:
!
!  Input:
!      Variable Name   Variable Desription
!                  p   Pressure.
!              iLast   Last index used in the saturation table.
!
!  Output:
!      Variable Name   Variable Desription
!         CondVapSat   Thermal conductivity of the saturated vapor phase.
!              iLast   Current index used in the saturation table.
!
!  Variable Declarations:
!
      INTEGER(sik), INTENT(INOUT) :: iLast
      INTEGER(sik) :: i
!
      REAL(sdk), INTENT(IN) :: p
      REAL(sdk) :: pLocal
!
      LOGICAL :: found
!
! condgSat (W/m/K) saturation vapor phase thermal conductivity.
      REAL(sdk), PARAMETER, DIMENSION(nElements) :: condgSat = &
     &          (/16.7e-03_sdk,  17.4e-03_sdk,  18.1e-03_sdk,  19.0e-03_sdk,  19.7e-03_sdk, &
     &            20.4e-03_sdk,  21.2e-03_sdk,  22.2e-03_sdk,  23.1e-03_sdk,  24.0e-03_sdk, &
     &            25.0e-03_sdk,  25.7e-03_sdk,  26.8e-03_sdk,  28.7e-03_sdk , 29.7e-03_sdk, &
     &            31.0e-03_sdk,  31.9e-03_sdk,  33.6e-03_sdk,  35.2e-03_sdk,  37.2e-03_sdk, &
     &            38.8e-03_sdk,  40.5e-03_sdk,  43.2e-03_sdk,  45.3e-03_sdk,  47.9e-03_sdk, &
     &            51.0e-03_sdk,  54.2e-03_sdk,  57.7e-03_sdk,  61.3e-03_sdk , 67.3e-03_sdk, &
     &            73.2e-03_sdk,  79.8e-03_sdk,  88.3e-03_sdk,  99.1e-03_sdk, 116.7e-03_sdk, &
     &           138.0e-03_sdk, 174.0e-03_sdk, 293.0e-03_sdk, 331.0e-03_sdk, 377.0e-03_sdk, &
     &           464.0e-03_sdk/)
!
!     Limit the pressure to range in the saturation table.
      pLocal = MAX(MIN(p, pSatCond(nElements)), pSatCond(1))
!
!     Find the two points that bracket pLocal.
      i = FindIntervalPSat(pLocal, iLast, found)
      IF (found) THEN
!
!        Interpolation in table
         CondVapSat = condgSat(i) + (condgSat(i+1) - condgSat(i)) * (pLocal - pSatCond(i)) / &
        & (pSatCond(i+1) - pSatCond(i))
      ELSE
         !CALL error(1, ' *CondVapSat* FindIntervalPSat failed.')
      END IF
!
   END FUNCTION CondVapSat
!
!
   REAL(sdk) FUNCTION TSatOfPCond(p, iLast)
!
      IMPLICIT NONE
!
!  Purpose: This function returns tSat given pSat.
!
!  Programmed by Jay Spore, ISL, Date (11/07)
!
!  Subroutine Argument Descriptions:
!
!  Input:
!      Variable Name   Variable Desription
!                  p   Saturation pressure.
!              iLast   Last index used in the saturation table.
!
!  Output:
!      Variable Name   Variable Desription
!        TSatOfPCond   tSat given pSat.
!              iLast   Current index used in the saturation table.
!
!  Variable Declarations:
!
      INTEGER(sik), INTENT(INOUT) :: iLast
      INTEGER(sik) :: i
!
      REAL(sdk), INTENT(IN) :: p
      REAL(sdk) :: pLocal
!
      LOGICAL :: found
!
! tSatCond (C) sat temperature for saturation thermal conductivity liquid and vapor phase.
      REAL(sdk), PARAMETER, DIMENSION(nElements) :: tSatCond = &
     &  (/0.01_sdk, 10.0_sdk,  20.0_sdk,  30.0_sdk,  40.0_sdk, &
     &   50.0_sdk,  60.0_sdk,  70.0_sdk,  80.0_sdk,  90.0_sdk, &
     &  100.0_sdk, 110.0_sdk, 120.0_sdk, 130.0_sdk, 140.0_sdk, &
     &  150.0_sdk, 160.0_sdk, 170.0_sdk, 180.0_sdk, 190.0_sdk, &
     &  200.0_sdk, 210.0_sdk, 220.0_sdk, 230.0_sdk, 240.0_sdk, &
     &  250.0_sdk, 260.0_sdk, 270.0_sdk, 280.0_sdk, 290.0_sdk, &
     &  300.0_sdk, 310.0_sdk, 320.0_sdk, 330.0_sdk, 340.0_sdk, &
     &  350.0_sdk, 360.0_sdk, 370.0_sdk, 371.0_sdk, 372.0_sdk, &
     &  373.0_sdk/)
!
!     Limit the pressure to range in the saturation table.
      pLocal = MAX(MIN(p, pSatCond(nElements)), pSatCond(1))
!
!     Find the two points that bracket pLocal
      i = FindIntervalPSat(pLocal, iLast, found)
      IF (found) THEN
!
!        Interpolation in table
         TSatOfPCond = tSatCond(i) + (tSatCond(i+1) - tSatCond(i)) * (pLocal - pSatCond(i)) / &
        & (pSatCond(i+1) - pSatCond(i)) + 273.15_sdk
      ELSE
         !CALL error(1, ' *TsatOfPCond* FindIntervalPSat failed.')
      END IF
!
   END FUNCTION TSatOfPCond
!
!
   REAL(sdk) FUNCTION ConductivityVap(p, iLast) ! t, rho, iLast)
!
      !USE EosIAPWSData, ONLY: tCrit, pCrit
!
      IMPLICIT NONE
!
!  Purpose: This function returns water vapor thermal conductivity based on IAPWS standard.
!
!  Programmed by Jay Spore, ISL, Date (11/07)
!
!  Subroutine Argument Descriptions:
!
!  Input:
!      Variable Name   Variable Desription
!                  p   Pressure.
!                  t   Temperature.
!                rho   Density.
!              iLast   Last index used in the saturation table.
!
!  Output:
!      Variable Name   Variable Desription
!    ConductivityVap   Vapor thermal conductivity at p, t, and rho.
!              iLast   Current index used in the saturation table.
!
!  Variable Declarations:
!
      INTEGER(sik), INTENT(INOUT) :: iLast
!
      REAL(sdk), INTENT(IN) :: p !, t, rho
!
!!     Check to see if pressure is below the critical pressure.
!      IF (p < pCrit .AND. t < tCrit) THEN
!!
!!        Pressure is below the critical pressure.
!!        Check to see if vapor is saturated or subcooled.
!         IF (t <= TSatOfPCond(p, iLast)) THEN
!!
!!           Vapor is saturated or subcooled, so use saturated vapor thermal conductivity.
!            ConductivityVap = CondVapSat(p, iLast)
!         ELSE
!!
!!           Vapor is super-heated use IAPWS fit.
!            ConductivityVap = ConductivityIAPWS(rho, p, t)
!         END IF
!      ELSE
!!
!!        Vapor is super-critical use IAPWS fit.
!         ConductivityVap = ConductivityIAPWS(rho, p, t)
!      END IF
      ConductivityVap = CondVapSat(p, iLast)
!
   END FUNCTION ConductivityVap
!
!
   REAL(sdk) FUNCTION ConductivityLiq(p, t, rho, iLast)
!
      USE EosIAPWSData, ONLY: tCrit, pCrit
!
      IMPLICIT NONE
!
!  Purpose: This function returns water liquid thermal conductivity based on IAPWS standard.
!
!  Programmed by Jay Spore, ISL, Date (11/07)
!
!  Subroutine Argument Descriptions:
!
!  Input:
!      Variable Name   Variable Desription
!                  p   Pressure.
!                  t   Temperature.
!                rho   Density.
!              iLast   Last index used in the saturation table.
!
!  Output:
!      Variable Name   Variable Desription
!    ConductivityLiq   Liquid thermal conductivity at p, t, and rho.
!              iLast   Current index used in the saturation table.
!
!  Variable Declarations:
!
      INTEGER(sik), INTENT(INOUT) :: iLast
!
      REAL(sdk), INTENT(IN) :: p, t, rho
!
!     Check if pressure and temperature are below the critical point.
      IF (p < pCrit .AND. t < tCrit) THEN
!
!        Check to see if liquid is saturated or super-heated.
         IF (t >= TSatOfPCond(p, iLast)) THEN
!
!           Liquid is saturated or super-heated so use saturated liquid thermal conductivity.
            ConductivityLiq = CondLiqSat(p, iLast)
         ELSE IF (t <= 273.16_sdk) THEN
!
!           Below triple point use saturated phase thermal conductivity based on pressure.
            ConductivityLiq = CondLiqSat(p, iLast)
         ELSE
!
!           Liquid is subcooled use IAPWS fit for liquid phase thermal conductivity.
            ConductivityLiq = ConductivityIAPWS(rho, p, t)
         END IF
      ELSE
!
!        Liquid is super-critical use IAPWS fit for liquid phase thermal conductivity.
         ConductivityLiq = ConductivityIAPWS(rho, p, t)
      END IF
!
   END FUNCTION ConductivityLiq
!
!
   REAL(sdk) FUNCTION ConductivityIAPWS(rho, p, t)
!
!      USE GlobalDat, ONLY: one, zero
!
      IMPLICIT NONE
!
!  Purpose: This function returns water/steam thermal conductivity based on IAPWS fit.
!
!  Programmed by Jay Spore, ISL, Date (11/07)
!
!  Subroutine Argument Descriptions:
!
!  Input:
!      Variable Name   Variable Desription
!                  p   Presssure
!                  t   Temperature.
!                rho   Density.
!
!  Output:
!      Variable Name   Variable Desription
!  ConductivityIAPWS   Thermal conductivity at t and rho.
!
!  Variable Declarations:
!
      REAL(sdk), INTENT(IN) :: rho, t, p
      REAL(sdk) :: rhoBar, tBar, lamda0, lamda1, lamda2, s, q, deltaT, tMax
      REAL(sdk), PARAMETER :: tRef = 647.26_sdk              ! Reference temperature for fit.
      REAL(sdk), PARAMETER :: rhoRef = 317.7_sdk             ! Reference density for fit.
      REAL(sdk), PARAMETER :: tBarMin = 273.15_sdk / tRef    ! Lower limit for validity of fit.
      REAL(sdk), PARAMETER :: tMax1 = 500.0_sdk + 273.15_sdk ! Upper limit for validity of fit for P <= 100 MPa.
                                                             ! If tBar get's large enough lamda0 will go negative.
      REAL(sdk), PARAMETER :: tMax2 = 650.0_sdk + 273.15_sdk ! Upper limit for validity for fit for P <= 70 MPa
      REAL(sdk), PARAMETER :: tMax3 = 800.0_sdk + 273.15_sdk ! Upper limit for validity for fit for P <= 40 MPa
!
!     Fit coefficients.
      REAL(sdk), PARAMETER, DIMENSION(0:3) :: a = (/0.0102811_sdk, 0.0299621_sdk, 0.0156146_sdk, -0.00422464_sdk/)
      REAL(sdk), PARAMETER, DIMENSION(0:2) :: b = (/-0.39707_sdk, 0.400302_sdk, 1.06_sdk/)
      REAL(sdk), PARAMETER, DIMENSION(2) :: bb = (/-0.171587_sdk, 2.392190_sdk/)
      REAL(sdk), PARAMETER, DIMENSION(4) :: d = (/0.0701309_sdk, 0.011852_sdk, 0.00169937_sdk, -1.02_sdk/)
      REAL(sdk), PARAMETER, DIMENSION(6) :: c = (/0.642857_sdk, -4.11717_sdk, -6.17937_sdk, 0.00308976_sdk, &
     &                                            0.0822994_sdk, 10.0932_sdk/)
!
!     Constants in fits.
      REAL(sdk), PARAMETER :: nineFifths =  1.8_sdk
      REAL(sdk), PARAMETER :: fourteenFifths = 2.8_sdk
!
      CHARACTER(LEN=130) :: mess
!
!     Determine upper limit for fit.
      IF (p <= 40.0e+06_sdk) THEN
         tMax = tMax3
      ELSE IF (p <= 70.0e+06_sdk) THEN
         tMax = tMax2
      ELSE
         tMax = tMax1
      END IF
!
!     Calculate non-dimensional density and temperature.
      rhoBar = rho / rhoRef
      tBar = MAX(MIN(t, tMax) / tRef, tBarMin)
!
!     Ideal gas limit.
      lamda0 = SQRT(tBar) * (a(0) + tBar* (a(1) + tBar * (a(2) + tBar * a(3))))
!
!     Density function.
      lamda1 = b(0) + b(1) * rhoBar + b(2) * EXP(bb(1) * (rhoBar + bb(2))**2)
!
!     Density and temperature function.
      deltaT = ABS(tBar - one) + c(4)
      q = 2.0_sdk + c(5) / (deltaT**0.6_sdk)
      IF (tBar >= one) THEN
         s = one / deltaT
      ELSE
         s = c(6) / (deltaT**0.6_sdk)
      END IF
      lamda2 = (d(1) / tBar**10 + &
     &          d(2)) * rhoBar**nineFifths * EXP(c(1) * (one - rhoBar**fourteenFifths)) + &
     &          d(3) * s * (rhoBar**q) * EXP((q /(one + q)) * (one - rhoBar**(one + q))) + &
     &          d(4) * EXP(c(2) * tBar**1.5_sdk + c(3) / rhoBar**5)
!
!     Thermal conductivity is the sum of the three functions.  Reference value for thermal conductivity
!     is one W/m/K.
      ConductivityIAPWS = lamda0 + lamda1 + lamda2
!
      IF (ConductivityIAPWS <= 0.0_sdk) THEN
         WRITE (mess, '(2(a, es14.4))') ' Bad thermal conductivity at P = ', p, ' T = ', t
         !CALL error(2, mess)
         ConductivityIAPWS = 1.0e-10_sdk
      END IF
!
   END FUNCTION ConductivityIAPWS
!
!
   INTEGER(sik) FUNCTION FindIntervalPSat(pLocal, iLast, found)
!
      IMPLICIT NONE
!
!  Purpose: This function returns the interval in the saturation table that brackets pLocal.
!
!  Programmed by Jay Spore, ISL, Date (11/07)
!
!  Subroutine Argument Descriptions:
!
!  Input:
!      Variable Name   Variable Desription
!             pLocal   Pressure.
!              iLast   Last index used in the saturation table.
!
!  Output:
!      Variable Name   Variable Desription
!   FindIntervalPSat   Index such that pSatCond(i) <= pLocal <= pSatCond(i+1).
!              iLast   Current index used in the saturation table.
!              found   Logical if .TRUE. implies index has been found.
!
!  Variable Declarations:
!
      INTEGER(sik), INTENT(INOUT) :: iLast
      INTEGER(sik) :: i
!
      REAL(sdk), INTENT(IN) :: pLocal
!
      LOGICAL, INTENT(OUT) :: found
!
!     Initialize success logical to default which is failed.
      found = .FALSE.
!
!     Ensure that iLast is contained within array bounds for pSatCond.
      iLast = MAX(1_sik, MIN(nElements - 1_sik, iLast))
!
!     Decide whether to search up or search down.
      IF (pLocal >= pSatCond(iLast)) THEN
!
!        Search up.
         DO i = iLast, nElements - 1_sik
            IF (pLocal >= pSatCond(i) .AND. pLocal <= pSatCond(i+1)) THEN
!
!              Index found.
               found = .TRUE.
               EXIT
            END IF
         END DO
      ELSE
!
!        Search down
         iLast = MAX(iLast - 1_sik, 1_sik)
         DO i = iLast, 1_sik, -1_sik
            IF (pLocal >= pSatCond(i) .AND. pLocal <= pSatCond(i+1)) THEN
!
!              Index found.
               found = .TRUE.
               EXIT
            END IF
         END DO
      END IF
!
!     Check to determine if index has been found or not.
      IF (found) THEN
!
!        Index found.  Save it in iLast and load the output variable.
         iLast = i
         FindIntervalPSat = i
      ELSE
!
!        Index has not been found.  Try one more time looking at all pressure intervals
!        in sequence.
         DO i = 1, nElements - 1_sik
            IF (pLocal >= pSatCond(i) .AND. pLocal <= pSatCond(i+1)) THEN
!
!              Index found.  Save it in iLast and load the output variable.
               found = .TRUE.
               iLast = i
               FindIntervalPSat = i
               EXIT
            END IF
         END DO
      END IF
!
   END FUNCTION FindIntervalPSat
!
!
END MODULE EosIAPWSConductivity
