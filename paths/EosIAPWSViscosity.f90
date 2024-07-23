MODULE EosIAPWSViscosity
!
!     (c) 2002 The Pennsylvania University Applied Research Laboratory
!     This Software was developed for Purdue Subcontract Agreement 640-01812-2
!     under NCR Prime Contract no. NRC-04-97-046 (Task #2)
!
!  Purpose: This module contains functions to calculate the IAPWS thermal viscosity of water and steam.
!
!  Programmed by Jay Spore, ISL, Date (11/07)
!
   USE IntrType, ONLY: sdk, sik
!
   IMPLICIT NONE
!
!  Reference: The International Association for the Properties of Water and Steam, Vejle, Denmark, August 2003.
!             Revised Release on the IAPS Formulation 1985 for the Viscosity of Ordinary Water Substance.
!
!  Number of points in the saturationed viscosity table.
   INTEGER(sik), PARAMETER, PRIVATE :: nElements = 41
   REAL(sdk),PRIVATE,PARAMETER :: one=1.0_sdk,zero=0.0_sdk
!
! pSatVis (Pa) sat pressure for saturation viscosity liquid and vapor phase.
   REAL(sdk), PARAMETER, DIMENSION(nElements), PRIVATE :: pSatVis = &
  &(/0.0006117e+06_sdk, 0.001228e+06_sdk, 0.002339e+06_sdk, 0.004247e+06_sdk, 0.007385e+06_sdk, &
  &  0.01235e+06_sdk,   0.01995e+06_sdk,  0.0312e+06_sdk,   0.04741e+06_sdk,  0.07018e+06_sdk, &
  &  0.10142e+06_sdk,   0.14338e+06_sdk,  0.19867e+06_sdk,  0.27028e+06_sdk,  0.36154e+06_sdk, &
  &  0.47616e+06_sdk,   0.61823e+06_sdk,  0.79219e+06_sdk,  1.0028e+06_sdk,   1.2552e+06_sdk, &
  &  1.5549e+06_sdk,    1.9077e+06_sdk,   2.3196e+06_sdk,   2.7971e+06_sdk,   3.3469e+06_sdk, &
  &  3.9762e+06_sdk,    4.6923e+06_sdk,   5.503e+06_sdk,    6.4166e+06_sdk,   7.4418e+06_sdk, &
  &  8.5879e+06_sdk,    9.8651e+06_sdk,  11.284e+06_sdk,   12.858e+06_sdk,   14.601e+06_sdk, &
  & 16.529e+06_sdk,    18.666e+06_sdk,   21.044e+06_sdk,   21.297e+06_sdk,   21.554e+06_sdk, &
  & 21.814e+06_sdk/)
!
   PRIVATE :: VisLiqSat, VisVapSat, TSatOfPVis, ViscosityIAPWS, FindIntervalPSat
!
CONTAINS
!
!
   REAL(sdk) FUNCTION VisLiqSat(p, iLast)
!
      IMPLICIT NONE
!
!  Purpose: This function returns viscosity of saturated liquid water at pressure p.
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
!          VisLiqSat   Viscosity of the saturated liquid phase.
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
! visfSat (Pa-s) saturation liquid phase viscosity.
      REAL(sdk), PARAMETER, DIMENSION(nElements) :: visfSat = &
     &        (/1791.0e-06_sdk , 1306.0e-06_sdk , 1002.0e-06_sdk , 797.4e-06_sdk , 653.0e-06_sdk, &
     &           546.8e-06_sdk ,  466.4e-06_sdk ,  403.9e-06_sdk , 354.3e-06_sdk , 314.4e-06_sdk, &
     &           281.7e-06_sdk ,  254.7e-06_sdk ,  232.1e-06_sdk , 212.9e-06_sdk , 196.5e-06_sdk, &
     &           182.5e-06_sdk ,  170.2e-06_sdk ,  159.6e-06_sdk , 150.1e-06_sdk , 141.8e-06_sdk, &
     &           134.3e-06_sdk ,  127.6e-06_sdk ,  121.5e-06_sdk , 116.0e-06_sdk , 110.9e-06_sdk, &
     &           106.1e-06_sdk ,  101.7e-06_sdk ,   97.5e-06_sdk ,  93.51e-06_sdk,  89.66e-06_sdk, &
     &            85.9e-06_sdk ,   82.17e-06_sdk,   78.41e-06_sdk,  74.54e-06_sdk,  70.43e-06_sdk, &
     &            65.88e-06_sdk,   60.33e-06_sdk,   52.07e-06_sdk,  50.75e-06_sdk,  49.07e-06_sdk, &
     &            47.81e-06_sdk/)
!
!     Limit the pressure to range in the saturation table.
      pLocal = MAX(MIN(p, pSatVis(nElements)), pSatVis(1))
!
!     Find the two points that bracket pLocal.
      i = FindIntervalPSat(pLocal, iLast, found)
      IF (found) THEN
!
!        Interpolation in table
         VisLiqSat = visfSat(i) + (visfSat(i+1) - visfSat(i)) * (pLocal - pSatVis(i)) / &
        & (pSatVis(i+1) - pSatVis(i))
      ELSE
         !CALL error(1, ' *VisVapSat* FindIntervalPSat failed.')
      END IF
!
   END FUNCTION VisLiqSat
!
!
   REAL(sdk) FUNCTION VisVapSat(p, iLast)
!
      IMPLICIT NONE
!
!  Purpose: This function returns Viscosity of saturated steam at pressure p.
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
!          VisVapSat   Viscosity of the saturated vapor phase.
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
! visgSat (Pa-s) saturation vapor phase viscosity.
      REAL(sdk), PARAMETER, DIMENSION(nElements) :: visgSat = &
     &           (/9.22e-06_sdk,  9.46e-06_sdk,  9.73e-06_sdk, 10.01e-06_sdk, 10.31e-06_sdk, &
     &            10.62e-06_sdk, 10.93e-06_sdk, 11.26e-06_sdk, 11.59e-06_sdk, 11.93e-06_sdk, &
     &            12.27e-06_sdk, 12.61e-06_sdk, 12.96e-06_sdk, 13.3e-06_sdk , 13.65e-06_sdk, &
     &            13.99e-06_sdk, 14.34e-06_sdk, 14.68e-06_sdk, 15.03e-06_sdk, 15.37e-06_sdk, &
     &            15.71e-06_sdk, 16.06e-06_sdk, 16.41e-06_sdk, 16.76e-06_sdk, 17.12e-06_sdk, &
     &            17.49e-06_sdk, 17.88e-06_sdk, 18.28e-06_sdk, 18.7e-06_sdk , 19.15e-06_sdk, &
     &            19.65e-06_sdk, 20.21e-06_sdk, 20.85e-06_sdk, 21.61e-06_sdk, 22.55e-06_sdk, &
     &            23.82e-06_sdk, 25.72e-06_sdk, 29.68e-06_sdk, 30.48e-06_sdk, 31.53e-06_sdk, &
     &            33.71e-06_sdk/)
!
!     Limit the pressure to range in the saturation table.
      pLocal = MAX(MIN(p, pSatVis(nElements)), pSatVis(1))
!
!     Find the two points that bracket pLocal.
      i = FindIntervalPSat(pLocal, iLast, found)
      IF (found) THEN
!
!        Interpolation in table
         VisVapSat = visgSat(i) + (visgSat(i+1) - visgSat(i)) * (pLocal - pSatVis(i)) / &
        & (pSatVis(i+1) - pSatVis(i))
      ELSE
         !CALL error(1, ' *VisVapSat* FindIntervalPSat failed.')
         WRITE(*,*) 'error'
      END IF
!
   END FUNCTION VisVapSat
!
!
   REAL(sdk) FUNCTION TSatOfPVis(p, iLast)
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
!         TSatOfPVis   tSat given pSat.
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
! tSatVis (C) sat temperature for saturation viscosity liquid and vapor phase.
      REAL(sdk), PARAMETER, DIMENSION(nElements) :: tSatVis = &
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
      pLocal = MAX(MIN(p, pSatVis(nElements)), pSatVis(1))
!
!     Determine whether to search up or down from iLast.
      i = FindIntervalPSat(pLocal, iLast, found)
!
!     Check to see if two-points that bracket pLocal have been found.
      IF (found) THEN
!
!        Interpolation in table.
         TSatOfPVis = tSatVis(i) + (tSatVis(i+1) - tSatVis(i)) * (pLocal - pSatVis(i)) / &
        & (pSatVis(i+1) - pSatVis(i)) + 273.15_sdk
      ELSE
         !CALL error(1, ' *TsatOfPVis* FindIntervalPSat failed.')
      END IF
!
   END FUNCTION TSatOfPVis
!
!
   REAL(sdk) FUNCTION ViscosityVap(p,iLast) ! t, rho, dRhodP, iLast)
!
      !USE EosIAPWSData, ONLY: tCrit, pCrit
!
      IMPLICIT NONE
!
!  Purpose: This function returns water vapor viscosity based on IAPWS standard.
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
!             dRhodP   Derivative of density w.r.t. change in pressure along a line of constant temperature.
!              iLast   Last index used in the saturation table.
!
!  Output:
!      Variable Name   Variable Desription
!       ViscosityVap   Vapor viscosity at p, t, rho, and dRhodP.
!              iLast   Current index used in the saturation table.
!
!  Variable Declarations:
!
      INTEGER(sik), INTENT(INOUT) :: iLast
!
      REAL(sdk), INTENT(IN) :: p !, t, rho, dRhodP
!
!!     Check to see if pressure and temperature are below the critical point.
!      IF (p < pCrit .AND. t < tCrit) THEN
!!
!!        Pressure is below the critical pressure.
!!        Check to see if vapor is saturated or subcooled.
!         IF (t <= TSatOfPVis(p, iLast)) THEN
!!
!!           Vapor is saturated or subcooled, so use saturated vapor viscosity.
!            ViscosityVap = VisVapSat(p, iLast)
!         ELSE
!!
!!           Vapor is super-heated use IAPWS fit.
!            ViscosityVap = ViscosityIAPWS(rho, p, t, dRhodP)
!         END IF
!      ELSE
!!
!!        Vapor is super-critical use IAPWS fit.
!         ViscosityVap = ViscosityIAPWS(rho, p, t, dRhodP)
!      END IF
        ViscosityVap=VisVapSat(p,iLast)
!
   END FUNCTION ViscosityVap
!
!
   REAL(sdk) FUNCTION ViscosityLiq(p, t, rho, dRhodP, iLast)
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
!             dRhodP   Derivative of density w.r.t. change in pressure along a line of constant temperature.
!              iLast   Last index used in the saturation table.
!
!  Output:
!      Variable Name   Variable Desription
!       ViscosityLiq   Liquid viscosity at p, t, rho, and dRhodP.
!              iLast   Current index used in the saturation table.
!
!  Variable Declarations:
!
      INTEGER(sik), INTENT(INOUT) :: iLast
!
      REAL(sdk), INTENT(IN) :: p, t, rho, dRhodP
!
!     Check if pressure and temperature are below the critical point.
      IF (p < pCrit .AND. t < tCrit) THEN
!
!        Check to see if liquid is saturated or super-heated.
         IF (t >= TSatOfPVis(p, iLast)) THEN
!
!           Liquid is saturated or super-heated so use saturated liquid viscosity.
            ViscosityLiq = VisLiqSat(p, iLast)
         ELSE IF (t <= 273.16_sdk) THEN
!
!           Liquid temperature is below the triple point for water use saturated
!           liquid viscosity.
            ViscosityLiq = VisLiqSat(p, iLast)
         ELSE
!
!           Liquid is subcooled use IAPWS fit for liquid phase viscosity.
            ViscosityLiq = ViscosityIAPWS(rho, p, t, dRhodP)
         END IF
      ELSE
!
!        Liquid is super-critical use IAPWS fit for liquid phase viscosity.
         ViscosityLiq = ViscosityIAPWS(rho, p, t, dRhodP)
      END IF
!
   END FUNCTION ViscosityLiq
!
!
   REAL(sdk) FUNCTION ViscosityIAPWS(rho, p, t, dRhodP)
!
      !USE GlobalDat, ONLY: one
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
!                  p   Pressure.
!                  t   Temperature.
!                rho   Density.
!             dRhodP   Derivative of density w.r.t. change in pressure along a line of constant temperature.
!
!  Output:
!      Variable Name   Variable Desription
!     ViscosityIAPWS   Viscosity at t, rho and dRhodP.
!
!  Variable Declarations:
!
      REAL(sdk), INTENT(IN) :: rho, t, dRhodP, p
      REAL(sdk) :: rhoBar, tBar, mu0, mu1, mu2, x, y, chiBar, tMax
      REAL(sdk), PARAMETER :: tRef = 647.226_sdk              ! Reference temperature for fit.
      REAL(sdk), PARAMETER :: rhoRef = 317.763_sdk            ! Reference density for fit.
      REAL(sdk), PARAMETER :: muRef = 55.071e-06_sdk          ! Reference viscosity for fit.
      REAL(sdk), PARAMETER :: pRef = 22.115e+06_sdk           ! Reference pressure for fit.
      REAL(sdk), PARAMETER :: tBarMin = 273.15_sdk / tRef     ! Lower limit for validity of fit.
      REAL(sdk), PARAMETER :: tMax1 = 150.0_sdk + 273.15_sdk  ! Upper limit for validity of fit for P <= 500 MPa.
      REAL(sdk), PARAMETER :: tMax2 = 600.0_sdk + 273.15_sdk  ! Upper limit for validity of fit for P <= 350 MPa.
      REAL(sdk), PARAMETER :: tMax3 = 900.0_sdk + 273.15_sdk  ! Upper limit for validity of fit for P <= 300 MPa.
      REAL(sdk), PARAMETER :: rhoBarMax = 10.0_sdk            ! In meta-stable state or below triple point
                                                              ! liquid phase density can become large.  Arbitrary
                                                              ! upper limit that keeps viscosity from going to zero.
!
!     Fit coefficients.
      REAL(sdk), PARAMETER, DIMENSION(0:3) :: h = (/1.0_sdk, 0.978197_sdk, 0.579829_sdk, -0.202354_sdk/)
      REAL(sdk), PARAMETER :: h00 = 0.5132047_sdk
      REAL(sdk), PARAMETEr :: h10 = 0.3205656_sdk
      REAL(sdk), PARAMETER :: h40 =-0.7782567_sdk
      REAL(sdk), PARAMETER :: h50 = 0.1885447_sdk
      REAL(sdk), PARAMETER :: h01 = 0.2151778_sdk
      REAL(sdk), PARAMETER :: h11 = 0.7317883_sdk
      REAL(sdk), PARAMETER :: h21 = 1.241044_sdk
      REAL(sdk), PARAMETER :: h31 = 1.476783_sdk
      REAL(sdk), PARAMETER :: h02 =-0.2818107_sdk
      REAL(sdk), PARAMETER :: h12 =-1.070786_sdk
      REAL(sdk), PARAMETER :: h22 =-1.263184_sdk
      REAL(sdk), PARAMETER :: h03 = 0.1778064_sdk
      REAL(sdk), PARAMETER :: h13 = 0.460504_sdk
      REAL(sdk), PARAMETER :: h23 = 0.2340379_sdk
      REAL(sdk), PARAMETER :: h33 =-0.4924179_sdk
      REAL(sdk), PARAMETER :: h04 =-0.0417661_sdk
      REAL(sdk), PARAMETER :: h34 = 0.1600435_sdk
      REAL(sdk), PARAMETER :: h15 =-0.01578386_sdk
      REAL(sdk), PARAMETER :: h36 =-0.003629481_sdk
!
      CHARACTER(LEN=130) :: mess
!
!     Set the upper limit for the fit.
      IF (P <= 300.0e+06_sdk) THEN
         tMax = tMax3
      ELSE IF (p <= 350.0e+06_sdk) THEN
         tMax = tMax2
      ELSE
         tMax = tMax1
      END IF
!
!     Calculate non-dimensional density and temperature.
      rhoBar = MIN(rho / rhoRef, rhoBarMax)
      tbar = MAX(MIN(t, tMax) / tRef, tBarMin)
!
!     Ideal gas limit.
      mu0 = SQRT(tbar) / (h(0) + (h(1) + (h(2) + h(3) / tbar) / tbar) / tbar)
!
!     Density and temperature function.
      x = one / tBar - one
      y = rhobar - one
      mu1 = h00 + x * (h10 +                      x**3 * (h40 + x * h50)) + &
       y * (h01 + x * (h11 + x * (h21 + x * h31))  + &
       y * (h02 + x * (h12 + x *  h22) + &
       y * (h03 + x * (h13 + x * (h23 + x * h33)) + &
       y * (h04 +                    x**3 * h34 + &
       y * (      x *  h15 + &
       y * (                         x**3 * h36))))))
      mu1 = EXP(rhoBar * mu1)
!
!     This term is only important close to the critical point.
      IF (0.996_sdk <= tBar .AND. tBar <= 1.01_sdk .AND. 0.71_sdk <= rhoBar .AND. rhoBar <= 1.36_sdk) THEN
         chiBar = rhoBar * dRhodP * pRef / rhoRef
         IF (chiBar >= 21.93_sdk) THEN
            mu2 = 0.922_sdk * chiBar**0.0263_sdk
         ELSE
            mu2 = one
         END IF
      ELSE
         mu2 = one
      ENDIF
!
!     Viscosity is the product of the three terms and must be multiplied times muRef in order to
!     get to Pa-s units.
      ViscosityIAPWS = mu0 * mu1 * mu2 * muRef
!
      IF (ViscosityIAPWS <= 0.0_sdk) THEN
         WRITE (mess, '(3(a, es14.4))') ' Bad viscosity = ', ViscosityIAPWS, ' at P = ', p, ' T = ', t
         !CALL error(2, mess)
         ViscosityIAPWS = 1.0e-10_sdk
      END IF
!
   END FUNCTION ViscosityIAPWS
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
      IF (pLocal >= pSatVis(iLast)) THEN
!
!        Search up.
         DO i = iLast, nElements - 1_sik
            IF (pLocal >= pSatVis(i) .AND. pLocal <= pSatVis(i+1)) THEN
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
            IF (pLocal >= pSatVis(i) .AND. pLocal <= pSatVis(i+1)) THEN
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
            IF (pLocal >= pSatVis(i) .AND. pLocal <= pSatVis(i+1)) THEN
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
END MODULE EosIAPWSViscosity
