MODULE EosIAPWSData
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
!     From stcom.h
      REAL(sdk) :: ttrip    ! Triple point temperature.
      REAL(sdk) :: ptrip    ! Triple point pressure.
      REAL(sdk) :: vtrip    ! Specific volume at the triple point.
      REAL(sdk) :: tcrit    ! Critical point temperature.
      REAL(sdk) :: pcrit    ! Critical point pressure.
      REAL(sdk) :: vcrit    ! Critical point specific volume.
      REAL(sdk) :: tmin     ! Minimum temperature for the steam tables.
      REAL(sdk) :: pmin     ! Minimum pressure for the steam tables.
      REAL(sdk) :: tmax     ! Maximum temperature for the steam tables.
      REAL(sdk) :: pmax     ! Maximum pressure for the steam tables.
!
      REAL(sdk) :: s(26)                 ! Contains properties from/to steam tables.
      LOGICAL :: getprops(26)            ! If getprops(i) is .TRUE., then property is returned in s(i).
!
!     From sparms.h
      INTEGER(sik), PARAMETER :: tempbar = 1_sik    ! Temperature index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: presbar = 2_sik    ! Pressure index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: vbar = 3_sik       ! Mixture specific volume index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: ubar = 4_sik       ! Mixture specific internal energy index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: hbar = 5_sik       ! Mixture enthalpy index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: betabar = 6_sik    ! Mixture coefficient of thermal expansion index in the s and getProps.
      INTEGER(sik), PARAMETER :: kapabar = 7_sik    ! Mixture isothermal compressibility index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: cpbar = 8_sik      ! Mixture specific heat at constant pressure index in the s and
                                                    ! getProps arrays.
      INTEGER(sik), PARAMETER :: qualbar = 9_sik    ! Steam quality index in the s and getProps.
      INTEGER(sik), PARAMETER :: tsatbar = 10_sik   ! Saturation temperature index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: psatbar = 10_sik   ! Saturation pressure index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: vsubf = 11_sik     ! Liquid phase specific volume index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: vsubg = 12_sik     ! Vapor phase specific volume index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: usubf = 13_sik     ! Liquid phase specific internal energy index in the s and getProps.
      INTEGER(sik), PARAMETER :: usubg = 14_sik     ! Vapor phase specific internal energy index in the s and getProps.
      INTEGER(sik), PARAMETER :: hsubf = 15_sik     ! Liquid phase enthalpy index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: hsubg = 16_sik     ! Vapor phase enthalpy index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: betasubf = 17_sik  ! Liquid coefficient of thermal expansion index in the s and getProps.
      INTEGER(sik), PARAMETER :: betasubg = 18_sik  ! Vapor coefficient of thermal expansion index in the s and getProps.
      INTEGER(sik), PARAMETER :: kapasubf = 19_sik  ! Liquid isothermal compressibility index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: kapasubg = 20_sik  ! Vapor isothermal compressibility index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: cpsubf = 21_sik    ! Liquid phase specific heat at constant pressure index in the
                                                    ! s and getProps arrays.
      INTEGER(sik), PARAMETER :: cpsubg = 22_sik    ! Vapor phase specific heat at constant pressure index in the
                                                    ! s and getProps arrays.
      INTEGER(sik), PARAMETER :: entpybar = 24_sik  ! Mixture entropy index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: entpysubf = 25_sik ! Liquid phase entropy index in the s and getProps arrays.
      INTEGER(sik), PARAMETER :: entpysubg = 26_sik ! Vapor phase entropy index in the s and getProps arrays.
!
      CHARACTER(LEN=80), DIMENSION(2) :: newrecord  ! Header information concerning when the generation of the steam tables.
      REAL(sdk), DIMENSION(:), ALLOCATABLE :: tables   ! Container array for the steam tables.
!
!     From gibbpnt.h
!     Sub-tables in tables.
!     Temperature table.
      INTEGER(sik) :: ptable1  ! Zero-point index for temperature table contained within tables.
      INTEGER(sik) :: ltable1  ! Number of values in this table = ntemp words long.
      INTEGER(sik) :: ntable1  ! Number of entries per temperature in this table.
      INTEGER(sik) :: stable1
      INTEGER(sik) :: lastTInd = 1_sik    ! Last index used in the temperatue table.
      INTEGER(sik) :: lastTSatInd = 1_sik ! Last tsat index used in the temperatue table.
!
!     Pressuretable.
      INTEGER(sik) :: ptable2  ! Zero-point index for pressure table contained within tables.
      INTEGER(sik) :: ltable2  ! Number of values in this table = npres words long.
      INTEGER(sik) :: ntable2  ! Number of values per pressure in this table.
      INTEGER(sik) :: stable2
      INTEGER(sik) :: lastPInd = 1_sik    ! Last index used in the pressure table.
      INTEGER(sik) :: lastPLiqInd = 1_sik ! Last p liquid index used in the pressure table.
      INTEGER(sik) :: lastPSatInd = 1_sik ! Last psat index used in the pressure table.
      INTEGER(sik) :: lastPSupInd = 1_sik ! Last p super critical index used in the pressure table.
      INTEGER(sik) :: lastPVapInd = 1_sik ! Last p vapor index used in the pressure table.
!
!     Saturation properties at each temperature table.
      INTEGER(sik) :: ptable3  ! Zero-point index for table3 contained within tables.
      INTEGER(sik) :: ltable3  ! Number of values in this table.
      INTEGER(sik) :: ntable3  ! Number of values per temperature in this table.
      INTEGER(sik) :: stable3
!
!     Saturation properties at each pressure table.
      INTEGER(sik) :: ptable4  ! Zero-point index for table4 contained within tables.
      INTEGER(sik) :: ltable4  ! Number of values in this table.
      INTEGER(sik) :: ntable4  ! Number of values per pressure in this table.
      INTEGER(sik) :: stable4
!
!     Single phase liquid properties table.
      INTEGER(sik) :: ptable5  ! Zero-point index for table5 contained within tables.
      INTEGER(sik) :: ltable5  ! Number of values in this table.
      INTEGER(sik) :: ntable5  ! Number of values per pressure-temperature point in this table.
      INTEGER(sik) :: stable5
!
!     Single-phase vapor properties table.
      INTEGER(sik) :: ptable6  ! Zero-point index for table6 contained within tables.
      INTEGER(sik) :: ltable6  ! Number of values in this table.
      INTEGER(sik) :: ntable6  ! Number of values per pressure-temperature point in this table.
      INTEGER(sik) :: stable6
!
!     Single-phase supercritical pressure properties table.
      INTEGER(sik) :: ptable7  ! Zero-point index for table7 contained within tables.
      INTEGER(sik) :: ltable7  ! Number of values in this table.
      INTEGER(sik) :: ntable7  ! Number of values per pressure-temperature point in this table.
      INTEGER(sik) :: stable7
!
!     Valid liquid table 5 limits table.
      INTEGER(sik) :: ptable8  ! Zero-point index for table8 contained within tables.
      INTEGER(sik) :: ltable8  ! Number of values in this table.
      INTEGER(sik) :: ntable8  ! Number of values per pressure-temperature point in this table.
      INTEGER(sik) :: stable8
!
!     Valid vapor table 6 limits table.
      INTEGER(sik) :: ptable9  ! Zero-point index for table9 contained within tables.
      INTEGER(sik) :: ltable9  ! Number of values in this table.
      INTEGER(sik) :: ntable9  ! Number of values per pressure-temperature point in this table.
      INTEGER(sik) :: stable9
!
!     Valid supercritical pressure table 7 limits table.
      INTEGER(sik) :: ptable10  ! Zero-point index for table10 contained within tables.
      INTEGER(sik) :: ltable10  ! Number of values in this table.
      INTEGER(sik) :: ntable10  ! Number of values per pressure-temperature point in this table.
      INTEGER(sik) :: stable10
!
!     From newstcom.h
      INTEGER(sik) :: ntemp          ! Number of temperatures.
      INTEGER(sik) :: npres          ! Number of pressures.
      INTEGER(sik) :: nsubcrittemp
      INTEGER(sik) :: nsupcrittemp
      INTEGER(sik) :: nsattemp       ! Number of saturation temperatures between the triple point
                                     ! and the critical point.
      INTEGER(sik) :: ofirstsattemp
      INTEGER(sik) :: olastsattemp
      INTEGER(sik) :: otemptrip1
      INTEGER(sik) :: otempcrit1
      INTEGER(sik) :: nsubcritpres
      INTEGER(sik) :: nsupcritpres   ! Number of pressure points between the critical point and the max. pressure.
      INTEGER(sik) :: nsatpres       ! Number of saturation pressures between the triple point
                                     ! and the critical point.
      INTEGER(sik) :: ofirstsatpres
      INTEGER(sik) :: olastsatpres
      INTEGER(sik) :: oprestrip2
      INTEGER(sik) :: oprescrit2
      INTEGER(sik) :: ofirstsupcritpres
      INTEGER(sik) :: olastsupcritpres
!
!     Fluid state flags used in EOS package.
      INTEGER(sik), PARAMETER :: liquid = 1_sik
      INTEGER(sik), PARAMETER :: vapor = 2_sik
      INTEGER(sik), PARAMETER :: supercritical = 3_sik
!
!     Gibbs function and derivatives.
      REAL(sdk), DIMENSION(11) :: g
!     g(1) - gibbs function.
!     g(2) - dgtildedp - specific volume - derivative of gibbs function with respect to pressure.
!     g(3) - dtgtilededp2 - second order derivative of gibbs function with respect to pressure.
!     g(4) - dgtildedt - minus entropy - derivative of gibbs function with respect to temperature.
!     g(5) - d2gtiledpdt - second order derivative of gibbs function with respect to pressure and temperature.
!     g(6) - ?
!     g(7) - d2gtildedt2 - second order derivative of gibbs function with respect to temperature.
!     g(8:11) - ?
!
CONTAINS
!
      SUBROUTINE AllocIAPWSTables(ntot, nuse)
!
!       Routine to allocate or reallocate if necessary the
!       H2O tables.
!
         USE Io, ONLY: outUnt

!
         IMPLICIT NONE
!
         INTEGER(sik), INTENT(IN) :: ntot
         INTEGER(sik), INTENT(OUT) :: nuse
         INTEGER(sik) :: ios = 0, i
!
         REAL(sdk), DIMENSION(:), ALLOCATABLE :: tmpTab
!
!        Test to see if tables has already been allocated.
!
         IF (ALLOCATED(tables)) THEN
!
!           If tables has already been allocated then determine the size
!           see if it needs to be bigger.
!
            nuse = SIZE(tables)
            IF (ntot > nuse) THEN
!
!             Tables must be de-allocated and re-allocated to ntot.
!             First allocate temporary array which will hold the current
!             contents of tables.
!
               ALLOCATE(tmpTab(nuse), STAT=ios)
               IF (ios /= 0) THEN
                  DO i = 1, SIZE(outUnt)
                     WRITE (outUnt(i), '(a)') '*AllocIAPWSTables* error allocation of tmpTab'
                  END DO
                  nuse = - 1
                  GO TO 999
               END IF
!
!             Save current contents of tables.
!
               tmpTab = tables
!
!             De-allocate tables and reallocate to a size of ntot.
!
               DEALLOCATE(tables, STAT=ios)
               IF (ios /= 0) THEN
                  DO i = 1, SIZE(outUnt)
                     WRITE (outUnt(i), '(a)') '*AllocIAPWSTables* error deallocation of tables'
                  END DO
                  nuse = - 1
                  GO TO 999
               END IF
               ALLOCATE(tables(ntot), STAT=ios)
!
!             Load tables with the saved contents and de-allocate temporary array.
!
               tables(1:nuse) = tmpTab(1:nuse)
               DEALLOCATE(tmpTab)
            END IF
         ELSE
!
!           First time to allocate tables.  Allocate it to ntot.
!
            ALLOCATE(tables(ntot), STAT=ios)
         END IF
         IF (ios /= 0) THEN
            DO i = 1, SIZE(outUnt)
               WRITE (outUnt(i), '(a)') '*AllocIAPWSTables* error in allocation of tables'
            END DO
            nuse = - 1
         ELSE
!
!           If everything is OK return nuse equal to ntot.
!
            nuse = ntot
         END IF
999      CONTINUE
         RETURN
!
      END SUBROUTINE AllocIAPWSTables
!
!
      SUBROUTINE SetCritPointIAPWS(tcrit, pcrit, vcrit)
!
         USE IntrType, ONLY: sdk, sik
!         USE Io
!
         IMPLICIT NONE
!
!  Purpose: This routine was loads pcrit and tcrit for the h2o steam tables, which are needed
!   by the viscosity and thermal conductivity functions for h2o.
!
!  Programmed by Jay Spore, ISL, Date (11/10)
!
!  Subroutine Argument Descriptions:
!
!  Input:
!      Variable Name   Variable Desription
!              tcrit   Critical temperature for IAPWS steam tables.
!              pcrit   Critical pressure for IAPWS steam tables.
!              vcrit   Critical specific volume for IAPWS steam tables.
!
!  Variable Declarations:
!
         REAL(sdk), INTENT(OUT) :: tcrit, pcrit, vcrit
!
         tcrit =   6.470960000000000E+02_sdk
         pcrit =   2.206400000000000E+07_sdk
         vcrit =   3.105590062111801E-03_sdk
!
      END SUBROUTINE SetCritPointIAPWS
!
END MODULE EosIAPWSData
