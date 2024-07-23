MODULE EosData
!
!     (c) 2002 The Pennsylvania University Applied Research Laboratory
!     This Software was developed for Purdue Subcontract Agreement 640-01812-2
!     under NRC Prime Contract no. NRC-04-97-046 (Task #2)
!
      USE IntrType, ONLY: sdk, sik
!
      IMPLICIT NONE
!
!     iGas is a namelist parameter. !     It has the following definition:
!         iGas = 1 implies air
!         iGas = 2 implies hydrogen
!         iGas = 3 implies helium
!         iGas = 4 implies argon
!         iGas = 5 implies nitrogen
!         iGas = 6 implies xenon
!         iGas = 7 implies krypton
!         iGas = 8 implies non-ideal helium
!         iGas = 12-17 implies a user defined mixture of ideal gases.  The number of
!                ideal gas species included in the TRACE mixture is iGas - 10.
      INTEGER(sik), PARAMETER :: iGasDef = 1 ! set default value
      INTEGER(sik), SAVE :: iGas = iGasDef
!
!
      INTEGER(sik), PARAMETER :: maxeos = 7       ! Maximum number of different fluids allowed in a TRACE model.
!
      INTEGER(sik), PARAMETER :: eosH2o = 1
      INTEGER(sik), PARAMETER :: eosD2o = 2
      INTEGER(sik), PARAMETER :: eosHe = 3
      INTEGER(sik), PARAMETER :: eosNa = 4
      INTEGER(sik), PARAMETER :: eosPbBi = 5
      INTEGER(sik), PARAMETER :: eosN2 = 6
      INTEGER(sik), PARAMETER :: eosAir = 7
      CHARACTER(LEN=5), DIMENSION(maxeos) :: eosNames = (/ 'h2o  ', 'd2o  ', 'he   ', 'na   ', 'pbbi ', &
     &'n2   ', 'air  ' /)
!
!     ceoslp( 1, eosIndex) = 1st coeff of saturation vapor temperature function
!     ceoslp( 2, eosIndex) = 2nd coeff of saturation vapor temperature function
!     ceoslp( 3, eosIndex) = 3rd coeff of saturation vapor temperature function
!     ceoslp( 4, eosIndex) = vapor specific heat at constant volume
!     ceoslp( 5, eosIndex) = reference temperature
!     ceoslp( 6, eosIndex) = saturation vapor internal energy at reference temp
!     ceoslp( 7, eosIndex) = liquid specific heat at constant volume
!     ceoslp( 8, eosIndex) = saturation liquid internal energy at reference temp
!     ceoslp( 9, eosIndex) = microscopic density of liquid
!     ceoslp(10, eosIndex) = heat of evaporation at reference temperature
!     ceoslp(11, eosIndex) = reference pressure
!     ceoslp(12, eosIndex) = gas constant for vapor
!     ceoslp(13, eosIndex) = noncondensable gas thermal conductivity coefficient
!     ceoslp(14, eosIndex) = liquid thermal conductivity
!     ceoslp(15, eosIndex) = square of reciprocal sound speed for liquid
!     ceoslp(16, eosIndex) = gamma ratio of vapor specific heats
!     ceoslp(17, eosIndex) = noncondensable gas specific heat at constant volume
!     ceoslp(18, eosIndex) = noncondensable gas gamma, ratio of specific heats
!     ceoslp(19, eosIndex) = noncondensable gas thermal conductivity exponent
!     ceoslp(20, eosIndex) = upper limit on pressure for low-pressure properties
!     ceoslp(21, eosIndex) = upper limit on temp. for low-pressure properties
!     ceoslp(22, eosIndex) = noncondensable gas specific heat at constant pressure
!     ceoslp(23, eosIndex) = vapor specific heat at constant pressure
!     ceoslp(24, eosIndex) = liquid specific heat at constant pressure
!     ceoslp(25, eosIndex) = gas constant for noncondensable gas
!     ceoslp(26, eosIndex) = saturation liquid enthalpy at reference temperature
!     ceoslp(27, eosIndex) = saturation vapor enthalpy at reference temperature
!     ceoslp(28, eosIndex) = (vap.gas const.-noncond.gas const.)/vap.gas const.
!     ceoslp(29, eosIndex) = datum temperature for the enthalpy of liquid
!     ceoslp(30, eosIndex) = minimum allowable pressure
!     ceoslp(31, eosIndex) = maximum allowable pressure
!     ceoslp(32, eosIndex) = minimum allowable liquid temperature
!     ceoslp(33, eosIndex) = maximum allowable liquid temperature
!     ceoslp(34, eosIndex) = minimum allowable vapor temperature
!     ceoslp(35, eosIndex) = maximum allowable vapor temperature
!     ceoslp(36, eosIndex) = saturation pressure at ceosop(32) (triple-point pres)
!     ceoslp(37, eosIndex) = critical pressure (2.166e+07 pa for d2o)
!     ceoslp(38, eosIndex) = critical temperature (643.89 k for d2o)
!     ceoslp(39, eosIndex) = sat.press.lower limit for high-press.expression
!     ceoslp(40, eosIndex) = sat.temp. lower limit for high-press.expression
!     ceoslp(41, eosIndex) = minimum allowed non-condensable gas temperature.
!
      REAL(sdk) ceoslp(41, maxeos)
      REAL(sdk), DIMENSION(maxeos) :: aeos14, ceos1, ceos2, ceos3
      INTEGER(sik), SAVE :: nActEos = 0

      INTEGER(sik), DIMENSION(maxeos) :: actFluids
      CHARACTER(LEN=5), DIMENSION(maxeos) :: actNames
      CHARACTER(LEN=5), PARAMETER :: blanks5 = '     '
      CHARACTER(LEN=5), DIMENSION(maxeos) :: fluids = blanks5
      CHARACTER(LEN=5), PARAMETER :: fluidsDef = 'h2o'
!
      REAL(sdk), SAVE :: maxP = - 1.0e+10_sdk
      REAL(sdk), SAVE :: minP = 1.0e+10_sdk
      REAL(sdk), SAVE :: maxTL = - 1.0e+10_sdk
      REAL(sdk), SAVE :: minTL = 1.0e+10_sdk
      REAL(sdk), SAVE :: maxTV = - 1.0e+10_sdk
      REAL(sdk), SAVE :: minTV = 1.0e+10_sdk
!
!    Keep old input.
!
      INTEGER(sik), SAVE :: id2o = 0
!
!     Molecular weight of D2O.
!           =2.0*2.01410+15.99937  for d2o
      REAL(sdk), PARAMETER :: d2oMWt = 20.02757_sdk
!
!     Molecular weight of H2O.
!           =2*1.00797 + 15.99937  for h2o
      REAL(sdk), PARAMETER :: h2oMWt = 18.01531_sdk
!
!     Molecular weight of Na.
      REAL(sdk), PARAMETER :: naMWt = 22.9898_sdk
!
!     Molecular weight of PbBi.
      REAL(sdk), PARAMETER :: pbBiMWt = 0.44_sdk * 207.2_sdk + 0.56_sdk * 208.9808_sdk
!
!     Unverisal gas constant
!           =6.022169e+26*1.380622e-23
      REAL(sdk), PARAMETER :: gasCon = 8314.339_sdk
!
!     use_IAPWS_st is namelist parameter, set default value
!     use_IAPWS_st = true if you want to use ASME steam tables.
!     Default has been set to .FALSE. to preserve old curve fit methodology, although
!     the official guidance should be to set it to .TRUE.
      LOGICAL, PARAMETER :: use_IAPWS_stDef = .FALSE.
      LOGICAL :: use_IAPWS_st = use_IAPWS_stDef
!
!     use_D2O_st is namelist parameter, set default value
!     use_D2O_st = true if you want to use D2O steam tables.
!     Default has been set to .FALSE. to preserve old curve fit methodology.
      LOGICAL, PARAMETER :: use_D2O_stDef = .FALSE.
      LOGICAL :: use_D2O_st = use_D2O_stDef
!
!
!   CONTAINS
!
!
!      SUBROUTINE WriteFluidsAr
!!
!         USE Io
!         USE Util, ONLY: ConvertUCtoLC
!!
!         IMPLICIT NONE
!!
!!  Purpose: This routine writes out the fluids array.
!!
!!  Programmed by Jay Spore, LANL, Date (09/03)
!!
!!  Variable Declarations:
!!
!         INTEGER(sik) :: i, iBeg, iEnd, j
!!
!         DO i = 1, nActEos
!            CALL ConvertUCtoLC(fluids(i))
!         END DO
!         IF (nActEos > 1_sik) THEN
!            iBeg = 1
!            DO i = 1, nActEos, 5
!               iEnd = MIN(iBeg + 4, nActEos)
!               WRITE (iout, '(5(a,i3,2a))') (' fluids(',j,') = ', fluids(j), j = iBeg, iEnd)
!               iBeg = iBeg + 5
!            END DO
!         END IF
!!
!      END SUBROUTINE WriteFluidsAr
!
!
END MODULE EosData
