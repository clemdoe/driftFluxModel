MODULE Eos
    !
!     (c) 2002 The Pennsylvania University Applied Research Laboratory
!     This Software was developed for Purdue Subcontract Agreement 640-01812-2
!     under NRC Prime Contract no. NRC-04-97-046 (Task #2)
!
!
      USE EosData
!
      IMPLICIT NONE
!
!   Equation of State (EOS) Module.
!
      INTEGER(sik), PARAMETER, PRIVATE :: sdkx = sdk
!
!      INTEGER(sik) :: iLastLiqVisc = 1_sik  ! Last index used in the liquid viscosity interpolation.
!      INTEGER(sik) :: iLastVapVisc = 1_sik  ! Last index used in the vapor viscosity interpolation.
!      INTEGER(sik) :: iLastLiqCond = 1_sik  ! Last index used in the liquid conductivity interpolation.
!      INTEGER(sik) :: iLastVapCond = 1_sik  ! Last index used in the vapor conductivity interpolation.
!!
!!     Fit constants for D2o
!      REAL(sdk), PARAMETER, PRIVATE :: a15 = 1.553036000e-03_sdk
!      REAL(sdk), PARAMETER, PRIVATE :: c6 = 2.419016400e06_sdk, c7 =-2.764925723e10_sdk, c8 = 1.759743109e05_sdk
!      REAL(sdk), PARAMETER, PRIVATE :: c9 = 1.074572362_sdk, c10 =-1.173284768e-02_sdk, c11 =-3.824564826e-06_sdk
!      REAL(sdk), PARAMETER, PRIVATE :: a11 = 1.111753303e-03_sdk, a12 = 6.919077884e02_sdk
!      REAL(sdk), PARAMETER, PRIVATE :: a13 = 1.445279294e-03_sdk
!!
!!     Fits for H2O steam specific internal energy.
!      REAL(sdk), PARAMETER, PRIVATE, DIMENSION(11) :: ave = (/ 2.4949771766385e06_sdk, &
!     & 2.5600870370371e06_sdk, 2.5915500000006e06_sdk, 2.6606000000024e06_sdk, 3.8201600000097e06_sdk, &
!     &-1.2103411633350e08_sdk, 2.2000000000000e06_sdk, 2.2000000000000e06_sdk, 2.2000000000000e06_sdk, &
!     & 2.2000000000000e06_sdk, 2.2000000000000e06_sdk /)
!      REAL(sdk), PARAMETER, PRIVATE, DIMENSION(11) :: bve = (/ 2.0855856331827e-01_sdk, &
!     & 3.1086111111026e-02_sdk, 8.7749999997567e-03_sdk,-1.3545000000581e-02_sdk,-2.3019900000170e-01_sdk, &
!     & 1.8018803375785e+01_sdk, 0.0_sdk, 0.0_sdk, 0.0_sdk, 0.0_sdk, 0.0_sdk /)
!      REAL(sdk), PARAMETER, PRIVATE, DIMENSION(11) :: cve = (/-1.3553894579716e-07_sdk, &
!     &-6.8988888888580e-09_sdk,-1.7499999999663e-09_sdk, 6.4250000004682e-10_sdk, 1.4068900000098e-08_sdk, &
!     &-8.7442426507726e-07_sdk, 0.0_sdk, 0.0_sdk, 0.0_sdk, 0.0_sdk, 0.0_sdk /)
!      REAL(sdk), PARAMETER, PRIVATE, DIMENSION(11) :: dve = (/ 2.8522684989198e-14_sdk, &
!     & 4.3203703703379e-16_sdk, 4.2999999998503e-17_sdk,-4.2100000001248e-17_sdk,-3.1786000000187e-16_sdk, &
!     & 1.4091076856088e-14_sdk, 0.0_sdk, 0.0_sdk, 0.0_sdk, 0.0_sdk, 0.0_sdk /)
!      REAL(sdk), DIMENSION(11), PARAMETER, PRIVATE :: avg = (/ 1.0666845123419_sdk, 1.0735412407407_sdk, &
!     & 1.0777730000000_sdk, 1.0851130000007_sdk, 1.1639800000015_sdk, 3.8898867259868_sdk, &
!     & 2.7168710524682_sdk, 3.9749829999964_sdk, 1.2946929999997_sdk, 1.0590519999963_sdk, &
!     & 1.1430199999838_sdk /)
!      REAL(sdk), DIMENSION(11), PARAMETER, PRIVATE :: bvg = (/ 2.8310838172462e-08_sdk, 2.6518055555551e-09_sdk, &
!     &-2.4300000008021e-11_sdk,-1.9307000001824e-09_sdk,-1.6338350000254e-08_sdk, &
!     &-3.8595945559811e-07_sdk,-2.2832718294604e-07_sdk,-3.0657099999960e-07_sdk, &
!     &-2.4834999999979e-08_sdk,-2.4615999996941e-09_sdk,-7.7095999988588e-09_sdk /)
!      REAL(sdk), DIMENSION(11), PARAMETER, PRIVATE :: cvg = (/-2.1151097428905e-14_sdk, &
!     &-6.3461111111128e-16_sdk,-7.1979999998378e-17_sdk, 8.9100000014826e-17_sdk, &
!     & 9.5856000001448e-16_sdk, 1.7476370114910e-14_sdk, 1.0417331983836e-14_sdk, &
!     & 1.0637899999985e-14_sdk, 7.8979999999944e-16_sdk, 8.8399999991573e-17_sdk, 1.9335999997331e-16_sdk /)
!      REAL(sdk), DIMENSION(11), PARAMETER, PRIVATE :: dvg = (/ 4.7404001285964e-21_sdk, &
!     & 3.9824074074117e-23_sdk, 4.8799999990422e-25_sdk,-3.8960000003946e-24_sdk, &
!     &-2.1194000000274e-23_sdk,-2.6377008249858e-22_sdk,-1.5842822199773e-22_sdk, &
!     &-1.2257999999981e-22_sdk,-8.0799999999948e-24_sdk,-8.0799999992269e-25_sdk,-1.4639999997924e-24_sdk /)
!      REAL(sdk), DIMENSION(10), PARAMETER, PRIVATE :: acp = (/-7.9678485852270e02_sdk, &
!     &-9.7082632232795e02_sdk,-1.6649701690752e03_sdk,-6.1420486441088e03_sdk,-8.2289951961933e04_sdk, &
!     &-6.5842104212475e05_sdk, 3.4561620732510e05_sdk, 1.9798369474597e06_sdk,-9.6249385211359e07_sdk, &
!     &-1.1074934463333e07_sdk /)
!      REAL(sdk), DIMENSION(10), PARAMETER, PRIVATE :: bcp = (/ 2.8187658437259e01_sdk, 2.8324981030402e01_sdk, &
!     & 3.3159363169596e01_sdk, 6.3630987079837e01_sdk, 5.3773958896061e02_sdk, 3.7934294783212e03_sdk, &
!     &-2.2129380791446e02_sdk,-1.4782551342826e04_sdk, 4.3633668884423e05_sdk, 4.8073794630970e04_sdk /)
!      REAL(sdk), DIMENSION(10), PARAMETER, PRIVATE :: ccp = (/-1.0180624999920e-01_sdk,-9.7656200001157e-02_sdk, &
!     &-1.0861179999898e-01_sdk,-1.7762319999965e-01_sdk,-1.1612491999609_sdk   ,-7.2924928000022_sdk, &
!     &-2.4524285999925_sdk    , 3.1656481897637e+01_sdk,-6.5887615106930e02_sdk,-6.9212173247881e01_sdk /)
!      REAL(sdk), DIMENSION(10), PARAMETER, PRIVATE :: dcp = (/ 1.2499999999912e-04_sdk, &
!     & 1.1600000000110e-04_sdk, 1.2399999999915e-04_sdk, 1.7599999999975e-04_sdk, &
!     & 8.5599999997375e-04_sdk, 4.7040000000014e-03_sdk, 3.1479999999958e-03_sdk, &
!     &-2.0843356864237e-02_sdk, 3.3146147264269e-01_sdk, 3.3091693999800e-02_sdk /)
!      REAL(sdk), PARAMETER, PRIVATE :: c26 = 0.3_sdk
!      REAL(sdk), PARAMETER, PRIVATE :: a11H2o = 0.10000887519691e-02_sdk
!      REAL(sdk), PARAMETER, PRIVATE :: a12H2o = 0.76916250454393e03_sdk
!      REAL(sdk), PARAMETER, PRIVATE :: a13H2o = 0.13001153775598e-02_sdk
!!
!!     Pressure limits on transition regime between old and new
!!     formulations for superheated vapor.
!!     PLIM1    Pressure lower bound for transition region.
!!     PLIM2    Pressure upper bound for transition region. (Must be >
!!              lower bound.
!      REAL(sdk), PARAMETER, PRIVATE :: PLIM1 = 190.0e05_sdk, PLIM2 = 200.0e05_sdk
!!
!      REAL(sdk), PARAMETER, PRIVATE :: paSign = 0.1_sdk
!      REAL(sdk), PARAMETER :: rhocNa = 214.1_sdk, tlNaSwitch = 1644.26_sdk
!!
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: p => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: el => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: ev => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: tl => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: tv => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: eva => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: tsat => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: rol => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: rov => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: pa => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: alp => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: rova => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: tssn => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: dtsdp => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: deldp => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: deldt => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: devdp => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: devdt => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: drolp => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: drovp => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: drolt => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: drovt => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: hvst => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: hlst => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: dhvsp => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: dhlsp => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: dtssp => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: devat => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: devap => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: drvap => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: drvat => NULL()
!      REAL(sdk), POINTER, DIMENSION(:, :), PRIVATE :: xgn => NULL()
!      REAL(sdk), POINTER, DIMENSION(:, :), PRIVATE :: drvax => NULL()
!      REAL(sdk), POINTER, DIMENSION(:, :), PRIVATE :: devax => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: cpLiq => NULL()
!      REAL(sdk), POINTER, DIMENSION(:), PRIVATE :: cpVap => NULL()
!      LOGICAL, POINTER, DIMENSION(:), PRIVATE :: phaseChange => NULL()
!!
!      REAL(sdk), POINTER, PRIVATE :: alpt => NULL()
!!
      !PRIVATE :: Set3DEosPtrs, Set1DEosPtrs
      !PRIVATE :: ThermH, ThermD, ThermHe, ThermNa, ThermPbBi, ThermIAPWS, ThermN2, ThermAir, ThermD2oST
      !PRIVATE :: RhoLiH, RhoLiD, RhoLiHe, RhoLiNa, RhoLiPbBi, RhoLiIAPWS, RhoLiN2, RhoLiAir, RhoLiD2oST
      !PRIVATE :: SatDeH, SatDeD, SatDeHe, SatDeNa, SatDePbBi, SatDeIAPWS, SatDeN2, SatDeAir, SatDeD2oST
      !PRIVATE :: SatPrH, SatPrD, SatPrHe, SatPrNa, SatPrPbBi, SatPrIAPWS, SatPrN2, SatPrAir, SatPrD2oST
      !PRIVATE :: FPropH, FPropD, FPropHe, FPropNa, FPropPbBi, FPropIAPWS, FPropN2, FPropAir, FPropD2oST
      !PRIVATE :: SatTmH, SatTmD, SatTmHe, SatTmNa, SatTmPbBi, SatTmIAPWS, SatTmN2, SatTmAir, SatTmD2oST
      !PRIVATE :: ViscLD, ViscLHe, ViscLNa, ViscLPbBi, ViscLN2, ViscLAir
      !PRIVATE :: ViscVH, ViscVNa, ViscVPbBi, ViscVN2, ViscVAir, ViscD2oST
      !PRIVATE :: HeVH, HeVD, HeVHe, HeVNa, HeVPbBi, HeVIAPWS, HeVN2, HeVAir, HeVD2oST
      !PRIVATE :: ThcLD, ThcLHe, ThcLNa, ThcLPbBi, ThcLN2, ThcLAir
      !PRIVATE :: ThcVH, ThcVHe, ThcVNa, ThcVPbBi, ThcVN2, ThcVAir
      !PRIVATE :: SigmaHe, SigmaNa, SigmaPbBi, SigmaN2, SigmaAir
      !PRIVATE :: CpllH, CpllD, CpllHe, CpllNa, CpllPbBi, CpllIAPWS, CpllN2, CpllAir, CpllD2oST
      !PRIVATE :: CpvvH, CpvvD, CpvvHe, CpvvNa, CpvvPbBi, CpvvIAPWS, CpvvN2, CpvvAir, CpvvD2oST
      !PRIVATE :: NullEosPtrs
      !PRIVATE :: RhoVapAir, RhoVapN2, RhoVapIAPWS, RhoVapPbBi, RhoVapNa, RhoVapHe, RhoVapD, RhoVapH, RhoVapD2oST
      !PRIVATE :: CalcRhoSH2o, CalcRhoGHe, CalcRhoSD2o, CalcRhoGNa, CalcRhoGN2, CalcRhoGAir
      !PRIVATE :: CalcRhoGPbBi
!
CONTAINS

      SUBROUTINE RhoLiIAPWS(p, tl, rhol, iterate) !drldp, drldt, 
!
         USE EosIAPWSCrunch
         USE EosIAPWSData
!
         IMPLICIT NONE
!
!     subroutine RhoLiH evaluates the density of h2o liquid and its
!     derivatives with respect to total pressure and liquid temper-
!     ature as a function of total pressure and liquid temperature
!     Uses IAPWS properties
!
!          total pressure              p      in (pa)
!          liquid temperature          tl     in (k)
!          liquid density              rol    in (kg/m**3)
!          drol/dp                     drldp  in (kg/m**3/pa)
!          drol/dt                     drldt  in (kg/m**3/k)
!
         REAL(sdk), INTENT(IN) :: p, tl
         REAL(sdk), INTENT(OUT) :: rhol !, drldp, drldt
         REAL(sdk) pres, tlimited
!
         LOGICAL :: err
         LOGICAL, INTENT(OUT) :: iterate
!
         CHARACTER(LEN=120) :: errString
!
!     density and its derivatives
!
         iterate = .FALSE.
         CALL clearprops(getprops)
         getprops(vbar) = .TRUE.
         getprops(betabar) = .TRUE.
         getprops(kapabar) = .TRUE.
!
!  limit p to be greater than triple point pressure
!
         pres = MAX(p, ceoslp(36, eosH2o))
         s(presbar) = pres
         IF (pres < ceoslp(37, eosH2o)) THEN
!
!  pressure below the critical pressure
!
            tlimited = MIN(MAX(tl, ceoslp(32, eosH2o) - 0.1_sdk), ceoslp(38, eosH2o))
            s(tempbar) = tlimited
            CALL getsnglatpt(tables, liquid, getprops, tlimited, pres, s, err, iterate)
         ELSE
!
!  pressure above the critical pressure
!
             s(tempbar) = tl
             CALL getsnglatpt(tables, supercritical, getprops, tl, pres, s, err, iterate)
         END IF
         IF (err) THEN
            WRITE (errString, 800) pres, tl
800         FORMAT ('*RhoLiIAPWS* Error from getsnglatpt: pres, tl = ', 2 es13.4)
            !CALL error(1, errString)
         END IF
         rhol = 1.0_sdk / s(vbar)
         !drldp = rhol * s(kapabar)
         !drldt = - rhol * s(betabar)
!
      END SUBROUTINE RhoLiIAPWS
!
      REAL(sdkx) FUNCTION CpllIAPWS(tl, p, iterate)
!
         USE EosIAPWSCrunch
         USE EosIAPWSData
!
         IMPLICIT NONE
!
!     Function CpllIAPWS evaluates the specific heat of h2o liquid
!     as a function of liquid temperature and total pressure.
!     Uses IAPWS water properties.
!
!          liquid specific energey     tl        in (K)
!          total pressure              p         in (Pa)
!          liquid specific heat        CpllIAPWS in (J/kg/K)
!
         REAL(sdk), INTENT(IN) :: tl, p
         REAL(sdk) :: t, pres
!
         LOGICAL, INTENT(OUT) :: iterate
         LOGICAL :: err
!
         CHARACTER(LEN=120) :: errString
!
!        For the IAPWS tables, CpllIAPWS is called with tl instead of h or el.
!
         err = .FALSE.
         iterate = .FALSE.
         CALL clearprops(getprops)
         getprops(cpbar) = .TRUE.
         t = tl
         pres = MAX(p, ceoslp(36, eosH2o))
         s(presbar) = pres
         IF (p < ceoslp(37, eosH2o)) THEN
!
!           Pressure is below the critical pressure.  Temperature is not allowed to be above
!           the critical temperature for the liquid phase properties.
            t = MIN(t, tcrit - 0.1_sdk)
            s(tempbar) = t
!
!           Get single phase liquid properties.
            CALL getsnglatpt(tables, liquid, getprops, t, pres, s, err, iterate)
         ELSE
!
!           Pressure is above the critical pressure.
            s(tempbar) = t
            CALL getsnglatpt(tables, supercritical, getprops, t, pres, s, err, iterate)
         END IF
!
!        Check to determine if getsnglatpt returned an error or not.
         IF (err) THEN
            WRITE (errString, 800) t, p
800         FORMAT ('*CpllIAPWS* Error from getsnglatpu for liquid: t, p = ', 2 es13.4)
            !CALL error(1, errString)
         END IF
!
!        Save the liquid phase specific heat.
         CpllIAPWS = s(cpbar)
!
      END FUNCTION CpllIAPWS
!      
      REAL(sdkx) FUNCTION Sigma(tsat)
!
         IMPLICIT NONE
!
!     surface tension of water.
!     e. schmidt.  properties of water and steam in si units -- 1979.
!
         REAL(sdk), INTENT(IN) :: tsat
         REAL(sdk) :: tt
!
         tt = MAX((647.15_sdk-tsat)/647.15_sdk, 0.1_sdk)
         Sigma = 0.2358_sdk * star(tt, 1.256_sdk) * (1.0_sdk - 0.625_sdk * tt)
!
      CONTAINS
!
         REAL(sdk) FUNCTION star(qanty, expnt)
!
            IMPLICIT NONE
!
            REAL(sdk), INTENT(IN) :: qanty, expnt
!
            star = EXP(expnt*LOG(qanty))
!
         END FUNCTION star
!
      END FUNCTION Sigma 
      
END MODULE Eos
