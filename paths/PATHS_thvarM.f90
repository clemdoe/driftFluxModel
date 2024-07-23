!--------------------------------------------------------------------------------!
!     (c) 2007 Purdue University, PARCS Development Group, Prof. Thomas J. Downar!
!--------------------------------------------------------------------------------!
MODULE PATHS_thvarM

      USE IntrType, ONLY: sdk, sik

      IMPLICIT NONE
      
      ! Input and Output files
      CHARACTER (len=80) :: inpfile
      CHARACTER (len=80) :: outfile

      ! file name = thvarM - variable declarations for t/h calc.
      
      !Replaced PARCS variables for fixing TF, DC
      LOGICAL :: freezedc
      LOGICAL :: freezetf
      REAL(sdk) :: frozendc
      REAL(sdk) :: frozentf
      
      ! Reading variables for radial arrays
      INTEGER(sik) :: nfbxy, nrowy
      ! Used to read in each hex row and hold number of inputs
      INTEGER(sik), POINTER, DIMENSION(:,:) :: hcols
      
      ! Variables for radial channel data
      INTEGER(sik) :: nchfb, nchwr, nchby
      
      ! Integer for axial node reading
      INTEGER(sik) :: nzpl, nzlr, nzur
  
      ! relaxation for power change when coupled
      REAL(sdk) :: pow_rlx

      LOGICAL:: paths_onlyflag   ! Flag for turning on PATHS
      INTEGER(sik) :: pwr_inp    ! Indicates power input type 
      REAL(sdk):: grav                                                  !Gravity
      LOGICAL:: flagpaths                                                !Flag for PATHS convergence

      ! Chimney model      
      LOGICAL     :: PATHS_usechimney   ! logical to activate
      REAL(sdk)   :: PATHS_hzchimney    ! actual height of chimney
      INTEGER(sik) :: nchm      
      
      ! For Area Change
      LOGICAL     :: PATHS_acard
      
      ! Steam Tables
      LOGICAL:: stable  !Flag for if steam tables are used
      
      ! Subcooling fixup for polynomials
      INTEGER(sik) :: stm_opt  
      
      ! For printing extra verbosity/debugging information
!      LOGICAL:: PATHS_debug
      INTEGER(sik) :: PATHS_verbos
      INTEGER(sik) :: PATHS_debug
      INTEGER(sik) :: PATHS_prntfq      
      
      ! GEOMETRY 
      INTEGER(sik) :: chgeomid                                     ! channel geometry id
      INTEGER(sik) :: chtypeid                                     ! channel type id
      
      ! User specified conductivity for clad / fuel
      INTEGER(sik) :: num_kfuel
      REAL(sdk),POINTER, DIMENSION(:) :: dat_kfuel

      INTEGER(sik) :: num_kclad
      REAL(sdk),POINTER, DIMENSION(:) :: dat_kclad
      
      !  Orifice grouping and Orifice area
      INTEGER(sik):: PATHS_nasm                               !number of assemblies
      INTEGER(sik):: PATHS_nchan                              !number of channels
      INTEGER(sik):: PATHS_nchantype                          !number of channel types
      INTEGER(sik):: PATHS_nz                                 !number of axial nodes
      INTEGER(sik):: PATHS_nax                                !number of axial faces (PATHS_nz+1)
      INTEGER(sik):: PATHS_nfm,PATHS_ngm,PATHS_ncm            !number of fuel nodes
      INTEGER :: PATHS_rct   ! reference chan type
      INTEGER(sik), POINTER, DIMENSION(:) :: PATHS_conf      ! TH fuel configuration
      INTEGER(sik), POINTER, DIMENSION(:):: PATHS_ct_map      !(nfuelfa)        !chan type no. to neut. node n.
      INTEGER(sik), POINTER, DIMENSION(:):: PATHS_chantype    !(PATHS_nchan)    !chan type no. to chan. no.
      INTEGER(sik), POINTER, DIMENSION(:):: PATHS_nasmchan    !(PATHS_nchan)    !number of neutronic nodes in a channel
      INTEGER(sik), POINTER, DIMENSION(:,:):: PATHS_nrod        !(PATHS_nchantype)    !number of fuel rods in an assembly
!      INTEGER(sik), POINTER, DIMENSION(:):: PATHS_nrod        !(PATHS_nchantype)    !number of fuel rods in an assembly
      INTEGER(sik), POINTER, DIMENSION(:):: PATHS_nwrod       !(PATHS_nchantype)    !number of water rods in an assembly
      REAL(sdk),    POINTER, DIMENSION(:):: PATHS_hz          !(PATHS_nz)       !axial height
      REAL(sdk),    POINTER, DIMENSION(:,:):: PATHS_dhz         !(PATHS_nchantype)   !channel hydraulic diameter      
!      REAL(sdk),    POINTER, DIMENSION(:):: PATHS_dhz         !(PATHS_nchantype)   !channel hydraulic diameter
      REAL(sdk),    POINTER, DIMENSION(:):: PATHS_rrod        !(PATHS_nchantype)   !radius of fuel rod
      REAL(sdk),    POINTER, DIMENSION(:):: PATHS_rwrod       !(PATHS_nchantype)   !radius of water rod
      REAL(sdk),    POINTER, DIMENSION(:,:):: PATHS_kloss     !(PATHS_nz,PATHS_nchantype)   !channel loss coefficient
      REAL(sdk),    POINTER, DIMENSION(:,:):: PATHS_regnm     ! Aaron -- added for Liao correlation      
      REAL(sdk),    POINTER, DIMENSION(:,:):: PATHS_achan     !(PATHS_nz,PATHS_nchantype)   !cell areas
!      REAL(sdk),    POINTER, DIMENSION(:):: PATHS_achan       !(PATHS_nchantype)   !channel area
      REAL(sdk),    POINTER, DIMENSION(:,:):: PATHS_aloss     !(PATHS_nz+1,PATHS_nchantype)   
      !user-input loss coeff for abrupt area change      

      ! Flow volume
      REAL(sdk),    POINTER, DIMENSION(:,:):: PATHS_hydvol     !(PATHS_nz+1,PATHS_nchantype)   
      !user-input loss coeff for abrupt area change            
      
      REAL(sdk), POINTER, DIMENSION(:)   :: PATHS_rgh     !roughness
      REAL(sdk), POINTER, DIMENSION(:)   :: PATHS_pit     !pin pitch
      REAL(sdk), POINTER, DIMENSION(:)   :: PATHS_rf      !fuel pellet radius
      REAL(sdk), POINTER, DIMENSION(:)   :: PATHS_tc      !clad thickness
      REAL(sdk), POINTER, DIMENSION(:)   :: PATHS_kgap    !gap conductance
      ! Direct Energy      
      REAL(sdk)                          :: PATHS_fracdc,PATHS_fracdwr,PATHS_fracdvb  ! Direct energy fractions      
      ! INITIAL CONDITIONS
      REAL(sdk):: PATHS_RTP                                   !Rated Thermal Power (core power)
      REAL(sdk):: PATHS_hin                                   !Inlet enthalpy
      REAL(sdk):: PATHS_Tin                                   !Inlet enthalpy
      REAL(sdk):: PATHS_Pout                                  !Core outlet pressure
      REAL(sdk):: PATHS_mdotcore                              !Core flow rate
      REAL(sdk):: PATHS_symmmult                              !Multiplier for core flow and power for cases with symmetry
      REAL(sdk),    POINTER, DIMENSION(:,:):: PATHS_Prel      !(PATHS_nz,    PATHS_nchan)   !Relative Power
      REAL(sdk),    POINTER, DIMENSION(:,:):: PATHS_Preld     !(PATHS_nz,    PATHS_nchan)   !Old Relative Power
      ! CONTROL PARAMETERS
      !  Friction Model, void model, bypass model, matrix solver, tolorances
      INTEGER(sik):: PATHS_mvd    ! Void model
                                  ! 1: EPRI 2:GE ramp, 3:Modified Bestion,  4:Homogeneous, 5:Bestion TRACE,
                                  ! 6: Kataoka-Ishii, 7: Liao-Parlos-Griffith, 8: Zuber-Findlay      
      INTEGER(sik):: PATHS_mfr    ! Wall Friction correlation
                                  ! 1: Blasius,  2:Churchill
      INTEGER(sik):: PATHS_mtpf                                 !Two-Phase Friction Multiplier model:
      INTEGER(sik):: PATHS_mscb   ! Subcooled Boiling model
                                  ! 1:EPRI  2:Equilibrium
      INTEGER(sik):: PATHS_mhtc                                 !Heat transfer coefficient: 1 is Chen, 2 is Thom  
      REAL(sdk)   :: PATHS_thtol                                !Convergene tolerance
      REAL(sdk)   :: PATHS_thur                                 !Underrelaxation parameter
      INTEGER(sik):: PATHS_mfs   !Factor F and S option: 1 is the Chen's, 2 is the Collier's, 3 refers to RELAP
      INTEGER(sik):: PATHS_mdb   !modified Dittus-Bolter model: 1 is Ben's, 2 is Yunlin's
      INTEGER(sik):: PATHS_aopt                                 !Option for abrupt area change pressure loss calc
      
      ! Options for orifice 
      INTEGER(sik) :: PATHS_orifopt                             !Option for inputting orifice kloss 
                                                                !  factors separately from KA_CHAN card
      INTEGER(sik)                        :: PATHS_numorif      !number of different orifice types
      INTEGER(sik), POINTER, DIMENSION(:) :: PATHS_oriftype     !(PATHS_nchan)    !orif type no. to chan. no.
      REAL(sdk),    POINTER, DIMENSION(:) :: PATHS_orifloss     !(PATHS_numorif)   
                                                                !user-input loss coeff for each orifice type      
                                                                
      ! Variables for CPR calculation
      INTEGER(sik):: PATHS_cpropt                               !Option for CPR calculation
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_xcrit          !(PATHS_nax,PATHS_nchan)  !critical quality
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_cpr            !(PATHS_nax,PATHS_nchan)  !critical power ratio
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_xeq            !(PATHS_nax,PATHS_nchan)  !equilibrium quality
      REAL(sdk), POINTER, DIMENSION(:)  :: PATHS_boil_len       !(PATHS_nchan)            !boiling length estimate based on xeq                                                                
      
      ! TH Variables
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_u              !(PATHS_nax,PATHS_nchan)  !velocity
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_unew           !(PATHS_nax,PATHS_nchan)  !velocity updated
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_P              !(PATHS_nax,PATHS_nchan)  !pressure
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_enth           !(PATHS_nax,PATHS_nchan)  !enthalpy
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_dens           !(PATHS_nax,PATHS_nchan)  !density      
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_densnew        !(PATHS_nax,PATHS_nchan)  !density updated
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_qual           !(PATHS_nax,PATHS_nchan)  !quality
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_vf             !(PATHS_nax,PATHS_nchan)  !void fraction
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_vgj            !(PATHS_nax,PATHS_nchan)  !drift velocity
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_tcool          !(PATHS_nax,PATHS_nchan)  !coolant temperature
      ! Solver 
      INTEGER :: thsolver                                       ! TH linear solver
      INTEGER(sik):: PATHS_matsize                              !Matrix Size, number of equations
      INTEGER(sik):: PATHS_matnnz                               !Number of non-zero elements
      INTEGER(sik), POINTER, DIMENSION(:):: PATHS_pt            !(64)       !PARDISO internal pointer
      INTEGER(sik), POINTER, DIMENSION(:):: PATHS_iparam        !(64)       !input parameters for PARDISO
      INTEGER(sik), POINTER, DIMENSION(:):: PATHS_matia         !(PATHS_matsize+1)!column indices
      INTEGER(sik), POINTER, DIMENSION(:):: PATHS_matja         !(PATHS_matnnz)   !row indices
      REAL(sdk),    POINTER, DIMENSION(:):: PATHS_mata          !(PATHS_matnnz)   !Values of nonzero elements in the matrix A
      REAL(sdk),    POINTER, DIMENSION(:):: PATHS_matx          !(PATHS_matsize)  !Values of elements in the vector x
      REAL(sdk),    POINTER, DIMENSION(:):: PATHS_matb          !(PATHS_matsize)  !Values of elements in the vector b
      REAL(sdk)                          :: PATHS_err           ! nonlinear residual error of PATHS continuity/momentum eqns
      ! Fuel variables      
      REAL(sdk), POINTER, DIMENSION(:,:):: PATHS_tw             !(nz,nchan)  !wall temperature
      REAL(sdk), POINTER, DIMENSION(:,:,:):: PATHS_tavg         !(ncm,nz,nchan)  !fuel averaged temperature of each node
      REAL(sdk), POINTER, DIMENSION(:,:,:):: PATHS_tfuel        !(ncm+1,nz,nchan)  !fuel temperature of each mesh point  
      REAL(sdk), POINTER, DIMENSION(:) ::  keff
      REAL(sdk), POINTER, DIMENSION(:) ::  tavgo
      REAL(sdk), POINTER, DIMENSION(:) ::  rfuel

      ! Hexagonal geometry
      INTEGER(sik), POINTER, DIMENSION(:,:):: PATHS_iaass       !(-nxfc:nxfc,-nyfc:nyfc)
      INTEGER(sik), POINTER, DIMENSION(:,:):: PATHS_layh        !(-nxfc:nxfc,-nyfc:nyfc)
      INTEGER(sik), POINTER, DIMENSION(:):: PATHS_iasytype
      ! Fuel parameters added by linsen 10/12/2010  (Assuming only one type of fuel rod)  
      INTEGER(sik)::PATHS_nr
      REAL(sdk)::PATHS_pfa
      ! Iteration criteria
      REAL(sdk):: PATHS_epsout
      REAL(sdk):: PATHS_epsin
      INTEGER(sik):: PATHS_noutmax
      INTEGER(sik):: PATHS_ninmax
      
      CHARACTER(len=100) :: PATHS_depfname
END MODULE PATHS_thvarM
